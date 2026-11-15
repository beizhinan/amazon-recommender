"""NCF（NeuMF）用于显式评分预测：GMF + MLP 双分支，MSE 回归。

注意：经典 NeuMF 面向隐式反馈排序；这里用于评分预测，输出连续评分。
"""
from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn

from .common import RatingData, log


class _NeuMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, gmf_dim: int, mlp_dim: int,
                 layers: tuple[int, ...], mu: float) -> None:
        super().__init__()
        self.user_gmf = nn.Embedding(n_users, gmf_dim)
        self.item_gmf = nn.Embedding(n_items, gmf_dim)
        self.user_mlp = nn.Embedding(n_users, mlp_dim)
        self.item_mlp = nn.Embedding(n_items, mlp_dim)

        mlp_layers: list[nn.Module] = []
        in_dim = 2 * mlp_dim
        for out_dim in layers:
            mlp_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        self.out = nn.Linear(gmf_dim + layers[-1], 1)
        self.global_bias = nn.Parameter(torch.tensor(float(mu)))
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf = self.user_gmf(users) * self.item_gmf(items)
        mlp_in = torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=-1)
        h = self.mlp(mlp_in)
        x = torch.cat([gmf, h], dim=-1)
        return self.out(x).squeeze(-1) + self.global_bias


class NCF:
    name = "NCF"

    def __init__(self, gmf_dim: int = 32, mlp_dim: int = 32,
                 layers: tuple[int, ...] = (64, 32, 16),
                 lr: float = 0.001, batch_size: int = 8192, epochs: int = 10,
                 weight_decay: float = 1e-5, val_ratio: float = 0.05,
                 seed: int = 42) -> None:
        self.gmf_dim = gmf_dim
        self.mlp_dim = mlp_dim
        self.layers = tuple(layers)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.val_ratio = val_ratio
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _NeuMF | None = None
        self.best_epoch = 0
        self.history: list[dict] = []

    def fit(self, data: RatingData) -> "NCF":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        n = len(data.r_tr)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(self.val_ratio * n))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        dev = self.device
        u_all = torch.as_tensor(data.u_tr, dtype=torch.long, device=dev)
        i_all = torch.as_tensor(data.i_tr, dtype=torch.long, device=dev)
        r_all = torch.as_tensor(data.r_tr, dtype=torch.float32, device=dev)

        u_val, i_val = u_all[val_idx], i_all[val_idx]
        r_val = r_all[val_idx]
        u_tr, i_tr, r_tr = u_all[tr_idx], i_all[tr_idx], r_all[tr_idx]

        mu = float(r_tr.mean().item())
        log(f"[NCF] 开始训练 device={self.device} epochs={self.epochs} "
            f"gmf_dim={self.gmf_dim} mlp_dim={self.mlp_dim} batch={self.batch_size}")
        self.model = _NeuMF(data.n_users, data.n_items, self.gmf_dim, self.mlp_dim,
                            self.layers, mu).to(dev)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_rmse = float("inf")
        best_state = None
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            order = torch.randperm(len(u_tr), device=dev)
            total_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                bidx = order[start:start + self.batch_size]
                optimizer.zero_grad()
                pred = self.model(u_tr[bidx], i_tr[bidx])
                loss = loss_fn(pred, r_tr[bidx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bidx)

            train_rmse = float(np.sqrt(total_loss / len(u_tr)))
            val_rmse = self._eval_rmse(u_val, i_val, r_val)
            self.history.append({"epoch": epoch, "train_rmse": round(train_rmse, 4),
                                 "val_rmse": round(val_rmse, 4)})
            is_best = val_rmse < best_rmse
            if is_best:
                best_rmse = val_rmse
                self.best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in
                              self.model.state_dict().items()}
            log(f"[NCF] epoch {epoch:2d}/{self.epochs} train_rmse={train_rmse:.4f} "
                f"val_rmse={val_rmse:.4f}{' *best' if is_best else ''}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        log(f"[NCF] 训练完成 best_epoch={self.best_epoch} val_rmse={best_rmse:.4f}")
        return self

    @torch.no_grad()
    def _eval_rmse(self, u, i, r) -> float:
        self.model.eval()
        preds = []
        for start in range(0, len(u), self.batch_size * 2):
            preds.append(self.model(u[start:start + self.batch_size * 2],
                                    i[start:start + self.batch_size * 2]))
        pred = torch.cat(preds)
        return float(torch.sqrt(torch.mean((pred - r) ** 2)).item())

    @torch.no_grad()
    def predict(self, data: RatingData) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        dev = self.device
        u = torch.as_tensor(data.u_te, dtype=torch.long, device=dev)
        i = torch.as_tensor(data.i_te, dtype=torch.long, device=dev)
        out = []
        for start in range(0, len(u), self.batch_size * 2):
            out.append(self.model(u[start:start + self.batch_size * 2],
                                  i[start:start + self.batch_size * 2]))
        pred = torch.cat(out).cpu().numpy().astype(np.float64)
        fallback = np.zeros(len(pred), dtype=bool)
        return pred, fallback

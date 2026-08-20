"""The translator, held fixed across every arm.

Two of them, and the difference between them is the capacity control:

``mlp``    UGRN's projector — two hidden layers 128 / 64, ReLU, scalar output,
           Adam at 1e-3 — with a linear output and a weighted mean-squared error
           on the residual effect instead of a sigmoid and a binary
           cross-entropy, because the target here is a measured continuous
           response, not a binary edge label.
``ridge``  weighted ridge regression, alpha chosen on perturbations held out
           from inside the training fold, by the same rule for every arm.

Both are trained with the per-gene reliability weight as the sample weight, the
same weight the score uses.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .. import config as C
from .arms import ArmFeatures


def pair_index(rows: np.ndarray, n_readout: int, self_col: np.ndarray, device: str):
    """Every (perturbation, read-out gene) pair except the perturbed gene's own
    column, which is the knockdown itself and trivially predictable."""
    r = np.repeat(rows, n_readout)
    c = np.tile(np.arange(n_readout, dtype=np.int64), rows.size)
    keep = c != self_col[r]
    return (
        torch.as_tensor(r[keep], device=device),
        torch.as_tensor(c[keep], device=device),
    )


class Standardizer:
    def __init__(self, arm: ArmFeatures, rows: torch.Tensor, cols: torch.Tensor, n: int = 200_000):
        idx = torch.randperm(rows.numel(), device=rows.device)[: min(n, rows.numel())]
        x = arm.batch(rows[idx], cols[idx]).float()
        self.mean = x.mean(dim=0, keepdim=True)
        self.sd = x.std(dim=0, keepdim=True).clamp_min(1e-6)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.sd


class MLP(torch.nn.Module):
    def __init__(self, dim: int, hidden: tuple[int, ...]):
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev = dim
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):  # noqa: D102
        return self.net(x).squeeze(-1)


class AdditiveMLP(torch.nn.Module):
    """``f(perturbed gene) + g(read-out gene)``: same architecture per branch, no
    path between the two genes.

    The node-only control exists to answer whether a score can be had from the
    two genes' separate properties — how variable each is, how many strong
    partners it has — without knowing anything about the pair. A single MLP over
    the concatenation answers a different question, because it can learn any
    interaction between the two gene's properties. Summing two branches makes
    the no-interaction restriction structural.
    """

    def __init__(self, dim: int, split_at: int, hidden: tuple[int, ...]):
        super().__init__()
        self.split_at = split_at
        self.left = MLP(split_at, hidden)
        self.right = MLP(dim - split_at, hidden)

    def forward(self, x):  # noqa: D102
        return self.left(x[:, : self.split_at]) + self.right(x[:, self.split_at :])


def _rows_per_chunk(dim: int, budget: float, lo: int = 8192, hi: int = 262144) -> int:
    """Rows per gathered block, so a wide arm (TranscriptFormer's 5120-wide pair
    embedding) does not allocate a far larger block than a narrow one."""
    return int(min(hi, max(lo, budget / max(dim, 1))))


def _predict(arm, std, model, rows_np, n_readout, device, chunk=None) -> np.ndarray:  # noqa: D103
    chunk = chunk or _rows_per_chunk(arm.dim, 1.2e8)
    out = np.zeros((rows_np.size, n_readout), dtype=np.float32)
    cols_all = torch.arange(n_readout, device=device)
    with torch.no_grad():
        for k, r in enumerate(rows_np):
            rr = torch.full((n_readout,), int(r), device=device, dtype=torch.long)
            preds = []
            for s in range(0, n_readout, chunk):
                x = std(arm.batch(rr[s : s + chunk], cols_all[s : s + chunk]).float())
                preds.append(model(x))
            out[k] = torch.cat(preds).float().cpu().numpy()
    return out


def fit_mlp(
    arm: ArmFeatures,
    y: torch.Tensor,
    w: torch.Tensor,
    rows_train: np.ndarray,
    rows_test: np.ndarray,
    self_col: np.ndarray,
    cfg=C.MLP,
    seed: int = C.SEED,
    arm_test: ArmFeatures | None = None,
    n_readout_test: int | None = None,
) -> dict:
    """``arm_test`` differs from ``arm`` only in the cross-cell-line direction,
    where the translator is trained on one line and read out on the other."""
    device = y.device
    torch.manual_seed(seed)
    tr_rows, tr_cols = pair_index(rows_train, y.shape[1], self_col, str(device))
    std = Standardizer(arm, tr_rows, tr_cols)
    split_at = getattr(arm, "split_at", None)
    model = (
        AdditiveMLP(arm.dim, split_at, cfg.hidden)
        if split_at
        else MLP(arm.dim, cfg.hidden)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = tr_rows.numel()
    y_sd = float(y[torch.as_tensor(rows_train, device=device)].std().clamp_min(1e-8))
    t0 = time.time()
    losses = []
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        total, seen = 0.0, 0
        for s in range(0, n, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            r, c = tr_rows[idx], tr_cols[idx]
            x = std(arm.batch(r, c).float())
            target = y[r, c] / y_sd
            weight = w[c]
            pred = model(x)
            loss = (weight * (pred - target) ** 2).sum() / weight.sum().clamp_min(1e-8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * idx.numel()
            seen += int(idx.numel())
        losses.append(total / max(seen, 1))
    model.eval()
    pred = _predict(
        arm_test or arm, std, model, rows_test, n_readout_test or y.shape[1], str(device)
    ) * y_sd
    return {
        "prediction": pred,
        "train_loss_first": float(losses[0]),
        "train_loss_last": float(losses[-1]),
        "seconds": round(time.time() - t0, 2),
        "n_train_pairs": int(n),
        "dim": int(arm.dim),
    }


def fit_ridge(
    arm: ArmFeatures,
    y: torch.Tensor,
    w: torch.Tensor,
    rows_train: np.ndarray,
    rows_test: np.ndarray,
    self_col: np.ndarray,
    alphas=C.RIDGE_ALPHAS,
    seed: int = C.SEED,
    chunk: int | None = None,
    arm_test: ArmFeatures | None = None,
    n_readout_test: int | None = None,
) -> dict:
    device = y.device
    chunk = chunk or _rows_per_chunk(arm.dim, 2.5e8)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(rows_train.size)
    n_val = max(1, rows_train.size // 5)
    inner_val = rows_train[perm[:n_val]]
    inner_tr = rows_train[perm[n_val:]]
    t0 = time.time()

    tr_rows, tr_cols = pair_index(rows_train, y.shape[1], self_col, str(device))
    std = Standardizer(arm, tr_rows, tr_cols)
    d = arm.dim + 1

    def accumulate(rows: np.ndarray):
        """Weighted normal equations: X'WX, X'Wy, y'Wy and the weight total."""
        r, c = pair_index(rows, y.shape[1], self_col, str(device))
        a = torch.zeros((d, d), device=device, dtype=torch.float64)
        b = torch.zeros(d, device=device, dtype=torch.float64)
        yty = 0.0
        wsum = 0.0
        for s in range(0, r.numel(), chunk):
            rr, cc = r[s : s + chunk], c[s : s + chunk]
            x = std(arm.batch(rr, cc).float())
            x = torch.cat([x, torch.ones((x.shape[0], 1), device=device)], dim=1).double()
            sw = w[cc].double().sqrt()
            xs = x * sw.unsqueeze(1)
            ys = y[rr, cc].double() * sw
            a += xs.T @ xs
            b += xs.T @ ys
            yty += float((ys * ys).sum())
            wsum += float((sw * sw).sum())
        return a, b, yty, wsum

    a_tr, b_tr, _, _ = accumulate(inner_tr)
    a_val, b_val, yty_val, wsum_val = accumulate(inner_val)
    eye = torch.eye(d, device=device, dtype=torch.float64)
    best, best_alpha = None, None
    for alpha in alphas:
        beta = torch.linalg.solve(a_tr + alpha * eye, b_tr)
        # weighted MSE in closed form from the validation normal equations
        err = float(beta @ (a_val @ beta) - 2.0 * (beta @ b_val) + yty_val)
        score = err / max(wsum_val, 1e-9)
        if best is None or score < best:
            best, best_alpha = score, float(alpha)

    a_all, b_all, _, _ = accumulate(rows_train)
    beta = torch.linalg.solve(a_all + best_alpha * eye, b_all)

    n_out = n_readout_test or y.shape[1]
    arm_out = arm_test or arm
    out = np.zeros((rows_test.size, n_out), dtype=np.float32)
    cols_all = torch.arange(n_out, device=device)
    with torch.no_grad():
        for k, r in enumerate(rows_test):
            rr = torch.full((n_out,), int(r), device=device, dtype=torch.long)
            x = std(arm_out.batch(rr, cols_all).float())
            x = torch.cat([x, torch.ones((x.shape[0], 1), device=device)], dim=1).double()
            out[k] = (x @ beta).float().cpu().numpy()
    return {
        "prediction": out,
        "alpha": best_alpha,
        "inner_val_weighted_mse": float(best),
        "seconds": round(time.time() - t0, 2),
        "dim": int(arm.dim),
    }


def fit_mean_only(y: torch.Tensor, rows_test: np.ndarray) -> dict:
    """The baseline that predicts no perturbation-specific response at all."""
    return {
        "prediction": np.zeros((rows_test.size, y.shape[1]), dtype=np.float32),
        "seconds": 0.0,
        "dim": 0,
    }

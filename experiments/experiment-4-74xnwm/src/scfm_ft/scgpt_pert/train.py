"""scGPT's perturbation fine-tuning loop, as published.

Optimizer, schedule, loss, gradient clipping, mixed precision, epoch count and
early stopping are the tutorial's. The validation quantity is the one the
tutorial selects on -- agreement between the predicted and measured
perturbation response -- computed here against the same effect matrix run 3
scores, over validation perturbations drawn from the fine-tuning folds only.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .. import config as C
from .data import BatchMaker, LineCells, describe_batch


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """scGPT's own loss (scgpt/loss.py), restated so the module is importable
    without executing ``scgpt/__init__``. Identical arithmetic."""
    m = mask.float()
    loss = torch.nn.functional.mse_loss(pred * m, target * m, reduction="sum")
    return loss / m.sum()


@torch.no_grad()
def predict_delta(
    model,
    cells: LineCells,
    pert_rows: np.ndarray,
    n_control: int,
    amp: bool = False,
    batch_cells: int = 8,
    seed: int = C.SEED,
) -> np.ndarray:
    """Predicted perturbation response for each perturbation, over all genes.

    The model's own query: the same control inputs are run with the
    perturbation flag set and with no flag, and the difference is averaged over
    the control cells. Returns (len(pert_rows), n_genes) on the log1p scale the
    model predicts in.

    Full precision by default even though the training step uses autocast: this
    is a difference between two nearly equal forward passes, and half precision
    cannot represent it.
    """
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_ctrl = int(cells.x_ctrl.shape[0])
    take = min(int(n_control), n_ctrl)
    idx = torch.randperm(n_ctrl, generator=g)[:take].to(cells.device)
    x = cells.x_ctrl[idx]
    G = cells.n_genes
    src = torch.arange(G, device=cells.device).unsqueeze(0)
    zero = torch.zeros(1, G, dtype=torch.long, device=cells.device)

    base = torch.zeros(take, G, device=cells.device, dtype=torch.float64)
    for s in range(0, take, batch_cells):
        v = x[s : s + batch_cells]
        b = v.shape[0]
        pad = torch.zeros(b, G, dtype=torch.bool, device=cells.device)
        with torch.autocast("cuda", enabled=amp):
            base[s : s + b] = model(
                src.expand(b, -1), v, zero.expand(b, -1), src_key_padding_mask=pad
            )["mlm_output"].double()

    out = np.zeros((len(pert_rows), G), dtype=np.float64)
    for k, p in enumerate(pert_rows):
        col = int(cells.self_col[int(p)])
        flags = torch.zeros(1, G, dtype=torch.long, device=cells.device)
        flags[0, col] = 1
        acc = torch.zeros(take, G, device=cells.device, dtype=torch.float64)
        for s in range(0, take, batch_cells):
            v = x[s : s + batch_cells]
            b = v.shape[0]
            pad = torch.zeros(b, G, dtype=torch.bool, device=cells.device)
            with torch.autocast("cuda", enabled=amp):
                acc[s : s + b] = model(
                    src.expand(b, -1), v, flags.expand(b, -1), src_key_padding_mask=pad
                )["mlm_output"].double()
        out[k] = (acc - base).mean(dim=0).cpu().numpy()
    return out


def validation_score(
    model, cells: LineCells, effect: np.ndarray, rows: np.ndarray, n_control: int, amp: bool = False
) -> dict:
    """Mean per-perturbation Pearson between predicted and measured response.

    The perturbed gene's own column is excluded, as it is in every scored
    quantity in run 3: the knockdown of a gene on itself is the assay working,
    not a prediction of the response.
    """
    pred = predict_delta(model, cells, rows, n_control, amp)
    self_cols = cells.self_col.cpu().numpy()
    rs = []
    for k, p in enumerate(rows):
        keep = np.ones(effect.shape[1], dtype=bool)
        keep[int(self_cols[int(p)])] = False
        a = pred[k][keep].astype(np.float64)
        b = effect[int(p)][keep].astype(np.float64)
        if np.std(a) <= 0 or np.std(b) <= 0:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    arr = np.asarray(rs, dtype=np.float64)
    return {
        "pearson": float(np.mean(arr)) if arr.size else float("nan"),
        "pearson_median": float(np.median(arr)) if arr.size else float("nan"),
        "n_perturbations": int(arr.size),
    }


def finetune(
    model,
    cells: LineCells,
    effect: np.ndarray,
    out_dir: Path,
    n_val_control: int = 32,
    log_every: int = 50,
) -> dict:
    """Run the recipe. Returns the history; leaves the best weights in ``model``."""
    R = C.RECIPE
    dev = cells.device
    train_rows = cells.cells_of(cells.is_train)
    val_perts = np.flatnonzero(cells.is_val & cells.usable)
    if train_rows.numel() == 0:
        raise RuntimeError("no training cells: check the ingest output and the split")
    maker = BatchMaker(cells, R.max_seq_len, R.batch_size, C.SEED)

    opt = torch.optim.Adam(model.parameters(), lr=R.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, R.schedule_interval, gamma=R.schedule_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=R.amp)

    # what actually enters the forward pass, written before the first step
    sample_batch = maker.batch(train_rows[: R.batch_size])
    sample = describe_batch(sample_batch, cells)
    (Path(out_dir) / "forward_pass_sample.json").write_text(json.dumps(sample, indent=2))
    print(json.dumps({k: v for k, v in sample.items() if k != "rows"}, indent=2), flush=True)
    for row in sample["rows"][:3]:
        print("  sample row:", json.dumps(row), flush=True)

    history: list[dict] = []
    best = {"score": -np.inf, "epoch": 0, "state": None}
    patience = 0
    t_start = time.time()
    for epoch in range(1, R.epochs + 1):
        model.train()
        t0 = time.time()
        losses, n_batches = [], 0
        for batch in maker.epoch(train_rows):
            with torch.autocast("cuda", enabled=R.amp):
                out = model(
                    batch["src"],
                    batch["values"],
                    batch["flags"],
                    src_key_padding_mask=batch["pad"],
                )["mlm_output"]
                mask = torch.ones_like(batch["values"], dtype=torch.bool)
                loss = masked_mse_loss(out, batch["target"], mask)
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), R.grad_clip, error_if_nonfinite=False
            )
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
            n_batches += 1
            if log_every and n_batches % log_every == 0:
                print(
                    f"  epoch {epoch:2d} batch {n_batches:4d} "
                    f"loss {np.mean(losses[-log_every:]):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
        sched.step()
        train_loss = float(np.mean(losses))
        val = validation_score(model, cells, effect, val_perts, n_val_control, amp=False)
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_first_100": float(np.mean(losses[:100])),
            "train_loss_last_100": float(np.mean(losses[-100:])),
            "n_batches": n_batches,
            "lr": float(sched.get_last_lr()[0]),
            "val_pearson": val["pearson"],
            "val_pearson_median": val["pearson_median"],
            "val_n_perturbations": val["n_perturbations"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(
            f"| epoch {epoch:2d} | loss {train_loss:.4f} | val pearson {val['pearson']:.4f} "
            f"| {rec['seconds']:.0f}s |",
            flush=True,
        )
        if val["pearson"] > best["score"]:
            best = {
                "score": val["pearson"],
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
            }
            patience = 0
        else:
            patience += 1
            if patience >= R.early_stop:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    final_state = copy.deepcopy(model.state_dict())
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {
        "history": history,
        "best_epoch": int(best["epoch"]),
        "best_val_pearson": float(best["score"]),
        "final_state": final_state,
        "best_state": best["state"],
        "n_train_cells": int(train_rows.numel()),
        "n_val_perturbations": int(val_perts.size),
        "total_seconds": round(time.time() - t_start, 1),
        "gene_subsampling": dict(maker.counters),
        "forward_pass_sample": {k: v for k, v in sample.items() if k != "rows"},
    }

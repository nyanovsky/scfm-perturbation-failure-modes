"""scGPT's perturbation fine-tuning loop, as published.

Optimizer, schedule, loss, gradient clipping, mixed precision, epoch count,
early stopping and the checkpoint selection rule are the tutorial's. The rule
is the one its training-loop cell applies: keep the epoch with the HIGHEST
validation Pearson correlation between the predicted and the measured mean
expression profile of each held-out perturbation, averaged over perturbations
(``val_score = val_metrics["pearson"]``), and stop when it has not improved for
``early_stop`` epochs.

The masked-MSE on the same held-out cells -- the training objective's own
validation value -- is recorded every epoch and its selected epoch is kept as
well, because it is the obvious alternative rule and reading the same run under
both says whether the choice matters. The agreement between the predicted and
the measured perturbation *response* is also recorded every epoch, in the four
forms below; it is a diagnostic and not a selection rule, since two of those
forms depend on a query the fine-tuning objective never trains (the model with
no perturbation flag).
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


def check_loss_matches_upstream(seed: int = 0, n: int = 4, g: int = 64) -> dict:
    """Compare the restated loss against scGPT's own, on random tensors.

    The installed ``scgpt/loss.py`` is loaded by file path, the same route the
    model class takes, so this compares against the published code rather than
    against a reading of it.
    """
    from .model import import_module_by_path

    upstream = import_module_by_path("scgpt.loss", "loss.py")
    gen = torch.Generator().manual_seed(seed)
    pred = torch.randn(n, g, generator=gen)
    target = torch.randn(n, g, generator=gen)
    mask = torch.rand(n, g, generator=gen) > 0.3
    mine = float(masked_mse_loss(pred, target, mask))
    theirs = float(upstream.masked_mse_loss(pred, target, mask))
    return {
        "upstream_file": str(getattr(upstream, "__file__", "")),
        "restated": mine,
        "upstream": theirs,
        "identical": mine == theirs,
    }


@torch.no_grad()
def predict_profiles(
    model,
    cells: LineCells,
    pert_rows: np.ndarray,
    n_control: int,
    amp: bool = False,
    batch_cells: int = 8,
    seed: int = C.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """What the model outputs for each perturbation, and for no perturbation.

    The same control cells are pushed through with the perturbation flag set
    and with no flag at all. Returns the flagged output averaged over those
    control cells, one row per perturbation, and the unflagged output averaged
    the same way, both on the log1p scale the model predicts in.

    Full precision by default even though the training step uses autocast: the
    quantities built from these are differences between two nearly equal
    forward passes, and half precision cannot represent them.
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

    flagged = np.zeros((len(pert_rows), G), dtype=np.float64)
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
        flagged[k] = acc.mean(dim=0).cpu().numpy()
    return flagged, base.mean(dim=0).cpu().numpy()


def predict_delta(
    model,
    cells: LineCells,
    pert_rows: np.ndarray,
    n_control: int,
    amp: bool = False,
    batch_cells: int = 8,
    seed: int = C.SEED,
) -> np.ndarray:
    """The model's own perturbation query: flagged output minus unflagged."""
    flagged, base = predict_profiles(model, cells, pert_rows, n_control, amp, batch_cells, seed)
    return flagged - base[None, :]


def _row_pearson(pred: np.ndarray, meas: np.ndarray, self_cols: np.ndarray) -> dict:
    """Per-perturbation Pearson across genes, the perturbed gene's own column out.

    That column is excluded here as it is in every scored quantity in run 3:
    the knockdown of a gene on itself is the assay working, not a prediction of
    the response.
    """
    rs = []
    for k in range(pred.shape[0]):
        keep = np.ones(pred.shape[1], dtype=bool)
        keep[int(self_cols[k])] = False
        a, b = pred[k][keep].astype(np.float64), meas[k][keep].astype(np.float64)
        if np.std(a) <= 0 or np.std(b) <= 0:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    arr = np.asarray(rs, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "n": int(arr.size),
    }


@torch.no_grad()
def measured_mean_profiles(cells: LineCells, rows: np.ndarray) -> np.ndarray:
    """Each perturbation's measured mean expression profile over its own cells.

    ``compute_perturbation_metrics`` averages the ground-truth cells of a
    condition and correlates that profile with the averaged prediction, so this
    is the target side of the tutorial's selection score. log1p CP10K, the
    scale the model predicts in.
    """
    poc = cells.pert_of_cell
    out = np.zeros((np.asarray(rows).size, cells.n_genes), dtype=np.float64)
    for k, p in enumerate(np.asarray(rows, dtype=np.int64)):
        sel = torch.nonzero(poc == int(p), as_tuple=False).squeeze(1)
        if sel.numel() == 0:
            continue
        out[k] = cells.x_pert[sel].double().mean(dim=0).cpu().numpy()
    return out


def tutorial_pearson(pred_mean: np.ndarray, true_mean: np.ndarray) -> dict:
    """scGPT's checkpoint-selection score, restated.

    ``compute_perturbation_metrics(...)["pearson"]`` in ``scgpt/utils/util.py``:
    for each held-out condition, the Pearson correlation ACROSS ALL GENES
    between the mean measured expression profile and the mean predicted
    expression profile, averaged over conditions. Not a delta and not restricted
    to differentially expressed genes; conditions whose measured profile is
    all-zero are skipped there and here.
    """
    rs = []
    for k in range(pred_mean.shape[0]):
        a = np.asarray(true_mean[k], dtype=np.float64)
        b = np.asarray(pred_mean[k], dtype=np.float64)
        if np.all(a == 0) or np.std(a) <= 0 or np.std(b) <= 0:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    arr = np.asarray(rs, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "min": float(np.min(arr)) if arr.size else float("nan"),
        "n": int(arr.size),
    }


def validation_score(
    model,
    cells: LineCells,
    effect: np.ndarray,
    rows: np.ndarray,
    n_control: int,
    amp: bool = False,
    profiles: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Predicted against measured response on the validation perturbations.

    Four numbers, because the choice of baseline is not innocent. The model is
    only ever trained with a perturbation flag on, so its unflagged output is
    an untrained query; the published evaluation instead reads the response
    against the measured control mean. Both baselines are reported, each once
    as it stands and once after removing the mean predicted and the mean
    measured response over these perturbations, which is the part that is
    specific to the perturbation rather than shared by all of them.

    ``profiles`` lets the caller pass forward passes it has already paid for.
    """
    if rows.size == 0:
        return {}
    flagged, base = (
        profiles if profiles is not None else predict_profiles(model, cells, rows, n_control, amp)
    )
    ctrl_mean = cells.x_ctrl.mean(dim=0).double().cpu().numpy()
    meas = effect[np.asarray(rows, dtype=np.int64)].astype(np.float64)
    self_cols = cells.self_col.cpu().numpy()[np.asarray(rows, dtype=np.int64)]
    out = {}
    for name, pred in (
        ("selfquery", flagged - base[None, :]),
        ("ctrlmean", flagged - ctrl_mean[None, :]),
    ):
        out[name] = _row_pearson(pred, meas, self_cols)
        out[f"{name}_centered"] = _row_pearson(
            pred - pred.mean(axis=0, keepdims=True),
            meas - meas.mean(axis=0, keepdims=True),
            self_cols,
        )
    return out


@torch.no_grad()
def validation_loss(
    model, cells: LineCells, rows: torch.Tensor, seed: int, force_perturbed: bool = True
) -> dict:
    """The training objective's own validation value: masked MSE on held-out cells.

    Not the tutorial's selection score -- that is ``tutorial_pearson`` -- but
    the obvious alternative rule, kept so the same run can be read under both.
    Re-seeded on every call, so the same control inputs and the same sampled
    gene positions are used at every epoch and the epochs are comparable.
    """
    model.eval()
    R = C.RECIPE
    maker = BatchMaker(cells, R.max_seq_len, R.batch_size, seed, force_perturbed)
    total, n = 0.0, 0
    for batch in maker.epoch(rows):
        out = model(
            batch["src"], batch["values"], batch["flags"], src_key_padding_mask=batch["pad"]
        )["mlm_output"]
        mask = torch.ones_like(batch["values"], dtype=torch.bool)
        b = int(batch["values"].shape[0])
        total += float(masked_mse_loss(out, batch["target"], mask)) * b
        n += b
    return {"loss": total / max(n, 1), "n_cells": n, "n_batches": maker.counters["batches"]}


def finetune(
    model,
    cells: LineCells,
    effect: np.ndarray,
    out_dir: Path,
    n_val_control: int = 32,
    log_every: int = 50,
    force_perturbed: bool = True,
) -> dict:
    """Run the recipe. Returns the history; leaves the best weights in ``model``."""
    R = C.RECIPE
    train_rows = cells.cells_of(cells.is_train)
    val_rows = cells.cells_of(cells.is_val)
    val_perts = np.flatnonzero(cells.is_val & cells.usable)
    if train_rows.numel() == 0:
        raise RuntimeError("no training cells: check the ingest output and the split")
    if val_rows.numel() == 0:
        raise RuntimeError("no validation cells: check the split")
    maker = BatchMaker(cells, R.max_seq_len, R.batch_size, C.SEED, force_perturbed)

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
    # the tutorial's rule (highest validation Pearson) and the training
    # objective's own validation value (lowest masked MSE), tracked side by
    # side. Each keeps its own patience counter and stops on its own score; a
    # rule that has stopped no longer updates its selection, so each selected
    # epoch is exactly the one that rule would have reached on its own.
    sel = {
        "pearson": {"score": -np.inf, "epoch": 0, "state": None, "patience": 0, "stopped": None},
        "mse": {"score": np.inf, "epoch": 0, "state": None, "patience": 0, "stopped": None},
    }
    val_true_mean = measured_mean_profiles(cells, val_perts)
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
        vloss = validation_loss(model, cells, val_rows, C.SEED + 7, force_perturbed)
        profiles = predict_profiles(model, cells, val_perts, n_val_control, amp=False)
        delta = validation_score(
            model, cells, effect, val_perts, n_val_control, amp=False, profiles=profiles
        )
        tut = tutorial_pearson(profiles[0], val_true_mean)
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_tutorial_pearson": tut["mean"],
            "val_tutorial_pearson_median": tut["median"],
            "val_tutorial_pearson_n": tut["n"],
            "train_loss_first_100": float(np.mean(losses[:100])),
            "train_loss_last_100": float(np.mean(losses[-100:])),
            "n_batches": n_batches,
            "lr": float(sched.get_last_lr()[0]),
            "val_loss": vloss["loss"],
            "val_cells": vloss["n_cells"],
            "val_n_perturbations": int(val_perts.size),
            "val_delta_pearson": {k: v["mean"] for k, v in delta.items()},
            "val_delta_pearson_median": {k: v["median"] for k, v in delta.items()},
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(
            f"| epoch {epoch:2d} | train {train_loss:.4f} | val {vloss['loss']:.4f} | "
            f"tutorial pearson {tut['mean']:.5f} | "
            + " ".join(f"{k} {v['mean']:+.3f}" for k, v in delta.items())
            + f" | {rec['seconds']:.0f}s |",
            flush=True,
        )
        for rule, score, better in (
            ("pearson", tut["mean"], tut["mean"] > sel["pearson"]["score"]),
            ("mse", vloss["loss"], vloss["loss"] < sel["mse"]["score"]),
        ):
            s = sel[rule]
            if s["stopped"] is not None:
                continue
            if better:
                s.update(
                    score=float(score),
                    epoch=epoch,
                    state=copy.deepcopy(model.state_dict()),
                    patience=0,
                )
                print(f"  {rule}: best model with score {score:.5f}", flush=True)
            else:
                s["patience"] += 1
                if s["patience"] >= R.early_stop:
                    s["stopped"] = epoch
                    print(f"  {rule}: early stop at epoch {epoch}", flush=True)
        if all(s["stopped"] is not None for s in sel.values()):
            break

    final_state = copy.deepcopy(model.state_dict())
    if sel["pearson"]["state"] is not None:
        model.load_state_dict(sel["pearson"]["state"])
    pear, mse = sel["pearson"], sel["mse"]
    return {
        "history": history,
        "best_epoch": int(pear["epoch"]),
        "selected_epoch": int(pear["epoch"]),
        "selected_val_pearson": float(pear["score"]),
        "selection_rule": (
            "highest validation Pearson between the predicted and the measured mean "
            "expression profile of each held-out perturbation, averaged over "
            'perturbations (the tutorial\'s val_metrics["pearson"])'
        ),
        "pearson_rule_stopped_at_epoch": pear["stopped"],
        "mse_selected_epoch": int(mse["epoch"]),
        "mse_rule_stopped_at_epoch": mse["stopped"],
        "best_val_loss": float(mse["score"]),
        "alternative_rule": "lowest masked MSE on the same held-out validation cells",
        "best_val_delta_pearson": (
            history[int(pear["epoch"]) - 1]["val_delta_pearson"] if pear["epoch"] else {}
        ),
        "final_state": final_state,
        "best_state": pear["state"],
        "mse_state": mse["state"],
        "n_train_cells": int(train_rows.numel()),
        "n_val_perturbations": int(val_perts.size),
        "force_perturbed_gene_in_sequence": bool(force_perturbed),
        "total_seconds": round(time.time() - t_start, 1),
        "gene_subsampling": dict(maker.counters),
        "forward_pass_sample": {k: v for k, v in sample.items() if k != "rows"},
    }

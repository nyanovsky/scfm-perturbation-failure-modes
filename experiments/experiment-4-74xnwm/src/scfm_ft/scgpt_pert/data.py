"""The batches scGPT's perturbation recipe consumes.

GEARS builds one training example per perturbed cell: a control cell's
expression as the input, a flag on the perturbed gene, and the perturbed cell's
own expression as the target, all on the CP10K/log1p scale. That is what this
module produces, from the matrices the ingest job wrote, with the recipe's own
gene-subsampling rule applied per batch.

One deliberate departure from the published loop, in the direction that helps
the model. Upstream samples ``max_seq_len`` input positions uniformly and lets
the perturbed gene fall outside the sample, which leaves the batch with an
all-zero flag vector while the target still carries a perturbation effect: the
model is asked which perturbation this is with the answer removed from its
input. Here the batch's own perturbed genes are always kept in the sample and
the rest of the positions are drawn uniformly as upstream draws them. The
fraction of batches that would have lost their perturbed gene is measured and
recorded, so the size of the departure is a number rather than an assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class LineCells:
    """One cell line's fine-tuning matrices, resident on the device."""

    cell_line: str
    x_pert: torch.Tensor  # (n_cells, G) log1p CP10K, the target cells
    pert_of_cell: torch.Tensor  # (n_cells,) index into the perturbation table
    x_ctrl: torch.Tensor  # (n_ctrl, G) log1p CP10K, the input pool
    self_col: torch.Tensor  # (n_pert,) each perturbation's own read-out column
    pert_target: np.ndarray
    fold: np.ndarray
    groups: np.ndarray
    gene_name: np.ndarray
    is_train: np.ndarray
    is_val: np.ndarray
    is_scored: np.ndarray
    usable: np.ndarray
    n_genes: int
    device: str
    meta: dict = field(default_factory=dict)

    @property
    def n_pert(self) -> int:
        return int(self.pert_target.size)

    def cells_of(self, mask: np.ndarray) -> torch.Tensor:
        """Row indices of the cells whose perturbation is selected by ``mask``."""
        keep = np.flatnonzero(mask & self.usable)
        sel = torch.as_tensor(np.isin(self.pert_of_cell.cpu().numpy(), keep), device=self.device)
        return torch.nonzero(sel, as_tuple=False).squeeze(1)

    def anchor_cp10k(self) -> np.ndarray:
        """The mean control profile, on the CP10K scale run 2's queries use."""
        return torch.expm1(self.x_ctrl).mean(dim=0).float().cpu().numpy()


def load_line(
    cells_dir: Path,
    cell_line: str,
    self_col: np.ndarray,
    gene_name: np.ndarray,
    split: dict,
    device: str = "cuda",
) -> LineCells:
    base = Path(cells_dir) / cell_line
    pert = np.load(base / "pert_cells.npz", allow_pickle=False)
    ctrl = np.load(base / "control_cells.npz", allow_pickle=False)
    idx = np.load(base / "perturbation_index.npz", allow_pickle=False)

    x_pert = torch.log1p(torch.as_tensor(pert["X"], dtype=torch.float32, device=device))
    x_ctrl = torch.log1p(torch.as_tensor(ctrl["X"], dtype=torch.float32, device=device))
    return LineCells(
        cell_line=cell_line,
        x_pert=x_pert,
        pert_of_cell=torch.as_tensor(pert["pert_index"].astype(np.int64), device=device),
        x_ctrl=x_ctrl,
        self_col=torch.as_tensor(np.asarray(self_col, dtype=np.int64), device=device),
        pert_target=idx["pert_target"].astype(str),
        fold=idx["fold"].astype(np.int64),
        groups=idx["groups"].astype(np.int64),
        gene_name=np.asarray(gene_name).astype(str),
        is_train=split["is_train"],
        is_val=split["is_val"],
        is_scored=split["is_scored"],
        usable=idx["usable"].astype(bool),
        n_genes=int(x_pert.shape[1]),
        device=device,
        meta={
            "n_pert_cells": int(x_pert.shape[0]),
            "n_control_cells": int(x_ctrl.shape[0]),
            "cells_available_median": float(np.median(idx["cells_available"])),
            "cells_kept_median": float(np.median(idx["cells_kept"])),
            "n_unusable_perturbations": int((~idx["usable"].astype(bool)).sum()),
        },
    )


class BatchMaker:
    """Batches of (input control cell, perturbation flag, target cell)."""

    def __init__(
        self,
        cells: LineCells,
        max_seq_len: int,
        batch_size: int,
        seed: int,
        force_perturbed: bool = True,
    ):
        self.cells = cells
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.n_genes = cells.n_genes
        self.subsampled = self.max_seq_len < self.n_genes
        #: upstream draws the batch's gene positions uniformly and lets a row
        #: lose its own perturbed gene, which leaves that row with no flag at
        #: all. Keeping it is the one deliberate departure from the recipe and
        #: it can only help the fine-tuned model, so it is the default here and
        #: ``False`` reproduces upstream exactly.
        self.force_perturbed = bool(force_perturbed)
        self.counters = {
            "batches": 0,
            "rows": 0,
            "batches_that_would_lose_the_perturbed_gene": 0,
            "rows_that_would_lose_the_perturbed_gene": 0,
        }

    def _positions(self, forced: torch.Tensor) -> torch.Tensor:
        """The recipe's random position sample, with ``forced`` guaranteed in it."""
        if not self.subsampled:
            return torch.arange(self.n_genes, device=self.cells.device)
        perm = torch.randperm(self.n_genes, generator=self.gen).to(self.cells.device)
        drawn = perm[: self.max_seq_len]
        uniq_forced = torch.unique(forced)
        missing = uniq_forced[~torch.isin(uniq_forced, drawn)]
        self.counters["batches"] += 1
        self.counters["rows"] += int(forced.numel())
        self.counters["rows_that_would_lose_the_perturbed_gene"] += int(
            (~torch.isin(forced, drawn)).sum()
        )
        if not self.force_perturbed:
            return drawn
        if missing.numel():
            self.counters["batches_that_would_lose_the_perturbed_gene"] += 1
            # replace the tail rather than growing the sequence, so the recipe's
            # sequence length is exactly preserved
            drawn = torch.cat([drawn[: self.max_seq_len - missing.numel()], missing])
        return drawn

    def batch(self, rows: torch.Tensor) -> dict:
        c = self.cells
        b = rows.numel()
        ctrl_idx = torch.randint(
            0, c.x_ctrl.shape[0], (b,), generator=self.gen
        ).to(c.device)
        pert_ids = c.pert_of_cell[rows]
        cols = c.self_col[pert_ids]
        positions = self._positions(cols)

        values = c.x_ctrl[ctrl_idx][:, positions]
        target = c.x_pert[rows][:, positions]
        flags = torch.zeros(b, self.n_genes, dtype=torch.long, device=c.device)
        flags[torch.arange(b, device=c.device), cols] = 1
        flags = flags[:, positions]
        src = positions.unsqueeze(0).expand(b, -1)
        pad = torch.zeros_like(values, dtype=torch.bool)
        return {
            "src": src,
            "values": values,
            "flags": flags,
            "target": target,
            "pad": pad,
            "positions": positions,
            "pert_ids": pert_ids,
            "self_cols": cols,
            "control_rows": ctrl_idx,
        }

    def epoch(self, rows: torch.Tensor):
        order = torch.randperm(rows.numel(), generator=self.gen).to(rows.device)
        rows = rows[order]
        for start in range(0, rows.numel(), self.batch_size):
            chunk = rows[start : start + self.batch_size]
            if chunk.numel() < 2:
                continue
            yield self.batch(chunk)


def describe_batch(batch: dict, cells: LineCells, n_show: int = 8) -> dict:
    """A readable sample of what actually enters the forward pass.

    Written before training starts, because a wrong loss mask or a flag on the
    wrong token would not crash anything: it would quietly train a different
    objective and every downstream number would be a measurement of that.
    """
    pos = batch["positions"].detach().cpu().numpy()
    values = batch["values"].detach().float().cpu().numpy()
    target = batch["target"].detach().float().cpu().numpy()
    flags = batch["flags"].detach().cpu().numpy()
    cols = batch["self_cols"].detach().cpu().numpy()
    pert_ids = batch["pert_ids"].detach().cpu().numpy()
    where = {int(p): i for i, p in enumerate(pos)}

    rows = []
    for b in range(min(n_show, values.shape[0])):
        col = int(cols[b])
        j = where.get(col)
        flagged = np.flatnonzero(flags[b] == 1)
        rows.append(
            {
                "row": b,
                "perturbation": str(cells.pert_target[int(pert_ids[b])]),
                "perturbed_gene": str(cells.gene_name[col]),
                "perturbed_gene_readout_column": col,
                "perturbed_gene_in_this_batch_sequence": j is not None,
                "sequence_position_of_perturbed_gene": None if j is None else int(j),
                "flagged_positions": [int(x) for x in flagged],
                "flag_is_on_the_perturbed_gene": bool(
                    flagged.size == 1 and j is not None and int(flagged[0]) == j
                ),
                "input_value_at_perturbed_gene_log1p_cp10k": None if j is None else float(values[b, j]),
                "target_value_at_perturbed_gene_log1p_cp10k": None if j is None else float(target[b, j]),
                "input_first_5_genes": [str(cells.gene_name[int(p)]) for p in pos[:5]],
                "input_first_5_values": [float(v) for v in values[b, :5]],
                "target_first_5_values": [float(v) for v in target[b, :5]],
            }
        )
    return {
        "batch_shape": list(values.shape),
        "sequence_length": int(pos.size),
        "gene_set_size": int(cells.n_genes),
        "input": "a randomly drawn non-targeting control cell, log1p CP10K over the sampled genes",
        "target": "the perturbed cell's own expression, log1p CP10K over the same genes",
        "loss_mask": "all sampled positions (masked_mse_loss with an all-ones mask), as the recipe does",
        "flags_per_row": [int((flags[b] == 1).sum()) for b in range(values.shape[0])],
        "n_rows_with_exactly_one_flag": int(((flags == 1).sum(axis=1) == 1).sum()),
        "n_rows_with_no_flag": int(((flags == 1).sum(axis=1) == 0).sum()),
        "input_value_mean": float(values.mean()),
        "target_value_mean": float(target.mean()),
        "input_zero_fraction": float((values == 0).mean()),
        "target_zero_fraction": float((target == 0).mean()),
        "rows": rows,
    }

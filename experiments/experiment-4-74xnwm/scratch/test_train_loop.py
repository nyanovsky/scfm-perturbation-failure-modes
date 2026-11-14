"""CPU check of the changed selection rule, with a stand-in model.

The training loop, the batch maker, the validation loss and the four response
agreements are exercised end to end on small random matrices, so a mistake in
the loop shows up here in seconds rather than twenty minutes into a GPU job.
The stand-in model is deliberately trivial: it is the plumbing that is under
test, not the transformer.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scfm_ft import config as C  # noqa: E402
from scfm_ft.scgpt_pert import train as trainmod  # noqa: E402
from scfm_ft.scgpt_pert.data import LineCells  # noqa: E402


class StandIn(torch.nn.Module):
    """Input values plus a response that depends on the gene and on which gene
    carries the perturbation flag, so the flag reaches every position the way
    attention does in the real model."""

    def __init__(self, n_genes: int):
        super().__init__()
        self.gene = torch.nn.Embedding(n_genes, 4)
        self.head = torch.nn.Linear(8, 1)

    def forward(self, src, values, flags, src_key_padding_mask=None):
        g = self.gene(src)
        ctx = (g * flags.float().unsqueeze(-1)).sum(dim=1, keepdim=True)
        h = torch.cat([g, ctx.expand_as(g)], dim=-1)
        return {"mlm_output": values + self.head(h).squeeze(-1)}


def fake_cells(n_pert=24, n_cells_per=8, n_ctrl=48, n_genes=40, seed=0) -> LineCells:
    rng = np.random.default_rng(seed)
    self_col = rng.permutation(n_genes)[:n_pert]
    x_ctrl = np.abs(rng.normal(1.0, 0.4, (n_ctrl, n_genes))).astype(np.float32)
    pert_of_cell = np.repeat(np.arange(n_pert), n_cells_per)
    x_pert = np.abs(rng.normal(1.0, 0.4, (pert_of_cell.size, n_genes))).astype(np.float32)
    for k in range(pert_of_cell.size):
        x_pert[k, self_col[pert_of_cell[k]]] = 0.0
    fold = np.tile(np.arange(5), n_pert // 5 + 1)[:n_pert]
    is_scored = np.isin(fold, C.MODEL_HELDOUT_FOLDS)
    rest = np.flatnonzero(~is_scored)
    is_val = np.zeros(n_pert, bool)
    is_val[rest[:4]] = True
    is_train = ~is_scored & ~is_val
    return LineCells(
        cell_line="test",
        x_pert=torch.as_tensor(x_pert),
        pert_of_cell=torch.as_tensor(pert_of_cell),
        x_ctrl=torch.as_tensor(x_ctrl),
        self_col=torch.as_tensor(self_col.astype(np.int64)),
        pert_target=np.array([f"G{i}" for i in range(n_pert)]),
        fold=fold,
        groups=np.arange(n_pert),
        gene_name=np.array([f"g{i}" for i in range(n_genes)]),
        is_train=is_train,
        is_val=is_val,
        is_scored=is_scored,
        usable=np.ones(n_pert, bool),
        n_genes=n_genes,
        device="cpu",
        meta={},
    )


def main() -> int:
    import dataclasses

    C.RECIPE = dataclasses.replace(
        C.RECIPE,
        amp=False,
        epochs=3,
        early_stop=2,
        max_seq_len=24,  # smaller than the gene set, so subsampling is live
        batch_size=8,
    )
    trainmod.C = C

    cells = fake_cells()
    effect = np.random.default_rng(1).normal(0, 1, (cells.n_pert, cells.n_genes)).astype(np.float32)
    model = StandIn(cells.n_genes)

    # the validation loss must be identical for the same weights, or epochs are
    # not comparable and the selection rule is reading noise
    val_rows = cells.cells_of(cells.is_val)
    a = trainmod.validation_loss(model, cells, val_rows, C.SEED + 7)
    b = trainmod.validation_loss(model, cells, val_rows, C.SEED + 7)
    assert a["loss"] == b["loss"], (a, b)
    assert a["n_cells"] == int(val_rows.numel()), (a, val_rows.numel())

    val_perts = np.flatnonzero(cells.is_val & cells.usable)
    delta = trainmod.validation_score(model, cells, effect, val_perts, 8)
    assert set(delta) == {"selfquery", "selfquery_centered", "ctrlmean", "ctrlmean_centered"}
    for k, v in delta.items():
        assert v["n"] == val_perts.size, (k, v)

    flagged, base = trainmod.predict_profiles(model, cells, val_perts, 8)
    d = trainmod.predict_delta(model, cells, val_perts, 8)
    assert np.allclose(d, flagged - base[None, :]), "the two query paths disagree"
    assert np.allclose(
        trainmod.validation_score(
            model, cells, effect, val_perts, 8, profiles=(flagged, base)
        )["ctrlmean"]["mean"],
        delta["ctrlmean"]["mean"],
    ), "passing the forward passes in changes the answer"

    # the tutorial's selection score, against a literal transcription of the
    # upstream loop it restates
    true_mean = trainmod.measured_mean_profiles(cells, val_perts)
    for k, p in enumerate(val_perts):
        sel = np.flatnonzero(cells.pert_of_cell.numpy() == p)
        assert np.allclose(true_mean[k], cells.x_pert.numpy()[sel].mean(axis=0)), k
    from scipy.stats import pearsonr

    literal = float(
        np.mean([pearsonr(true_mean[k], flagged[k])[0] for k in range(val_perts.size)])
    )
    tut = trainmod.tutorial_pearson(flagged, true_mean)
    assert abs(tut["mean"] - literal) < 1e-12, (tut, literal)
    assert tut["n"] == val_perts.size, tut

    with tempfile.TemporaryDirectory() as tmp:
        hist = trainmod.finetune(model, cells, effect, Path(tmp), n_val_control=8, log_every=0)
        sample = json.loads((Path(tmp) / "forward_pass_sample.json").read_text())
    losses = [h["val_loss"] for h in hist["history"]]
    pears = [h["val_tutorial_pearson"] for h in hist["history"]]
    # the recipe's rule: the epoch with the highest validation Pearson
    assert hist["selected_epoch"] == int(np.argmax(pears)) + 1, (hist["selected_epoch"], pears)
    assert hist["selected_val_pearson"] == max(pears)
    assert hist["best_epoch"] == hist["selected_epoch"]
    # and the alternative, tracked beside it on the same run
    assert hist["mse_selected_epoch"] == int(np.argmin(losses)) + 1, (hist, losses)
    assert hist["best_val_loss"] == min(losses)
    assert hist["mse_state"] is not None and hist["best_state"] is not None
    assert "pearson" in hist["selection_rule"]
    assert set(hist["best_val_delta_pearson"]) == set(delta)
    counters = hist["gene_subsampling"]
    assert counters["rows"] >= counters["rows_that_would_lose_the_perturbed_gene"] > 0
    assert sample["n_rows_with_no_flag"] == 0

    print(json.dumps({
        "val_loss_per_epoch": [round(x, 6) for x in losses],
        "val_tutorial_pearson_per_epoch": [round(x, 6) for x in pears],
        "selected_epoch": hist["selected_epoch"],
        "masked_mse_would_select": hist["mse_selected_epoch"],
        "selection_rule": hist["selection_rule"],
        "delta_pearson_at_selected": hist["best_val_delta_pearson"],
        "rows_that_would_lose_the_perturbed_gene": (
            counters["rows_that_would_lose_the_perturbed_gene"] / counters["rows"]
        ),
    }, indent=2))
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

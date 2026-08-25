"""One training arm, end to end: fine-tune, query, score.

    python -m scfm_ft.run_arm --cell-line hepg2 --arm own_trainable

Everything a later session needs to reuse the trained model is written to the
arm's own directory under the experiment artifacts root: the weights at the
selected epoch and at the last epoch, the exact training configuration, the
resolved gene table before and after fine-tuning (including the learned
projection where a table was substituted), the full loss and validation
history, the sample of what entered the forward pass, and the identity of the
base checkpoint it started from.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import followup, paths, score
from .ingest import training_split
from .scgpt_pert import checkpoint as ckptmod
from .scgpt_pert import data as datamod
from .scgpt_pert import model as modelmod
from .scgpt_pert import query as querymod
from .scgpt_pert import train as trainmod
from .sibling import run2, run3

#: finite-difference step for the gradient cross-check, in log1p CP10K units.
#: The feature itself is the exact forward-mode derivative where it is
#: available; this only sizes the difference it is checked against.
FD_STEP = 0.05


def load_targets(bundle_root: Path, cell_line: str) -> dict:
    t = np.load(Path(bundle_root) / cell_line / "targets.npz", allow_pickle=False)
    return {
        "effect": t["effect"].astype(np.float32),
        "pert_target": t["pert_target"].astype(str),
        "pert_gene_id": t["pert_gene_id"].astype(str),
        "self_col": t["pert_readout_col"].astype(np.int64),
        "readout_gene_id": t["readout_gene_id"].astype(str),
        "readout_gene_name": t["readout_gene_name"].astype(str),
    }


def substituted_table(arm, esm2_path: Path, readout_gene_id: np.ndarray) -> tuple[np.ndarray, dict]:
    """The wide table an ESM2 or random arm starts from, in read-out order."""
    if arm.table == "scgpt":
        return None, {"table": "scgpt", "note": "the published gene embedding rows"}
    install = run3("protein.install_arm")
    z = install._load(Path(esm2_path))
    emb = z["embedding"].astype(np.float32)
    gene_id = z["gene_id"].astype(str)
    mapped = z["mapped"].astype(bool) if "mapped" in z else np.ones(gene_id.size, bool)
    pos = {g: i for i, g in enumerate(gene_id)}
    idx = np.array([pos.get(g, -1) for g in readout_gene_id])
    out = np.zeros((readout_gene_id.size, emb.shape[1]), dtype=np.float32)
    present = idx >= 0
    out[present] = emb[idx[present]]
    report = {
        "table": arm.table,
        "dim": int(emb.shape[1]),
        "n_readout": int(readout_gene_id.size),
        "in_table": int(present.sum()),
        "with_protein": int((present & np.array([mapped[i] if i >= 0 else False for i in idx])).sum()),
        "zero_rows": int((np.linalg.norm(out, axis=1) == 0).sum()),
    }
    if arm.table == "random":
        # matched to the protein table's width and per-row scale, so the only
        # thing this arm lacks is the information in the rows
        rng = np.random.default_rng(C.SEED + 977)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        scale = float(np.median(norms[norms > 0])) if (norms > 0).any() else 1.0
        rand = rng.standard_normal(out.shape).astype(np.float32)
        rand /= np.linalg.norm(rand, axis=1, keepdims=True)
        out = rand * scale
        report.update(
            {
                "table": "random",
                "matched_to": arm.table if arm.table != "random" else C.ESM2_TABLE,
                "row_norm": scale,
                "note": "random unit rows at the protein table's median row norm and width",
            }
        )
    return out, report


def extract_stage(
    model,
    stage: str,
    arm_name: str,
    readout: pd.DataFrame,
    targets: dict,
    device: str,
) -> dict:
    """Both query families, read off the model as it stands.

    The value-substitution family is produced by run 2's ``extract_anchor``,
    unmodified, so the numbers are commensurable with the frozen blocks runs 2
    and 3 scored. The perturbation-flag family is the model's own query.
    """
    extract = run2("features.extract")
    forms_mod = run2("forms")
    C2 = run2("config")

    n_readout = int(readout.shape[0])
    q = querymod.ScGPTPertQuery(model, n_readout, device=device)
    anchors = extract.build_anchors(readout, None)
    profiles = anchors["mean_control"]["profiles"]
    layout = q.configure(readout, profiles[0], "anchor")

    perturbed = pd.DataFrame(
        {
            "readout_col": targets["self_col"],
            "gene_id": targets["pert_gene_id"],
            "target": targets["pert_target"],
        }
    )
    t0 = time.time()
    res = extract.extract_anchor(
        q,
        profiles,
        perturbed,
        FD_STEP,
        C2.VALUE_GRID_CP10K,
        log_every=100,
        exact_derivative=True,
    )
    value_forms = {
        form: res["forms"][form][C.FEATURE_FORM_KEY]
        for form in ("zero_out", "vvp", "gdt")
        if form in res["forms"]
    }

    # the model's own perturbation query at the same anchor
    anchor_native = np.asarray(q.to_native(profiles[0]), dtype=np.float32)
    pf = querymod.pert_flag_response(q, anchor_native, targets["self_col"])
    y_base = np.broadcast_to(pf["baseline"].astype(np.float64), pf["delta"].shape)
    pf_forms = forms_mod.normalize_forms(pf["delta"], y_base + pf["delta"], y_base, q)

    table, table_note = q.gene_embeddings()
    prefix = "ft" if stage == "final" else "pre"
    stem = C.model_stem(arm_name)
    return {
        "value_key": f"{prefix}_{stem}",
        "flag_key": f"{prefix}q_{stem}",
        "value_forms": value_forms,
        "flag_forms": {"zero_out": pf_forms[C.FEATURE_FORM_KEY]},
        "gene_table": table,
        "gene_table_note": table_note,
        "values": np.asarray(C2.VALUE_GRID_CP10K, dtype=np.float64),
        "value_keep": np.ones(len(C2.VALUE_GRID_CP10K), dtype=bool),
        "flag_values": np.zeros(1, dtype=np.float64),
        "flag_value_keep": np.ones(1, dtype=bool),
        "diagnostics": {
            "stage": stage,
            "layout": layout.as_dict(),
            "seconds": round(time.time() - t0, 1),
            "n_perturbed": int(perturbed.shape[0]),
            "pert_flag_baseline_median": float(np.median(pf["baseline"])),
            "pert_flag_delta_abs_median": float(np.median(np.abs(pf["delta"]))),
            "pert_flag_self_delta_median": float(
                np.median(pf["delta"][np.arange(perturbed.shape[0]), targets["self_col"]])
            ),
            **{k: v for k, v in res["diagnostics"].items() if k != "exact_vs_finite_difference"},
            "exact_vs_finite_difference": res["diagnostics"].get("exact_vs_finite_difference"),
        },
        "query_summary": res["summary"],
    }


def pack_run(cell_line: str, run_name: str, out_dir: Path) -> Path:
    """One archive per training run, holding everything the merge job reads.

    Staging into a job is capped at 128 files and sixteen runs produce ten times
    that in small tables, so each run leaves a single archive: its score tables,
    its training metadata, and its prediction-spread row. The weights and the
    prediction matrices stay in the run's own directory.
    """
    import tarfile

    scores = paths.scores_dir()
    want = [
        *sorted(scores.glob(f"restricted_{cell_line}__{run_name}.parquet")),
        *sorted(scores.glob(f"contrasts_{cell_line}__{run_name}.parquet")),
        *sorted(scores.glob(f"runs_{cell_line}__{run_name}.parquet")),
        *sorted(scores.glob(f"per_pert_{cell_line}__{run_name}.parquet")),
        *sorted(scores.glob(f"direct_{cell_line}__{run_name}.parquet")),
        *sorted(scores.glob(f"direct_{cell_line}__{run_name}__*.parquet")),
        *sorted(scores.glob(f"summary_{cell_line}__{run_name}.json")),
    ]
    meta = [
        out_dir / name
        for name in (
            "train_config.json",
            "loss_history.json",
            "provenance.json",
            "prediction_spread.json",
            "forward_pass_sample.json",
        )
    ]
    dest = paths.packs_dir() / f"pack_{cell_line}__{run_name}.tar.gz"
    with tarfile.open(dest, "w:gz") as tf:
        for f in want:
            tf.add(f, arcname=f"scores/{f.name}")
        for f in meta:
            if f.exists():
                tf.add(f, arcname=f"meta/{cell_line}__{run_name}/{f.name}")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
    ap.add_argument("--arm", required=True, choices=list(C.ARMS_BY_NAME))
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--esm2", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default="_work")
    ap.add_argument("--coexp-shuffle", action="store_true")
    ap.add_argument("--skip-scoring", action="store_true")
    ap.add_argument(
        "--heldout-folds",
        default=None,
        help="comma-separated folds withheld from fine-tuning and scored "
        "(default: the pre-registered pair)",
    )
    ap.add_argument(
        "--upstream-sampling",
        action="store_true",
        help="draw the batch's gene positions exactly as upstream does, letting a row "
        "lose its own perturbed gene and carry no perturbation flag",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="suffix distinguishing this run's outputs from the arm's main run",
    )
    a = ap.parse_args(argv)

    line, arm = a.cell_line, C.ARMS_BY_NAME[a.arm]
    device = "cuda"
    heldout = (
        tuple(int(x) for x in a.heldout_folds.split(",") if x != "")
        if a.heldout_folds
        else tuple(C.MODEL_HELDOUT_FOLDS)
    )
    force_perturbed = not a.upstream_sampling
    run_name = f"{arm.name}__{a.tag}" if a.tag else arm.name
    bundle_root = Path(a.bundle) if a.bundle else paths.bundle_root()
    cells_root = Path(a.cells) if a.cells else paths.cells_root()
    out_dir = paths.ensure(Path(a.out) if a.out else paths.arm_dir(run_name, line))
    work = paths.ensure(Path(a.work))
    print(f"=== {line} / {run_name} (heldout folds {heldout}, "
          f"perturbed gene forced into the sequence: {force_perturbed}) ===", flush=True)
    print(f"bundle {bundle_root}\ncells  {cells_root}\nout    {out_dir}", flush=True)

    readout = pd.read_parquet(paths.readout_table(line))
    targets = load_targets(bundle_root, line)
    if not np.array_equal(readout["gene_id"].to_numpy().astype(str), targets["readout_gene_id"]):
        raise RuntimeError("read-out gene order differs between the handoff table and the bundle")

    idx = np.load(Path(cells_root) / line / "perturbation_index.npz", allow_pickle=False)
    if not np.array_equal(idx["pert_target"].astype(str), targets["pert_target"]):
        raise RuntimeError("perturbation order differs between the cell matrices and the bundle")
    folds = {
        "fold": idx["fold"].astype(np.int64),
        "groups": idx["groups"].astype(np.int64),
        "model_heldout": idx["model_heldout"].astype(bool),
    }
    split = training_split(folds, C.SEED, heldout)
    print(
        f"split: {split['n_train']} train / {split['n_val']} val / {split['n_scored']} scored "
        f"perturbations ({split['n_train_groups']}/{split['n_val_groups']}/"
        f"{split['n_scored_groups']} groups)",
        flush=True,
    )

    cells = datamod.load_line(
        cells_root, line, targets["self_col"], targets["readout_gene_name"], split, device
    )
    print(f"cells: {cells.meta}", flush=True)

    cache = paths.checkpoint_cache()
    ckpt_meta = ckptmod.fetch(cache)
    if arm.table == "scgpt":
        table, table_report = None, {"table": "scgpt", "note": "the published gene embedding rows"}
    else:
        esm2 = Path(a.esm2) if a.esm2 else paths.esm2_table_file()
        table, table_report = substituted_table(arm, esm2, targets["readout_gene_id"])
    print(f"gene table: {json.dumps(table_report)}", flush=True)

    try:
        loss_check = trainmod.check_loss_matches_upstream()
    except Exception as exc:  # the check is evidence, not a dependency
        loss_check = {"error": f"{type(exc).__name__}: {exc}"}
    print(f"loss against scgpt/loss.py: {json.dumps(loss_check)}", flush=True)

    scgpt_positions = readout["scgpt_position"].to_numpy(dtype=np.int64)
    model, build_report = modelmod.build_arm_model(arm, cache, table, scgpt_positions, device)
    print(json.dumps({k: v for k, v in build_report.items() if k != "substitutions"}, indent=2), flush=True)

    blocks = {}
    blocks["init"] = extract_stage(model, "init", run_name, readout, targets, device)
    print(f"init features in {blocks['init']['diagnostics']['seconds']}s", flush=True)
    np.savez_compressed(
        out_dir / "gene_table_init.npz",
        resolved=blocks["init"]["gene_table"],
        table=model.encoder.table.detach().float().cpu().numpy(),
        gene_id=targets["readout_gene_id"],
        gene_name=targets["readout_gene_name"],
    )

    history = trainmod.finetune(
        model, cells, targets["effect"], out_dir, force_perturbed=force_perturbed
    )
    print(
        f"training done in {history['total_seconds']}s; the recipe selects epoch "
        f"{history['selected_epoch']} at validation Pearson "
        f"{history['selected_val_pearson']:.5f}; the masked-MSE rule would select epoch "
        f"{history['mse_selected_epoch']} at loss {history['best_val_loss']:.4f}; "
        f"response agreement at the selected epoch "
        f"{json.dumps(history['best_val_delta_pearson'])}",
        flush=True,
    )

    blocks["final"] = extract_stage(model, "final", run_name, readout, targets, device)
    print(f"final features in {blocks['final']['diagnostics']['seconds']}s", flush=True)

    # ---- durable model artifacts ---------------------------------------- #
    torch.save(history.pop("best_state"), out_dir / "model_best.pt")
    torch.save(history.pop("final_state"), out_dir / "model_final.pt")
    mse_state = history.pop("mse_state")
    if mse_state is not None:
        torch.save(mse_state, out_dir / "model_best_masked_mse.pt")
    del mse_state
    proj = getattr(model.encoder, "proj", None)
    np.savez_compressed(
        out_dir / "gene_table_final.npz",
        resolved=blocks["final"]["gene_table"],
        table=model.encoder.table.detach().float().cpu().numpy(),
        projection=(
            proj.weight.detach().float().cpu().numpy()
            if isinstance(proj, torch.nn.Linear)
            else np.zeros((0, 0), dtype=np.float32)
        ),
        gene_id=targets["readout_gene_id"],
        gene_name=targets["readout_gene_name"],
    )
    (out_dir / "loss_history.json").write_text(
        json.dumps(
            {
                "history": history["history"],
                "best_epoch": history["best_epoch"],
                "selected_epoch": history["selected_epoch"],
                "selected_val_pearson": history["selected_val_pearson"],
                "selection_rule": history["selection_rule"],
                "pearson_rule_stopped_at_epoch": history["pearson_rule_stopped_at_epoch"],
                "mse_selected_epoch": history["mse_selected_epoch"],
                "mse_rule_stopped_at_epoch": history["mse_rule_stopped_at_epoch"],
                "best_val_loss": history["best_val_loss"],
                "alternative_rule": history["alternative_rule"],
                "best_val_delta_pearson": history["best_val_delta_pearson"],
                "total_seconds": history["total_seconds"],
                "gene_subsampling": history["gene_subsampling"],
            },
            indent=2,
        )
    )
    train_config = {
        "cell_line": line,
        "arm": arm.name,
        "run_name": run_name,
        "arm_spec": {
            "table": arm.table,
            "trainable_table": arm.trainable_table,
            "projected": arm.project,
            "note": arm.note,
        },
        "recipe": vars(C.RECIPE),
        "recipe_source": (
            "scGPT tutorials/Tutorial_Perturbation.ipynb; model class "
            "scgpt.model.generation_model.TransformerGenerator"
        ),
        "deviation": (
            "the batch's own perturbed genes are always inside the sampled "
            f"{C.RECIPE.max_seq_len}-gene sequence; upstream samples uniformly and can drop them"
        ),
        "gene_subsampling_counters": history["gene_subsampling"],
        "force_perturbed_gene_in_sequence": force_perturbed,
        "sampling": (
            "upstream: the batch's gene positions are drawn uniformly and a row can lose "
            "its own perturbed gene"
            if not force_perturbed
            else "the batch's own perturbed genes are kept inside the sampled sequence"
        ),
        "split": {k: v for k, v in split.items() if not isinstance(v, np.ndarray)},
        "model_heldout_folds": list(heldout),
        "preregistered_heldout_folds": list(C.MODEL_HELDOUT_FOLDS),
        "cells": cells.meta,
        "n_train_cells": history["n_train_cells"],
        "seed": C.SEED,
        "table_report": table_report,
        "build_report": build_report,
    }
    (out_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, default=str))
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "base_checkpoint": {
                    "repo": C.SCGPT_REPO,
                    "revision": C.SCGPT_REVISION,
                    "files": ckpt_meta,
                },
                "substitutions": build_report["substitutions"],
                "loss_against_upstream": loss_check,
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "scgpt": build_report["scgpt_version"],
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
                "extraction": {
                    stage: blocks[stage]["diagnostics"] for stage in ("init", "final")
                },
                "gene_table_note": blocks["final"]["gene_table_note"],
            },
            indent=2,
            default=str,
        )
    )
    for stage in ("init", "final"):
        blocks[stage]["query_summary"].to_parquet(out_dir / f"query_summary_{stage}.parquet")

    # ---- the model's own prediction, with no translator ------------------ #
    all_rows = np.arange(targets["pert_target"].size)
    flagged, unflagged = trainmod.predict_profiles(model, cells, all_rows, 32, amp=False)
    direct = flagged - unflagged[None, :]
    ctrl_mean_log1p = cells.x_ctrl.mean(dim=0).float().cpu().numpy()
    np.savez_compressed(
        out_dir / "direct_prediction.npz",
        delta=direct.astype(np.float32),
        flagged=flagged.astype(np.float32),
        unflagged=unflagged.astype(np.float32),
        control_mean_log1p=ctrl_mean_log1p,
        pert_target=targets["pert_target"],
        gene_id=targets["readout_gene_id"],
        fold=folds["fold"],
    )
    # how perturbation-specific the prediction is, written here so the merge job
    # does not have to stage sixteen prediction matrices to recompute it
    spread = followup.spread_row(
        flagged, unflagged, ctrl_mean_log1p.astype(np.float64), line, run_name
    )
    (out_dir / "prediction_spread.json").write_text(json.dumps(spread, indent=2))
    print(json.dumps(spread), flush=True)
    del model
    torch.cuda.empty_cache()

    if a.skip_scoring:
        print("scoring skipped", flush=True)
        return 0

    # ---- scoring, through run 3's rule ----------------------------------- #
    bdir = score.stage_bundle(bundle_root, work / "bundle", line)
    installed = []
    for stage in ("init", "final"):
        b = blocks[stage]
        installed.append(
            score.write_block(
                bdir, line, b["value_key"], C.ANCHOR_VALUE, all_rows, b["gene_table"],
                b["value_forms"], b["values"], b["value_keep"],
            )
        )
        installed.append(
            score.write_block(
                bdir, line, b["flag_key"], C.ANCHOR_VALUE, all_rows, b["gene_table"],
                b["flag_forms"], b["flag_values"], b["flag_value_keep"],
            )
        )
    print(json.dumps(installed, indent=2), flush=True)

    summary = score.score_bundle(
        bdir, line, run_name, paths.scores_dir(), a.coexp_shuffle, device=device,
        eval_folds=heldout,
    )
    summary["installed_blocks"] = installed
    summary["heldout_folds"] = list(heldout)
    summary["force_perturbed_gene_in_sequence"] = force_perturbed
    # both baselines for the model's own prediction: the difference between the
    # flagged and the unflagged forward pass, and the published evaluation's
    # comparison against the measured control profile. The unflagged query is
    # never trained, so it is reported but not relied on.
    summary["direct_prediction"] = score.score_direct(
        bdir, line, direct, paths.scores_dir(), run_name, device=device, eval_folds=heldout,
        baseline="unflagged_model_output",
    )
    summary["direct_prediction_control_mean"] = score.score_direct(
        bdir, line, flagged - cells.x_ctrl.mean(dim=0).double().cpu().numpy()[None, :],
        paths.scores_dir(), run_name, device=device, eval_folds=heldout,
        baseline="measured_control_mean",
    )
    (paths.scores_dir() / f"summary_{line}__{run_name}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "bundle_inventory"}, indent=2, default=str), flush=True)

    pack = pack_run(line, run_name, out_dir)
    print(f"packed {pack}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

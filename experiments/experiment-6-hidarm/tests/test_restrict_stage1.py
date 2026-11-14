"""Which stage-1 units are unquotable until they are restricted, and which are not.

The failure this guards against is quiet and total: a five-fold row for a
fine-tuned checkpoint is a **training** number for three folds in five, and it
looks exactly like a legitimate score. Nothing downstream can tell them apart,
so the selection of which units get restricted is itself load-bearing and is
tested here rather than trusted to a glob.
"""

from __future__ import annotations

import json

import pytest

from scfm_hidarm import config as C
from scfm_hidarm import restrict_stage1


def write_unit(root, unit, cell_line, model, layer, k=C.K_PRIMARY, bundles=("b0",)):
    d = root / unit
    d.mkdir(parents=True, exist_ok=True)
    (d / "unit.json").write_text(json.dumps(
        {"unit": unit, "cell_line": cell_line, "model": model, "layer": layer, "k": k}
    ))
    for b in bundles:
        (d / b).mkdir(exist_ok=True)
        (d / b / f"per_pert_{cell_line}.parquet").write_bytes(b"")
    return d


def test_both_fine_tuned_models_and_frozen_scgpt_are_restricted():
    assert set(restrict_stage1.RESTRICTED_MODELS) == set(C.FT_MODELS) | {"scgpt"}


def test_scprint_is_not_restricted():
    """scPRINT never trained on this screen, so all five folds are held out for it."""
    assert "scprint" not in restrict_stage1.RESTRICTED_MODELS
    assert "transcriptformer" not in restrict_stage1.RESTRICTED_MODELS


def test_unit_dirs_selects_by_model_and_line_and_finds_every_bundle(tmp_path, monkeypatch):
    root = tmp_path / "sweep" / "stage1"
    write_unit(root, "hepg2-scgpt-l12-k64", "hepg2", "scgpt", 12, bundles=("b0", "b1"))
    write_unit(root, "hepg2-scgpt-ft-l12-k64", "hepg2", "scgpt_ft", 12)
    write_unit(root, "hepg2-scprint-l8-k64", "hepg2", "scprint", 8)
    write_unit(root, "jurkat-scgpt-l12-k64", "jurkat", "scgpt", 12)
    monkeypatch.setattr(restrict_stage1.paths, "sweep_dir", lambda name: root)

    got = restrict_stage1.unit_dirs("stage1", "hepg2", restrict_stage1.RESTRICTED_MODELS)
    assert {(u["model"], u["dir"].name) for u in got} == {
        ("scgpt", "b0"), ("scgpt", "b1"), ("scgpt_ft", "b0"),
    }
    # scPRINT is excluded by model, Jurkat by cell line
    assert not [u for u in got if u["model"] == "scprint"]


def test_a_bundle_with_no_per_pert_table_is_not_offered(tmp_path, monkeypatch):
    root = tmp_path / "sweep" / "stage1"
    d = write_unit(root, "hepg2-scgpt-ft-l12-k64", "hepg2", "scgpt_ft", 12, bundles=())
    (d / "b0").mkdir()
    monkeypatch.setattr(restrict_stage1.paths, "sweep_dir", lambda name: root)
    assert restrict_stage1.unit_dirs("stage1", "hepg2", ("scgpt_ft",)) == []


def test_no_units_is_a_failure_and_not_an_empty_result(tmp_path, monkeypatch):
    root = tmp_path / "sweep" / "stage1"
    root.mkdir(parents=True)
    monkeypatch.setattr(restrict_stage1.paths, "sweep_dir", lambda name: root)
    with pytest.raises(RuntimeError, match="cannot be quoted"):
        restrict_stage1.run("hepg2")


def test_the_eval_folds_are_run4s_two():
    assert tuple(C.MODEL_HELDOUT_FOLDS) == (0, 1)

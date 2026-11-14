"""The merge, on planted rows whose right answer is arithmetic.

The failure modes this pins are all silent. A reference row merged into the run
table overwrites a published number with a re-run of it. A shuffled-label band
carried across depths lets a layer that never had draws be quoted as though it
did. A null band taken as the max over the wrong variants turns a width control
into a machinery control. None of those raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scfm_hidarm import analysis
from scfm_hidarm import config as C

K = C.K_PRIMARY
HID = C.arm_name(C.model_stem("scgpt", "hid", 12, K))
PERM = C.arm_name(C.model_stem("scgpt", "hidperm", 12, K))
RAND = C.arm_name(C.model_stem("scgpt", "hidrand", 12, K))
RP = C.arm_name(C.model_stem("scgpt", "hidrp", 12, K))


def test_decode_arm_reads_a_run6_key_and_leaves_a_reference_empty():
    assert analysis.decode_arm(HID) == {"model": "scgpt", "variant": "hid",
                                        "layer": 12, "k": K}
    assert analysis.decode_arm(f"{HID}+shuffled_labels_d3")["variant"] == "hid"
    assert analysis.decode_arm("coexp") == {}
    assert analysis.decode_arm("scgpt_emb") == {}
    assert analysis.decode_arm("null_randpair") == {}


def planted() -> pd.DataFrame:
    """Four run-6 arms, one reference and one shuffled draw, in one cell."""
    rows = [
        (HID, 0.30, 0.20), (RP, 0.28, 0.19), (PERM, 0.05, 0.04), (RAND, 0.02, 0.01),
        ("coexp", 0.24, 0.15), (f"{HID}+shuffled_labels_d0", 0.06, 0.03),
    ]
    return pd.DataFrame(
        [
            {"cell_line": "hepg2", "split": "grouped", "translator": "mlp",
             "arm": a, "model": "", "feature_dim": 640, "n_perturbations": 524,
             "n_groups": 226, "largest_group": 149, "seconds": 90.0,
             "discrimination_ours_residual_mean": 0.5,
             C.PRIMARY_METRIC: p, C.ORDERING_METRIC: o}
            for a, p, o in rows
        ]
    )


def annotated(monkeypatch, shuffled_here: bool = True) -> pd.DataFrame:
    unit = {
        "unit": "u", "cell_line": "hepg2", "model": "scgpt", "layer": 12, "k": K,
        "captured_energy": {"k": K, "ratio": 0.9242, "meets_target": False},
        "shuffle_bases": [HID] if shuffled_here else [],
        "unit_dir": "/nonexistent",
    }
    monkeypatch.setattr(analysis, "unit_records", lambda sweep: [unit])
    return analysis.annotate(planted(), "stage1")


def test_annotate_attaches_the_cells_own_captured_energy_to_every_run6_row(monkeypatch):
    a = annotated(monkeypatch)
    run6 = a[a["model"] == "scgpt"]
    assert len(run6) == 5  # four variants plus the shuffled draw
    assert np.allclose(run6["captured_energy_ratio"], 0.9242)
    assert not run6["captured_energy_meets_target"].any()
    # a reference row has no run-6 cell and therefore no coverage number
    assert a[a["arm"] == "coexp"]["captured_energy_ratio"].isna().all()


def test_reference_and_shuffle_rows_are_flagged_and_separable(monkeypatch):
    a = annotated(monkeypatch)
    assert a[a["arm"] == "coexp"]["is_reference"].all()
    assert not a[a["arm"] == HID]["is_reference"].any()
    draw = a[a["is_shuffle_draw"]]
    assert len(draw) == 1 and draw["shuffle_base"].iloc[0] == HID


def test_the_band_is_hidperm_and_hidrand_and_never_hidrp(monkeypatch):
    """hidrp is a second REAL arm through a random basis, not a null."""
    a = annotated(monkeypatch)
    b = analysis.null_bands(a, C.PRIMARY_METRIC)
    assert len(b) == 1
    r = b.iloc[0]
    assert r["arm_score"] == 0.30
    assert r["hidperm_max"] == 0.05
    assert r["hidrand_max"] == 0.02
    assert r["hidrp_max"] == 0.28          # reported beside, never folded in
    assert bool(r["clears_hidperm"])
    assert r["shuffle_max"] == 0.06 and bool(r["clears_shuffle"])


def test_a_depth_without_its_own_draws_reports_no_shuffle_band(monkeypatch):
    """NaN, not another depth's number: the row says which nulls it actually had."""
    a = annotated(monkeypatch, shuffled_here=False)
    a = a[~a["is_shuffle_draw"]]
    b = analysis.null_bands(a, C.PRIMARY_METRIC)
    r = b.iloc[0]
    assert np.isnan(r["shuffle_max"]) and r["shuffle_n"] == 0
    assert r["clears_shuffle"] is None
    assert bool(r["clears_hidperm"])       # the machinery null is still there


def test_an_arm_inside_its_band_does_not_clear_it(monkeypatch):
    a = annotated(monkeypatch)
    a.loc[a["arm"] == HID, C.PRIMARY_METRIC] = 0.04   # below hidperm's 0.05
    b = analysis.null_bands(a, C.PRIMARY_METRIC)
    assert not bool(b.iloc[0]["clears_hidperm"])


def test_the_summary_columns_carry_both_metric_cells():
    assert C.PRIMARY_METRIC in analysis.SUMMARY_COLUMNS
    assert C.ORDERING_METRIC in analysis.SUMMARY_COLUMNS
    assert "captured_energy_ratio" in analysis.SUMMARY_COLUMNS
    assert "shuffle_draws_taken" in analysis.SUMMARY_COLUMNS

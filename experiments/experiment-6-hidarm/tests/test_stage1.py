"""The stage-1 driver's bookkeeping, which is where a wave silently loses an arm.

Nothing here fits a translator or touches a block. What it pins is the set of
decisions that decide *which* arms get scored and *which* nulls a row may be
quoted against --- every one of which fails by producing a plausible table
rather than an error:

- a unit or key carrying ``__`` makes run 3's reader mis-parse the block and the
  arm vanishes from the scored set with no error at all (this cost run 4 a round
  of re-scoring);
- shuffled-label draws are taken at the pre-registered primary layer only, so a
  depth that did not get them must not come back claiming it did;
- the reference arm list must contain no ``_resp`` arm, because run 6 emits no
  decoder-head reading at all and a head arm in the list would silently
  reintroduce the comparison the design makes post-hoc.
"""

from __future__ import annotations

import numpy as np
import pytest

from scfm_hidarm import config as C
from scfm_hidarm import stage1


def test_no_unit_name_or_arm_key_contains_the_forbidden_separator():
    for line in C.CELL_LINES:
        for model in C.MODELS:
            for layer in C.sweep_layers(model):
                name = stage1.unit_name(line, model, layer, C.K_PRIMARY)
                assert C.FORBIDDEN_IN_KEY not in name, name
                for arm in stage1.unit_arms(model, layer, C.K_PRIMARY):
                    assert C.FORBIDDEN_IN_KEY not in arm.rsplit("_resp", 1)[0], arm


def test_a_unit_supplies_exactly_the_four_variants():
    arms = stage1.unit_arms("scgpt", 12, C.K_PRIMARY)
    assert len(arms) == len(stage1.UNIT_VARIANTS) == 4
    assert {C.run_of_stem(a[: -len("_resp")])["variant"] for a in arms} == set(
        stage1.UNIT_VARIANTS
    )


def test_shuffle_draws_are_taken_at_the_primary_layer_only():
    """A depth without its own draws may not be quoted as though it had them."""
    for model in ("scgpt", "scprint", "scgpt_ft"):
        primary = C.primary_layer(model)
        assert stage1.shuffle_bases(model, primary, C.K_PRIMARY) == (
            C.arm_name(C.model_stem(model, "hid", primary, C.K_PRIMARY)),
        )
        for layer in C.sweep_layers(model):
            if layer != primary:
                assert stage1.shuffle_bases(model, layer, C.K_PRIMARY) == ()


def test_the_shuffle_base_is_the_arms_own_key():
    """Run 4's flag-family band and the frozen models' band are both refused."""
    base = stage1.shuffle_bases("scgpt_ft", C.primary_layer("scgpt_ft"), C.K_PRIMARY)[0]
    assert C.run_of_stem(base[: -len("_resp")])["model"] == "scgpt_ft"


def test_reference_arms_carry_no_head_reading():
    refs = stage1.reference_arms()
    assert not [a for a in refs if a.endswith("_resp")]
    assert "coexp" in refs and "null_randpair" in refs
    assert all(f"{m}_emb" in refs for m in C.REFERENCE_EMB_MODELS)


def test_plan_covers_every_sweep_layer_and_the_pilot_covers_one():
    full = stage1.plan("hepg2", ("scgpt", "scprint"), None, pilot=False)
    assert full == [("scgpt", l) for l in C.sweep_layers("scgpt")] + [
        ("scprint", l) for l in C.sweep_layers("scprint")
    ]
    pilot = stage1.plan("hepg2", ("scgpt", "scprint"), None, pilot=True)
    assert pilot == [("scgpt", C.primary_layer("scgpt")),
                     ("scprint", C.primary_layer("scprint"))]


def test_stored_block_is_read_at_k_store_not_at_the_scored_k():
    """The ladder is a column prefix of what is stored; nothing re-projects."""
    p = stage1.stored_block("scgpt", "hepg2", "hid", 12)
    assert p.name == f"{C.model_stem('scgpt', 'hid', 12, C.K_STORE)}.npz"
    assert f"k{C.K_STORE}" in p.name


def test_a_missing_extracted_block_fails_rather_than_skipping(tmp_path):
    with pytest.raises(RuntimeError, match="cannot compute it"):
        stage1.install_unit("hepg2", "transcriptformer", 12, C.K_PRIMARY, tmp_path)


def test_captured_energy_row_is_absent_rather_than_invented(tmp_path, monkeypatch):
    """A cell with no diagnostic returns None; it never returns a default."""
    monkeypatch.setattr(stage1.paths, "blocks_dir", lambda *a, **k: tmp_path / "nope")
    assert stage1.captured_energy_row("scgpt", "hepg2", 12, C.K_PRIMARY) is None


def test_arm_dim_at_the_primary_rung_is_ten_slots_times_k():
    """640, not 160: the rung moved on 2026-08-31 and the width matching with it."""
    assert C.N_SLOTS == 10
    assert C.N_SLOTS * C.K_PRIMARY == 640
    # k = 1 (dim 10) is the rung matched to null_randpair's 11; 640 is matched
    # to no run 3 reference, which is why hidrand is the width control there
    assert C.N_SLOTS * 1 == 10
    assert np.isin(1, C.K_LADDER)

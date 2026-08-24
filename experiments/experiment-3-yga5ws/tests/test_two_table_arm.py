"""The two-table arm's reading rule, and the agreement statistic behind it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scfm_pfm import analysis


def _per_pert(scores: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for arm, values in scores.items():
        for i, v in enumerate(values):
            rows.append({
                "cell_line": "hepg2", "scope": "full", "split": "grouped",
                "translator": "mlp", "target_source": "run1_all_controls",
                "arm": arm, "perturbation": f"p{i}",
                "residual_weighted": v, "centered_weighted": v,
            })
    return pd.DataFrame(rows)


def test_agreement_is_read_on_ranks_not_on_sizes():
    """A monotone but curved relation is agreement; Pearson understates it."""
    x = np.linspace(0.01, 1.0, 40)
    d = _per_pert({"scgpt_emb": list(x), "esm2_3b_emb": list(x**3)})
    got = analysis.arm_agreement(d, pairs=(("scgpt_emb", "esm2_3b_emb"),))
    row = got[got.metric == "residual_weighted"].iloc[0]
    assert row.spearman == pytest.approx(1.0, abs=1e-9)
    assert row.pearson < 0.95
    assert row.n_perturbations == 40


def test_agreement_needs_both_arms_and_pairs_on_the_perturbation():
    """An arm the block does not hold produces no row rather than a wrong one."""
    d = _per_pert({"scgpt_emb": [0.1, 0.2, 0.3, 0.4]})
    assert analysis.arm_agreement(d, pairs=(("scgpt_emb", "esm2_3b_emb"),)).empty


@pytest.mark.parametrize(
    "beats,spearman,expected",
    [
        (False, 0.80, "redundant"),      # no gain, same perturbations
        (True, 0.10, "complementary"),   # a gain, different perturbations
        (True, 0.80, "mixed"),           # a gain but the parts agree
        (False, 0.10, "mixed"),          # no gain but the parts disagree
        (False, 0.40, "mixed"),          # between the two thresholds
        (True, float("nan"), "mixed"),   # no agreement measurement
    ],
)
def test_the_reading_rule_covers_every_case_it_was_written_for(beats, spearman, expected):
    """Fixed before the arm was scored: no case may fall through to a tidy read."""
    assert analysis.concat_reading(beats, spearman) == expected

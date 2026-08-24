"""The group-resampled interval, and the label that decides which rows it uses.

Two things here decide published intervals. The cluster bootstrap has to be the
median of the whole resampled multiset, not of one row per group, and it has to
reduce to the row bootstrap when every group is a singleton. And a wave that
scored the same arms against a different target has to keep its own label on its
per-perturbation rows, or those rows silently replace the main target's and
every paired difference moves onto the wrong target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scfm_pfm import analysis, metrics


def test_weighted_median_matches_numpy_on_the_expanded_multiset():
    values = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    gidx = np.array([0, 0, 1, 2, 2])
    counts = np.array([[1, 1, 1], [2, 0, 1], [0, 3, 0], [1, 1, 0]], dtype=np.int32)
    got = metrics._weighted_median(values, gidx, counts)
    for r, c in enumerate(counts):
        expanded = np.repeat(values, c[gidx])
        assert np.isclose(got[r], np.median(expanded)), (r, c)


def test_singleton_groups_reproduce_the_row_bootstrap():
    rng = np.random.default_rng(0)
    v = rng.normal(size=400)
    row = metrics.bootstrap_median(v, n_resamples=4000, seed=7)
    grouped = metrics.bootstrap_median_grouped(v, np.arange(v.size), n_resamples=4000, seed=7)
    assert grouped["median"] == row["median"]
    width_row = row["hi"] - row["lo"]
    width_grp = grouped["hi"] - grouped["lo"]
    assert abs(width_grp - width_row) < 0.15 * width_row


def test_one_dominant_group_widens_the_interval():
    rng = np.random.default_rng(1)
    v = np.concatenate([rng.normal(0.4, 0.05, 150), rng.normal(0.0, 0.2, 350)])
    groups = np.concatenate([np.zeros(150, dtype=int), np.arange(1, 351)])
    row = metrics.bootstrap_median(v, seed=3)
    grouped = metrics.bootstrap_median_grouped(v, groups, seed=3)
    assert grouped["n_groups"] == 351
    assert grouped["largest_group"] == 150
    assert (grouped["hi"] - grouped["lo"]) > 1.5 * (row["hi"] - row["lo"])


def test_the_paired_group_bootstrap_uses_one_draw_for_both_arms():
    rng = np.random.default_rng(2)
    a = rng.normal(size=300)
    b = a + 0.05  # a constant offset: every resample must give the same difference
    groups = np.repeat(np.arange(60), 5)
    boot = metrics.bootstrap_median_difference_grouped(a, b, groups, seed=5)
    assert np.isclose(boot["difference"], -0.05, atol=1e-9)
    assert np.isclose(boot["lo"], -0.05, atol=1e-9)
    assert np.isclose(boot["hi"], -0.05, atol=1e-9)


def _wave(tmp_path, name, target_source, value, with_label_in_per_pert):
    d = tmp_path / name
    d.mkdir()
    keys = {"cell_line": "hepg2", "scope": "full", "split": "grouped",
            "arm": "coexp", "translator": "mlp"}
    runs = pd.DataFrame([{**keys, "target_source": target_source, "score": value}])
    runs.to_parquet(d / "runs_shard0.parquet", index=False)
    pp = pd.DataFrame([{**keys, "perturbation": g, "residual_weighted": value}
                       for g in ("A", "B", "C")])
    if with_label_in_per_pert:
        pp["target_source"] = target_source
    pp.to_parquet(d / "per_perturbation_shard0.parquet", index=False)
    return d


def test_a_later_wave_with_its_own_target_does_not_replace_the_main_rows(tmp_path):
    main = _wave(tmp_path, "sweep", None, 1.0, with_label_in_per_pert=False)
    alt = _wave(tmp_path, "alt", "hepg2__ctrl0", 2.0, with_label_in_per_pert=False)
    runs, per_pert = analysis.load_sweep(main, alt)
    assert set(runs.target_source) == {"run1_all_controls", "hepg2__ctrl0"}
    assert set(per_pert.target_source) == {"run1_all_controls", "hepg2__ctrl0"}
    kept = per_pert[per_pert.target_source == "run1_all_controls"]
    assert kept.shape[0] == 3
    assert np.allclose(kept.residual_weighted.to_numpy(), 1.0)

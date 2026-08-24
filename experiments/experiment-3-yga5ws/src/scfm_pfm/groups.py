"""Response-similarity groups, recovered from the bundle's effect matrix.

The primary split holds whole groups of correlated perturbations out together
(``translate/splits.py``), but the per-run bootstrap resampled individual
perturbations. Individual perturbations are not independent draws: HepG2's 524
perturbations fall into 226 response groups, one of which holds 149 of them, so
a row-wise interval is narrower than the design supports.

This module recomputes the same grouping the runner used -- average linkage on
1 - Pearson between measured effect vectors, cut at ``C.GROUP_CORR_THRESHOLD``
-- keyed by perturbation name, so any scored table can be re-bootstrapped over
groups instead of rows. Reproduction is checked against the ``n_groups`` and
``largest_group`` columns the sweep already recorded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import config as C
from .translate.splits import effect_groups


def load_effect(line: str, bundle_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """(effect, perturbation names) for one cell line's bundle."""
    z = np.load(Path(bundle_root) / line / "targets.npz", allow_pickle=False)
    return np.asarray(z["effect"], dtype=np.float64), z["pert_target"].astype(str)


class GroupTable:
    """Group id per perturbation name, per cell line and per scored row set.

    A row set is whatever perturbations one run actually scored; different arms
    cover different perturbations, so the clustering is redone per distinct set
    and cached on its hash.
    """

    def __init__(self, bundle_root: Path, threshold: float = C.GROUP_CORR_THRESHOLD):
        self.bundle_root = Path(bundle_root)
        self.threshold = float(threshold)
        self._effect: dict[str, tuple[np.ndarray, dict[str, int]]] = {}
        self._cache: dict[tuple[str, str], dict[str, int]] = {}

    def _line(self, line: str) -> tuple[np.ndarray, dict[str, int]]:
        if line not in self._effect:
            eff, names = load_effect(line, self.bundle_root)
            self._effect[line] = (eff, {n: i for i, n in enumerate(names)})
        return self._effect[line]

    def groups(self, line: str, names: list[str]) -> np.ndarray:
        """Group id for each name, in the order given."""
        eff, index = self._line(line)
        key = (line, hashlib.sha1("\x00".join(sorted(names)).encode()).hexdigest())
        if key not in self._cache:
            ordered = sorted(names, key=lambda n: index[n])
            rows = np.array([index[n] for n in ordered], dtype=np.int64)
            g = effect_groups(eff[rows], self.threshold)
            self._cache[key] = dict(zip(ordered, g.tolist(), strict=True))
        lut = self._cache[key]
        return np.array([lut[n] for n in names], dtype=np.int64)

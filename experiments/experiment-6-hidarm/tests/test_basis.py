"""The ``d -> k`` reduction, checked against arithmetic and against the real tables.

Invocation (from the run directory):  PYTHONPATH=src pytest tests/test_basis.py -q

Three failure classes, none of which raises on its own, and each of which would
quietly change every number run 6 reports.

**A basis that is not what it claims to be.** A sign flip between two fits, an
eigenvalue ordering that is not descending, a ``k = 16`` block that is not the
first sixteen columns of the ``k = 64`` block --- all of those still produce a
``W``, still project, and still score. The ladder in particular is a
pre-registration claim: ``K_LADDER`` is declared free *because* it is a column
prefix, so if it is not one then reading a second rung is a second decision and
the run has quietly acquired a degree of freedom. That is asserted at ``max abs
diff 0.0``, not to a tolerance.

**An accumulator that is not the matrix it stands in for.** ``SecondMoment``
exists so that raw ``dh`` never has to be held; the whole affordability argument
rests on it giving exactly what a batch ``X'X`` would. Checked against the
one-shot product, and the centred variant against explicit centring.

**A basis fitted on the perturbations it is used to score.** This is the run's
most contestable choice and the one a reader will go looking for, so it is
checked on the **real** handoff parquets rather than on a fixture: the pool
sizes are asserted at their measured values (365 HepG2, 512 Jurkat), the draw is
asserted reproducible at its seed, and the overlap with the in-scope set is
asserted to be exactly zero on **both** identifiers.

Where a value is asserted, the expected number is computed here by hand or read
off a planted construction, never by calling the code under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scfm_hidarm import config as C
from scfm_hidarm.basis import fit as F

#: the measured sizes of the two screened pools, counted from run 2's handoff
#: tables on 2026-08-31. These are load-bearing numbers rather than incidental
#: ones: the basis pool being ~2/3 the size of the scored set is what makes a
#: 64-gene stratified draw from it possible at all, and a change in either
#: number means the stratum filter or the read-out set moved under the run.
POOL_SIZES = {
    "hepg2": {"scored": 524, F.POOL_PRIMARY: 365},
    "jurkat": {"scored": 705, F.POOL_PRIMARY: 512},
}


def _planted(d: int, eigs: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A symmetric PSD matrix with known orthonormal eigenvectors and eigenvalues."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q @ np.diag(eigs) @ Q.T, Q


# --------------------------------------------------------------------------- #
# the eigendecomposition
# --------------------------------------------------------------------------- #
def test_eigh_recovers_planted_directions_and_eigenvalues() -> None:
    """A hand-built covariance returns its planted directions, up to sign.

    Up to sign because an eigenvector is only defined up to sign; the *choice*
    of sign is asserted separately to be deterministic.
    """
    d = 12
    eigs = np.array([9.0, 7.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
    M, Q = _planted(d, eigs)

    got = F.eigendecompose(M, k=d)
    assert np.allclose(got["eigenvalues"], eigs, atol=1e-10), got["eigenvalues"]
    # descending, which is what makes the ladder a prefix
    assert np.all(np.diff(got["eigenvalues"]) <= 1e-12)

    W = got["W"].astype(np.float64)
    for j in range(d):
        planted = Q[:, j]
        recovered = W[:, j]
        cos = abs(float(planted @ recovered))
        assert cos == pytest.approx(1.0, abs=1e-6), f"column {j}: |cos| = {cos}"

    assert got["explained"].sum() == pytest.approx(1.0, abs=1e-12)
    assert got["explained"][0] == pytest.approx(eigs[0] / eigs.sum(), abs=1e-12)


def test_k_ladder_is_a_column_prefix_at_max_abs_diff_zero() -> None:
    """``k = 16`` is EXACTLY the first sixteen columns of ``k = 64``.

    Bit-exact, not to a tolerance. ``config.K_LADDER`` is pre-registered as free
    precisely because reading a lower rung is a column slice of an already-fitted
    block; if it were a refit, every rung would be a separate decision.
    """
    d = 80
    rng = np.random.default_rng(11)
    X = rng.standard_normal((300, d))
    M = X.T @ X

    big = F.eigendecompose(M, k=C.K_STORE)["W"]
    small = F.eigendecompose(M, k=C.K_PRIMARY)["W"]
    diff = np.max(np.abs(big[:, : C.K_PRIMARY] - small))
    assert diff == 0.0, f"max abs diff {diff!r}, expected exactly 0.0"

    for k in C.K_LADDER:
        if k > C.K_STORE:
            continue
        rung = F.eigendecompose(M, k=k)["W"]
        assert np.max(np.abs(big[:, :k] - rung)) == 0.0, f"rung k={k} is not a prefix"


def test_rank_one_input_gives_exactly_one_non_zero_eigenvalue() -> None:
    """One direction in, one eigenvalue out, and it explains all of the energy."""
    d = 20
    rng = np.random.default_rng(3)
    v = rng.standard_normal(d)
    M = 5.0 * np.outer(v, v)

    got = F.eigendecompose(M, k=d)
    vals = got["eigenvalues"]
    assert vals[0] == pytest.approx(5.0 * float(v @ v), rel=1e-10)
    assert np.all(vals[1:] < 1e-10 * vals[0]), vals[:4]
    assert got["explained"][0] == pytest.approx(1.0, abs=1e-10)
    assert got["explained"].sum() == pytest.approx(1.0, abs=1e-12)

    # ``W`` is float32 by contract (it is what is stored and what the extractor
    # projects with), so alignment is asserted at float32 resolution, not 1e-8
    top = got["W"][:, 0].astype(np.float64)
    assert abs(float(top @ (v / np.linalg.norm(v)))) == pytest.approx(1.0, abs=1e-6)


def test_sign_fixing_is_deterministic_across_refits() -> None:
    """Refitting the same matrix gives a bit-identical ``W``.

    LAPACK's sign choice is not stable across builds or across a permuted input,
    so without the explicit pivot rule a refit would write a bit-different basis
    and therefore a bit-different feature block for arithmetically identical
    input --- a difference no downstream check could attribute.
    """
    M, _ = _planted(24, np.linspace(9.0, 0.5, 24), seed=7)
    a = F.eigendecompose(M, k=16)["W"]
    b = F.eigendecompose(M.copy(), k=16)["W"]
    assert a.tobytes() == b.tobytes()

    # and the rule itself: the largest-magnitude entry of each column is positive
    pivot = np.argmax(np.abs(a), axis=0)
    assert np.all(a[pivot, np.arange(a.shape[1])] > 0.0)


def test_eigendecompose_refuses_a_degenerate_matrix() -> None:
    """An all-zero second moment is a broken extraction, not a structureless basis."""
    with pytest.raises(RuntimeError, match="zero total energy"):
        F.eigendecompose(np.zeros((6, 6)), k=3)


# --------------------------------------------------------------------------- #
# the streaming accumulator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunks", [(1,), (7, 7, 7, 4), (25,), (1, 2, 3, 4, 5, 10)])
def test_second_moment_in_chunks_equals_one_shot(chunks: tuple[int, ...]) -> None:
    """Fed in chunks, the accumulator equals ``X.T @ X`` to 1e-8."""
    d = 9
    rng = np.random.default_rng(5)
    X = rng.standard_normal((sum(chunks), d)).astype(np.float32)

    acc = F.SecondMoment(d, centred=False)
    start = 0
    for n in chunks:
        acc.update(X[start : start + n])
        start += n

    expected = X.astype(np.float64).T @ X.astype(np.float64)
    assert acc.n == X.shape[0]
    assert np.max(np.abs(acc.matrix() - expected)) < 1e-8


def test_second_moment_centred_equals_explicit_centring() -> None:
    """The centred variant is the scatter of the centred data, and ``np.cov`` * (n-1)."""
    d = 6
    rng = np.random.default_rng(6)
    X = rng.standard_normal((40, d))

    acc = F.SecondMoment(d, centred=True)
    for chunk in np.array_split(X, 5):
        acc.update(chunk)

    Xc = X - X.mean(axis=0)
    assert np.max(np.abs(acc.matrix() - Xc.T @ Xc)) < 1e-8
    assert np.max(np.abs(acc.matrix() / (len(X) - 1) - np.cov(X, rowvar=False))) < 1e-8
    assert np.max(np.abs(acc.mean() - X.mean(axis=0))) < 1e-12


def test_uncentred_and_centred_differ_when_the_mean_is_not_zero() -> None:
    """The sensitivity arm is a different object, not a relabelling of the primary.

    ``BASIS_CENTRED = False`` is the pre-registered primary and the centred fit
    is the generic-per-gene-response control; if the two coincided the control
    would be vacuous.
    """
    d = 5
    rng = np.random.default_rng(8)
    X = rng.standard_normal((60, d)) + 3.0

    raw = F.SecondMoment(d, centred=False).update(X).matrix()
    cen = F.SecondMoment(d, centred=True).update(X).matrix()
    assert np.max(np.abs(raw - cen)) > 1.0


def test_second_moment_rejects_non_finite_and_wrong_width() -> None:
    """TranscriptFormer writes its self position as nan; it must not be summed in."""
    acc = F.SecondMoment(4, centred=False)
    with pytest.raises(RuntimeError, match="non-finite"):
        acc.update(np.array([[1.0, 2.0, np.nan, 4.0]]))
    with pytest.raises(ValueError):
        acc.update(np.zeros((3, 5)))
    with pytest.raises(RuntimeError, match="saw no rows"):
        F.SecondMoment(4).matrix()


# --------------------------------------------------------------------------- #
# the captured-energy diagnostic
# --------------------------------------------------------------------------- #
def test_captured_energy_is_exactly_one_when_the_bases_agree() -> None:
    """Same basis, same subspace: ratio exactly 1.0, and no resolvable angle.

    The ratio is asserted bit-exact --- it is a trace ratio in float64 over the
    same array twice, so nothing can move it.

    The angles are NOT asserted at 0.0, and the reason is a measured property of
    the stored basis rather than a tolerance chosen to make a test pass. ``W`` is
    float32 by contract, so its columns are orthonormal only to ~1e-7; arccos
    near 1 amplifies that, since ``arccos(1 - eps) ~ sqrt(2 eps)``, and a basis
    against itself measures ~0.012 degrees. That is the diagnostic's resolution
    floor, it is recorded in ``captured_energy``'s docstring, and it is three
    orders of magnitude below any angle the under-spanning case produces --- so
    it costs the diagnostic nothing. Asserted at 0.05 degrees, an order of
    magnitude above the observed floor and still far below a real angle.
    """
    d = 30
    rng = np.random.default_rng(12)
    X = rng.standard_normal((200, d))
    M = X.T @ X
    W = F.eigendecompose(M, k=C.K_STORE if C.K_STORE <= d else d)["W"]

    rows = F.captured_energy(M, W, W, ks=(1, 4, 16))
    assert [r["k"] for r in rows] == [1, 4, 16]
    for r in rows:
        assert r["ratio"] == 1.0, r
        assert r["max_principal_angle_deg"] < 0.05, r
        assert r["meets_target"] is True
        assert r["diagnostic_only"] is True


def test_captured_energy_of_an_under_spanning_basis_is_below_one_and_reported() -> None:
    """A deliberately wrong basis loses energy, and the loss can never be a gain.

    This asserts the DIRECTION of the error, which is the diagnostic's whole
    reason for being readable: ``W_all`` is by construction the ``k``-dimensional
    subspace capturing the most of the scored perturbations' energy, so no basis
    can exceed it. Under-spanning therefore loses signal and cannot manufacture
    it --- a false null, never a false positive.
    """
    d = 24
    rng = np.random.default_rng(13)
    X = rng.standard_normal((300, d)) * np.linspace(10.0, 0.1, d)
    M = X.T @ X

    W_all = F.eigendecompose(M, k=d)["W"]
    # the deliberately bad basis: the TRAILING eigenvectors, i.e. the directions
    # the scored perturbations move in least
    W_bad = np.ascontiguousarray(W_all[:, ::-1])

    rows = F.captured_energy(M, W_bad, W_all, ks=(1, 4, 8))
    assert rows, "no rung was reported"
    for r in rows:
        assert 0.0 <= r["ratio"] < 1.0, r
        assert r["captured_basis"] < r["captured_all"], r
        assert r["max_principal_angle_deg"] > 1.0, r
        assert r["meets_target"] is False


def test_captured_energy_never_exceeds_one_for_any_basis() -> None:
    """The maximality of ``W_all`` holds for arbitrary orthonormal bases, not just tidy ones."""
    d = 16
    rng = np.random.default_rng(14)
    X = rng.standard_normal((120, d)) @ np.diag(np.linspace(5.0, 0.2, d))
    M = X.T @ X
    W_all = F.eigendecompose(M, k=d)["W"]

    for seed in range(6):
        Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
        for r in F.captured_energy(M, Q.astype(np.float32), W_all, ks=(1, 2, 4, 8)):
            assert r["ratio"] <= 1.0 + 1e-12, r


# --------------------------------------------------------------------------- #
# the real draw, on the real handoff tables
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cell_line", C.CELL_LINES)
def test_real_pool_sizes(cell_line: str) -> None:
    """The two screened pools are the sizes the design record counted."""
    assert len(F.scored_perturbations(cell_line)) == POOL_SIZES[cell_line]["scored"]
    assert len(F.basis_pool(cell_line)) == POOL_SIZES[cell_line][F.POOL_PRIMARY]


@pytest.mark.parametrize("cell_line", C.CELL_LINES)
def test_pools_are_disjoint_by_construction(cell_line: str) -> None:
    """The stratum filter really does separate them, on both identifiers."""
    scored = F.scored_perturbations(cell_line)
    pool = F.basis_pool(cell_line)
    assert set(scored["gene_id"]) & set(pool["gene_id"]) == set()
    assert set(scored["target"]) & set(pool["target"]) == set()
    assert set(scored["effect_stratum"]) <= set(C.SCORED_STRATA)
    assert set(pool["effect_stratum"]) <= set(C.BASIS_STRATA)


@pytest.mark.parametrize("cell_line", C.CELL_LINES)
def test_basis_genes_contain_no_in_scope_perturbation(cell_line: str) -> None:
    """``0 of 64`` --- the check the whole design rests on, on the real tables."""
    genes = F.select_basis_genes(cell_line)
    assert len(genes) == C.N_BASIS_GENES

    leak = F.leakage_check(cell_line, genes)
    assert leak["n_overlap"] == 0, leak["detail"]
    assert leak["n_basis"] == C.N_BASIS_GENES
    assert leak["n_scored"] == POOL_SIZES[cell_line]["scored"]
    assert leak["detail"].startswith(f"0 of {C.N_BASIS_GENES}")


@pytest.mark.parametrize("cell_line", C.CELL_LINES)
def test_the_draw_is_reproducible_at_its_seed(cell_line: str) -> None:
    """Called twice, identical --- including the per-cell counts of the draw."""
    a = F.select_basis_genes(cell_line)
    b = F.select_basis_genes(cell_line)
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs["provenance"]["cell_counts"] == b.attrs["provenance"]["cell_counts"]
    assert list(a.columns) == list(F.POOL_COLUMNS)

    # and a different seed really does move it, or "seeded" would mean nothing
    other = F.select_basis_genes(cell_line, seed=C.BASIS_SEED + 1)
    assert list(other["gene_id"]) != list(a["gene_id"])


@pytest.mark.parametrize("cell_line", C.CELL_LINES)
def test_the_draw_comes_from_the_primary_pool_and_is_stratified(cell_line: str) -> None:
    """No fallback fires, and the 64 spread evenly over the 4x4 stratum grid.

    Evenness is the reason for stratifying at all: an unstratified draw from a
    pool that is itself skewed in expression would fit the basis on one tail of
    the range the scored perturbations span.
    """
    genes = F.select_basis_genes(cell_line)
    prov = genes.attrs["provenance"]

    assert set(genes["pool"]) == {F.POOL_PRIMARY}, "the fallback pool fired unexpectedly"
    assert prov["fallback_fired"] is False
    assert prov["n_from_pool"][F.POOL_PRIMARY] == C.N_BASIS_GENES
    assert prov["pool_sizes"][F.POOL_PRIMARY] == POOL_SIZES[cell_line][F.POOL_PRIMARY]
    assert prov["seed"] == C.BASIS_SEED
    assert prov["expression_column_used"] in F.EXPRESSION_COLUMNS

    counts = prov["cell_counts"]
    assert len(counts) == C.BASIS_N_EXPRESSION_BINS * C.BASIS_N_EFFECT_BINS
    assert sum(counts.values()) == C.N_BASIS_GENES
    assert max(counts.values()) - min(counts.values()) <= 1, counts

    # every drawn gene is a real read-out gene with a measured effect size
    assert genes["readout_col"].between(0, 10**6).all()
    assert genes["effect_rms_lfc"].notna().all()
    assert genes["gene_id"].is_unique


@pytest.mark.parametrize("cell_line", C.CELL_LINES)
def test_smoke_draw_is_a_smaller_leak_free_draw(cell_line: str) -> None:
    """The smoke path narrows the draw and nothing else."""
    genes = F.select_basis_genes(cell_line, smoke=True)
    assert len(genes) == C.SMOKE_N_BASIS_GENES
    assert F.leakage_check(cell_line, genes)["n_overlap"] == 0


def test_leakage_check_catches_a_planted_overlap() -> None:
    """A check that cannot fail is not a check.

    Both identifiers separately: a symbol collision and an id collision are
    different failures, and the two columns can disagree.
    """
    cell_line = "hepg2"
    genes = F.select_basis_genes(cell_line)
    scored = F.scored_perturbations(cell_line)

    by_id = genes.copy()
    by_id.loc[0, "gene_id"] = scored["gene_id"].iloc[0]
    caught = F.leakage_check(cell_line, by_id)
    assert caught["n_overlap"] == 1, caught["detail"]
    assert caught["overlap_gene_id"] == [scored["gene_id"].iloc[0]]

    by_target = genes.copy()
    by_target.loc[0, "target"] = scored["target"].iloc[0]
    caught = F.leakage_check(cell_line, by_target)
    assert caught["n_overlap"] == 1, caught["detail"]
    assert caught["overlap_target"] == [scored["target"].iloc[0]]


def test_dry_run_main_exits_zero_and_a_fit_without_dry_run_refuses() -> None:
    """The CLI does the selection and the leakage check, and nothing it cannot do.

    ``fit.py`` is torch-free and holds no raw delta-h by design; the streaming
    pass is ``extract/run.py``'s. A CLI that silently did nothing when asked to
    fit would look like a fit that produced no basis.
    """
    assert F.main(["--cell-line", "hepg2", "--dry-run"]) == 0
    with pytest.raises(SystemExit, match="extract.run"):
        F.main(["--cell-line", "hepg2"])


# --------------------------------------------------------------------------- #
# the on-disk format
# --------------------------------------------------------------------------- #
def test_basis_round_trips_through_disk_with_its_provenance(tmp_path, monkeypatch) -> None:
    """Every key the projector and the auditor need survives the write.

    The npz keys are a contract between this module and ``extract/run.py``, and
    a silently renamed key would surface as a ``KeyError`` in the middle of an
    eight-hour GPU pass. The provenance is a contract with the reader: the draw,
    the pool sizes, the seed, the leakage check and every reduction constant have
    to be recoverable from the file alone, without re-running the fit.
    """
    monkeypatch.setenv("SCFM_PFM6_DATA_ROOT", str(tmp_path))

    d, cell_line = 12, "hepg2"
    rng = np.random.default_rng(21)
    M = F.SecondMoment(d).update(rng.standard_normal((90, d))).matrix()
    got = F.eigendecompose(M, k=8)
    genes = F.select_basis_genes(cell_line)

    out = F.write_basis(
        "scgpt", cell_line, 12,
        W=got["W"], eigenvalues=got["eigenvalues"], explained=got["explained"],
        genes=genes, captured=F.captured_energy(M, got["W"], got["W"], ks=(1, 4)),
    )
    assert out.exists() and out.with_suffix(".json").exists()

    back = F.load_basis("scgpt", cell_line, 12)
    assert np.max(np.abs(back["W"] - got["W"])) == 0.0
    assert int(back["d"]) == d and int(back["k_store"]) == 8
    assert bool(back["centred"]) is C.BASIS_CENTRED
    assert str(back["model"]) == "scgpt" and str(back["cell_line"]) == cell_line
    assert list(back["basis_gene_id"]) == list(genes["gene_id"])
    assert set(back["basis_pool"]) == {F.POOL_PRIMARY}
    assert set(back["basis_stratum"]) <= set(C.BASIS_STRATA)

    prov = back["provenance"]
    assert prov["leakage_check"]["n_overlap"] == 0
    assert prov["draw"]["seed"] == C.BASIS_SEED
    assert prov["draw"]["pool_sizes"][F.POOL_PRIMARY] == POOL_SIZES[cell_line][F.POOL_PRIMARY]
    assert prov["config"]["BASIS_CENTRED"] is C.BASIS_CENTRED
    assert prov["config"]["K_LADDER"] == list(C.K_LADDER)
    assert len(prov["basis_genes"]) == C.N_BASIS_GENES
    assert all(r["diagnostic_only"] for r in prov["captured_energy"])


def test_write_basis_refuses_an_unrecorded_draw(tmp_path, monkeypatch) -> None:
    """A basis whose gene selection is unrecorded cannot be audited, so it is not written."""
    monkeypatch.setenv("SCFM_PFM6_DATA_ROOT", str(tmp_path))
    d = 6
    rng = np.random.default_rng(22)
    M = F.SecondMoment(d).update(rng.standard_normal((30, d))).matrix()
    got = F.eigendecompose(M, k=3)
    stripped = F.select_basis_genes("hepg2").copy()
    stripped.attrs.clear()

    with pytest.raises(RuntimeError, match="no draw record"):
        F.write_basis(
            "scgpt", "hepg2", 12,
            W=got["W"], eigenvalues=got["eigenvalues"], explained=got["explained"],
            genes=stripped,
        )


def test_load_basis_refuses_a_basis_with_no_provenance(tmp_path, monkeypatch) -> None:
    """The sibling json is required, not optional."""
    monkeypatch.setenv("SCFM_PFM6_DATA_ROOT", str(tmp_path))
    from scfm_hidarm import paths as P

    target = P.basis_file("scgpt", "hepg2", 3)
    P.ensure(target.parent)
    np.savez(target, W=np.eye(4, dtype=np.float32))
    with pytest.raises(RuntimeError, match="no sibling provenance"):
        F.load_basis("scgpt", "hepg2", 3)

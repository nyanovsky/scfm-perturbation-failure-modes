"""Each null destroys what it claims to destroy and preserves what it claims to.

A null that quietly changes more than it says makes the arm it is the null for
look better than it is, and nothing downstream can tell. So each property is
checked against arithmetic rather than against a second implementation:
``hidperm`` is a pure gather along axis 0 and therefore leaves every column's
multiset of values untouched; ``hidrand`` is a seeded draw and therefore
reproduces bit for bit; ``random_orthonormal`` returns a basis whose Gram matrix
is the identity.

⚠️ Means and standard deviations are compared after accumulating in float64 and
rounding back to float32. Summation is not associative in floating point, so a
permutation of the *same* multiset can move a float32 mean by an ulp --- the
multiset equality below is the exact statement, and the moment equality is
stated at the precision the block is stored in. That is a property fixed before
the test was run, not a tolerance widened until it passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from scfm_hidarm import config as C
from scfm_hidarm import install, nulls

LINE = "hepg2"
N_PERT, N_READOUT, K = 20, 50, 4
N_DOSE = len(C.VALUE_GRID_CP10K)
N_COLS = N_DOSE * K
KEY = C.model_stem("scgpt", "hid", 12, K)


def make_block(tmp_path, seed=C.SEED):
    """A real block whose slots have deliberately different scales.

    Different per slot because that is what ``hidrand`` has to match and what
    ``hidperm`` has to preserve: with every slot at the same scale, a null that
    got the per-slot scale wrong would look right.
    """
    rng = np.random.default_rng(seed)
    root = tmp_path / "bundle"
    base = root / LINE
    base.mkdir(parents=True, exist_ok=True)
    readout_gene_id = np.array([f"ENSG{i:011d}" for i in range(N_READOUT)], dtype="<U15")
    pert_gene_id = readout_gene_id[:N_PERT].astype("<U15")
    np.savez(
        base / "targets.npz",
        effect=rng.standard_normal((N_PERT, N_READOUT)).astype(np.float32),
        half0=rng.standard_normal((N_PERT, N_READOUT)).astype(np.float32),
        half1=rng.standard_normal((N_PERT, N_READOUT)).astype(np.float32),
        pert_target=np.array([f"T{i}" for i in range(N_PERT)], dtype="<U9"),
        pert_gene_id=pert_gene_id,
        pert_readout_col=np.arange(N_PERT, dtype=np.int64),
        pert_stratum=np.array(["S4_highest"] * N_PERT, dtype="<U10"),
        pert_knockdown=np.full(N_PERT, 0.8, dtype=np.float32),
        pert_n_de=np.full(N_PERT, 10, dtype=np.int64),
        readout_gene_id=readout_gene_id,
        readout_gene_name=np.array([f"G{i}" for i in range(N_READOUT)], dtype="<U10"),
        readout_pooled_cp10k=np.full(N_READOUT, 3.0, dtype=np.float32),
    )
    scale = np.array([0.1, 0.5, 1.0, 2.0, 8.0])[:, None] * np.ones((1, K))
    forms = {}
    for f in C.SLOT_FORMS:
        arr = rng.standard_normal((N_PERT, N_COLS, N_READOUT)).astype(np.float32)
        forms[f] = (arr * scale.reshape(1, N_COLS, 1)).astype(np.float32)
    install.write_block(
        root, LINE, KEY, forms["vvp"], forms["gdt"],
        np.arange(N_PERT, dtype=np.int64), pert_gene_id, readout_gene_id, K,
    )
    return root, install.block_path(root, LINE, KEY), forms


def moments(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-(column, gene) mean and sd, accumulated in float64, read at float32."""
    return (
        a.astype(np.float64).mean(axis=0).astype(np.float32),
        a.astype(np.float64).std(axis=0).astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# hidperm: the machinery null
# --------------------------------------------------------------------------- #
def test_hidperm_preserves_every_column_exactly(tmp_path):
    root, real, forms = make_block(tmp_path)
    out = install.block_path(root, LINE, nulls.null_key(KEY, "hidperm"))
    nulls.hidperm(real, out)
    z = np.load(out, allow_pickle=False)
    for f in C.SLOT_FORMS:
        got, want = z[f], forms[f]
        # the exact statement: a pure gather along axis 0 leaves the multiset
        assert np.array_equal(np.sort(got, axis=0), np.sort(want, axis=0))
        gm, gs = moments(got)
        wm, ws = moments(want)
        assert np.array_equal(gm, wm)
        assert np.array_equal(gs, ws)


def test_hidperm_destroys_the_pairing(tmp_path):
    """Most read-out genes get a genuinely different perturbation on each row.

    Fixed points are expected --- a real permutation has some --- so the claim
    is about the fraction, and it is read against the ``1/n`` a uniform
    permutation gives rather than against zero.
    """
    root, real, forms = make_block(tmp_path)
    out = install.block_path(root, LINE, nulls.null_key(KEY, "hidperm"))
    m = nulls.hidperm(real, out)
    z = np.load(out, allow_pickle=False)
    unchanged = np.array(
        [np.array_equal(z["vvp"][:, :, g], forms["vvp"][:, :, g]) for g in range(N_READOUT)]
    )
    assert unchanged.sum() == 0
    # per-row identity, not per-gene: the fraction of (pert, gene) cells whose
    # row did not move should sit near 1/n_pert, and far from 1
    assert m["n_fixed_points"] / m["n_pairs"] < 5.0 / N_PERT
    changed = (z["vvp"] != forms["vvp"]).any(axis=1).mean()
    assert changed > 0.8


def test_hidperm_uses_one_permutation_for_vvp_and_gdt(tmp_path):
    """Permuting the forms independently would split one feature vector across
    two perturbations, which destroys more than the pair association."""
    root, real, forms = make_block(tmp_path)
    out = install.block_path(root, LINE, nulls.null_key(KEY, "hidperm"))
    nulls.hidperm(real, out)
    z = np.load(out, allow_pickle=False)
    for g in range(N_READOUT):
        src = {tuple(forms["vvp"][p, :, g]): p for p in range(N_PERT)}
        for p in range(N_PERT):
            origin = src[tuple(z["vvp"][p, :, g])]
            assert np.array_equal(z["gdt"][p, :, g], forms["gdt"][origin, :, g])


def test_hidperm_is_reproducible_and_key_specific(tmp_path):
    root, real, _ = make_block(tmp_path)
    a = install.block_path(root, LINE, nulls.null_key(KEY, "hidperm"))
    nulls.hidperm(real, a)
    first = np.load(a, allow_pickle=False)["vvp"].copy()
    nulls.hidperm(real, a)
    assert np.array_equal(np.load(a, allow_pickle=False)["vvp"], first)
    # a different key must not reuse the same shuffle, or every layer of the
    # depth sweep would carry the identical permutation
    other = C.model_stem("scgpt", "hid", 4, K)
    assert nulls.derive_seed("hidperm", nulls.null_key(other, "hidperm")) != nulls.derive_seed(
        "hidperm", nulls.null_key(KEY, "hidperm")
    )


def test_the_null_block_still_reads_as_a_run6_block(tmp_path):
    """It keeps the labels, the coverage and the width of the arm it nulls."""
    root, real, _ = make_block(tmp_path)
    out = install.block_path(root, LINE, nulls.null_key(KEY, "hidperm"))
    nulls.hidperm(real, out)
    a, b = np.load(real, allow_pickle=False), np.load(out, allow_pickle=False)
    assert sorted(a.files) == sorted(b.files)
    assert C.DROPPED_RUN2_SLOT not in b.files
    for name in ("pert_index", "values", "value_keep", "dose_index", "pc_index",
                 "perturbed_gene_id", "readout_gene_id"):
        assert np.array_equal(a[name], b[name])
    assert b["vvp"].shape == a["vvp"].shape


def test_a_mis_named_output_is_refused(tmp_path):
    """Run 3 reads the key off the file name, so name and key may not drift."""
    root, real, _ = make_block(tmp_path)
    bad = real.parent / "wrong-name__mean_control.npz"
    with pytest.raises(RuntimeError, match="file name"):
        nulls.hidperm(real, bad, key=nulls.null_key(KEY, "hidperm"))


# --------------------------------------------------------------------------- #
# hidrand: width only
# --------------------------------------------------------------------------- #
def test_hidrand_is_bit_reproducible_at_its_seed(tmp_path):
    root, real, _ = make_block(tmp_path)
    out = install.block_path(root, LINE, nulls.null_key(KEY, "hidrand"))
    nulls.hidrand(real, out)
    first = {f: np.load(out, allow_pickle=False)[f].copy() for f in C.SLOT_FORMS}
    nulls.hidrand(real, out)
    z = np.load(out, allow_pickle=False)
    for f in C.SLOT_FORMS:
        assert np.array_equal(z[f], first[f])
    assert not np.array_equal(z["vvp"], z["gdt"])


def test_hidrand_matches_the_real_blocks_per_slot_scale(tmp_path):
    """Within the sampling error of an RMS on this many draws, not to a tuned tol.

    Each slot holds ``n_pert * k * n_readout`` = 4,000 iid draws, so the RMS
    estimate has a relative standard error of ``1/sqrt(2N)`` ~ 1.1%. The 5%
    band below is that error with room, fixed before the test ran.
    """
    root, real, forms = make_block(tmp_path)
    out = install.block_path(root, LINE, nulls.null_key(KEY, "hidrand"))
    m = nulls.hidrand(real, out)
    z = np.load(out, allow_pickle=False)
    for f in C.SLOT_FORMS:
        want = nulls.per_slot_scale(forms[f], K)
        assert np.allclose(np.array(m["per_slot_rms"][f]), want)
        got = nulls.per_slot_scale(z[f], K)
        assert np.allclose(got, want, rtol=0.05), f"{f}: {got} vs {want}"
    assert m["controls_for"] == "width only"
    assert "width only" in m["note"]


def test_hidrand_is_width_matched_by_construction(tmp_path):
    root, real, _ = make_block(tmp_path)
    out = install.block_path(root, LINE, nulls.null_key(KEY, "hidrand"))
    nulls.hidrand(real, out)
    a, b = np.load(real, allow_pickle=False), np.load(out, allow_pickle=False)
    assert b["vvp"].shape == a["vvp"].shape and b["gdt"].shape == a["gdt"].shape
    assert b["vvp"].shape[1] + b["gdt"].shape[1] == 10 * K


def test_per_slot_scale_refuses_a_partial_slot():
    with pytest.raises(RuntimeError, match="whole number"):
        nulls.per_slot_scale(np.zeros((2, 7, 3), dtype=np.float32), K)


# --------------------------------------------------------------------------- #
# the random basis hidrp is read through
# --------------------------------------------------------------------------- #
def test_random_orthonormal_is_orthonormal():
    for d, k in ((512, 64), (256, 16), (64, 64)):
        w = nulls.random_orthonormal(d, k, C.NULL_SEEDS["hidrp"])
        assert w.shape == (d, k)
        assert np.abs(w.astype(np.float64).T @ w.astype(np.float64) - np.eye(k)).max() < 1e-5


def test_random_orthonormal_is_a_function_of_its_seed_alone():
    a = nulls.random_orthonormal(64, 8, 11)
    assert np.array_equal(a, nulls.random_orthonormal(64, 8, 11))
    assert not np.array_equal(a, nulls.random_orthonormal(64, 8, 12))


def test_random_orthonormal_refuses_more_directions_than_dimensions():
    with pytest.raises(ValueError, match="orthonormal directions"):
        nulls.random_orthonormal(8, 16, 0)


# --------------------------------------------------------------------------- #
# stage 0 uses the identical machinery
# --------------------------------------------------------------------------- #
def test_normperm_is_the_same_operation_on_the_stage0_block(tmp_path):
    root, _, _ = make_block(tmp_path)
    rng = np.random.default_rng(C.SEED + 1)
    key = C.model_stem("scgpt", "norm", 12)
    t = np.load(root / LINE / "targets.npz", allow_pickle=False)
    norm = rng.random((N_PERT, N_DOSE, N_READOUT)).astype(np.float32)
    install.install_from_slots(
        root, LINE, key, {("vvp", d): norm[:, d, :][:, None, :] for d in range(N_DOSE)},
        np.arange(N_PERT, dtype=np.int64), t["pert_gene_id"].astype(str),
        t["readout_gene_id"].astype(str), 1,
    )
    real = install.block_path(root, LINE, key)
    out = install.block_path(root, LINE, nulls.null_key(key, "normperm"))
    nulls.hidperm(real, out)
    z = np.load(out, allow_pickle=False)
    assert z["vvp"].shape == (N_PERT, N_DOSE, N_READOUT)
    assert "gdt" not in z.files
    assert np.array_equal(
        np.sort(z["vvp"], axis=0), np.sort(norm, axis=0)
    )

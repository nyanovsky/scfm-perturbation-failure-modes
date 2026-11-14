"""A run-6 block round-trips through run 3's REAL ``LineBundle``.

This is the test the integration rests on. Run 6's claim is that it scores a new
*input* through run 3's unchanged rule, and the only thing standing between the
two is the on-disk block format: a synthetic run-2-shaped block of a
pseudo-model, read by run 3's unmodified ``resp`` arm. Every way that can go
wrong produces a plausible score rather than an error --- a mis-parsed key makes
the arm vanish from the scored set, an uncovered row makes ``run_cell`` skip the
model, a mis-ordered axis scrambles the dose axis inside the feature vector ---
so the assertions here are against run 3's actual imported code, never a
restatement of run 6's.

The bundle is synthetic and tiny, written in the real on-disk layout with the
dtypes copied from the production bundle at
``/biodata/nyanovsky/projects/scfm-pert-fail-modes/run4/bundle/hepg2/targets.npz``,
so the test runs anywhere and still exercises the reader it will meet in
production.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scfm_hidarm import config as C
from scfm_hidarm import install, sibling

torch = pytest.importorskip("torch")

LINE = "hepg2"
N_PERT, N_READOUT, K = 4, 6, 2
N_DOSE = len(C.VALUE_GRID_CP10K)
N_COLS = N_DOSE * K
KEY = C.model_stem("scgpt", "hid", 12, K)


def plant(form: str) -> np.ndarray:
    """``value = form_offset + 1000*pert + 10*column + gene``, exact in float32.

    Every entry names its own position, so a gather that returns the right
    *shape* from the wrong columns fails rather than passes.
    """
    off = 100000 * C.SLOT_FORMS.index(form)
    p = np.arange(N_PERT)[:, None, None] * 1000
    c = np.arange(N_COLS)[None, :, None] * 10
    g = np.arange(N_READOUT)[None, None, :]
    return (off + p + c + g).astype(np.float32)


def write_bundle(root, rng):
    """A minimal run-3 bundle: ``targets.npz`` and ``reference.json``.

    Keys and dtypes are run 3's own (``ingest/build.py:390``). ``coexpr.npz`` and
    the gene tables are deliberately absent --- ``LineBundle`` treats both as
    optional, and leaving them out proves the response block is read on its own
    rather than riding on a reference arm being present.
    """
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
    (base / "reference.json").write_text(
        json.dumps(
            {
                "cell_line": LINE,
                "n_perturbations": N_PERT,
                "n_readout_genes": N_READOUT,
                "reliability": {"shrinkage_lambda": 0.9, "median_weight": 0.6},
                "residual_ceiling": {"median_half_split": 0.5, "median_spearman_brown": 0.66},
                "raw_effect_ceiling": {"median_half_split": 0.6},
                "shared_response": {"mean_effect_l2": 1.0},
            },
            indent=2,
        )
    )
    return pert_gene_id, readout_gene_id


@pytest.fixture()
def bundle_root(tmp_path):
    rng = np.random.default_rng(C.SEED)
    root = tmp_path / "bundle"
    pert_gene_id, readout_gene_id = write_bundle(root, rng)
    install.write_block(
        root, LINE, KEY,
        plant("vvp"), plant("gdt"),
        np.arange(N_PERT, dtype=np.int64), pert_gene_id, readout_gene_id, K,
    )
    return root


def load(root):
    data = sibling.run3("translate.data")
    return data.LineBundle(LINE, root=root, device="cpu")


def arm(root):
    arms = sibling.run3("translate.arms")
    r3c = sibling.run3("config")
    b = load(root)
    spec = r3c.ArmSpec(C.arm_name(KEY), "scfm", ("resp",), model=KEY, anchor=C.ANCHOR)
    return b, arms.ArmFeatures(spec, b)


# --------------------------------------------------------------------------- #
def test_the_block_is_read_as_a_response_block(bundle_root):
    b = load(bundle_root)
    assert (KEY, C.ANCHOR) in b.features
    assert b.models_present(C.ANCHOR) == [KEY]


def test_resp_dim_is_ten_k(bundle_root):
    """``vvp`` + ``gdt`` and no ``zero_out``: exactly ``10k``, from run 3's own code."""
    _, af = arm(bundle_root)
    assert af._resp_dim() == 10 * K == af.dim
    assert "zero_out" not in np.load(
        bundle_root / LINE / "features" / install.block_filename(KEY)
    ).files


def test_row_map_has_no_minus_one(bundle_root):
    """An uncovered row makes ``runner.run_cell`` silently skip the model."""
    b = load(bundle_root)
    e = b.features[(KEY, C.ANCHOR)]
    assert int(e["row_map_np"].min()) == 0
    rows = np.arange(b.n_pert)
    assert b.models_for_scope("full", C.ANCHOR, rows) == [KEY]
    assert np.array_equal(b.scope_rows("full", C.ANCHOR), rows)


def test_batch_returns_the_planted_numbers_in_the_declared_column_order(bundle_root):
    """``batch`` is ``[vvp columns ... , gdt columns ...]``, dose-major within each."""
    b, af = arm(bundle_root)
    rows = torch.as_tensor([0, 3, 1, 1], dtype=torch.long)
    cols = torch.as_tensor([5, 0, 2, 4], dtype=torch.long)
    got = af.batch(rows, cols).cpu().numpy()
    assert got.shape == (4, 10 * K)

    expect = np.empty((rows.numel(), 10 * K), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows.tolist(), cols.tolist())):
        expect[i, :N_COLS] = plant("vvp")[r, :, c]
        expect[i, N_COLS:] = plant("gdt")[r, :, c]
    assert np.array_equal(got, expect)

    # and the columns decode as (dose, pc): column dose_i*k+pc of the vvp half
    z = np.load(bundle_root / LINE / "features" / install.block_filename(KEY))
    for dose_i in range(N_DOSE):
        for pc in range(K):
            col = dose_i * K + pc
            assert z["dose_index"][col] == dose_i and z["pc_index"][col] == pc
            assert z["value_cp10k"][col] == C.VALUE_GRID_CP10K[dose_i]
            assert np.array_equal(got[:, col], expect[:, col])


def test_resolve_arms_emits_the_run6_resp_arm(bundle_root):
    arms = sibling.run3("translate.arms")
    b = load(bundle_root)
    names = [s.name for s in arms.resolve_arms(b, [KEY], C.ANCHOR)]
    assert C.arm_name(KEY) == f"{KEY}_resp"
    assert f"{KEY}_resp" in names


def test_the_stage0_norm_block_is_vvp_only_and_dim_five(tmp_path):
    """Run 5's magnitude array installs as a five-wide block with no ``gdt``.

    Dim 5 against run 3's ``resp`` dim 11 is **not** width-matched, which is why
    the stage-0 arm is read against the dim-5 null and never ``null_randpair``.
    """
    rng = np.random.default_rng(C.SEED)
    root = tmp_path / "bundle"
    pert_gene_id, readout_gene_id = write_bundle(root, rng)
    key = C.model_stem("scgpt", "norm", 12)
    norm = rng.random((N_PERT, N_DOSE, N_READOUT)).astype(np.float32)
    slots = {("vvp", d): norm[:, d, :][:, None, :] for d in range(N_DOSE)}
    install.install_from_slots(
        root, LINE, key, slots,
        np.arange(N_PERT, dtype=np.int64), pert_gene_id, readout_gene_id, 1,
    )
    arms, r3c = sibling.run3("translate.arms"), sibling.run3("config")
    b = load(root)
    af = arms.ArmFeatures(
        r3c.ArmSpec(C.arm_name(key), "scfm", ("resp",), model=key, anchor=C.ANCHOR), b
    )
    assert af._resp_dim() == N_DOSE == 5
    rows = torch.as_tensor([2, 0], dtype=torch.long)
    cols = torch.as_tensor([1, 5], dtype=torch.long)
    got = af.batch(rows, cols).cpu().numpy()
    assert np.array_equal(got, np.stack([norm[2, :, 1], norm[0, :, 5]]))


# --------------------------------------------------------------------------- #
# the four ways a block fails by scoring rather than by raising
# --------------------------------------------------------------------------- #
def test_a_key_containing_the_forbidden_separator_is_refused(tmp_path):
    rng = np.random.default_rng(C.SEED)
    root = tmp_path / "bundle"
    p, r = write_bundle(root, rng)
    with pytest.raises(RuntimeError, match="vanish"):
        install.write_block(root, LINE, "scgpt__hid-l12-k2", plant("vvp"), plant("gdt"),
                            np.arange(N_PERT, dtype=np.int64), p, r, K)


def test_partial_coverage_is_refused(tmp_path):
    rng = np.random.default_rng(C.SEED)
    root = tmp_path / "bundle"
    p, r = write_bundle(root, rng)
    with pytest.raises(RuntimeError, match="SILENTLY SKIPS"):
        install.write_block(root, LINE, KEY, plant("vvp")[:-1], plant("gdt")[:-1],
                            np.arange(N_PERT - 1, dtype=np.int64), p[:-1], r, K)


def test_reordered_gene_labels_are_refused(tmp_path):
    rng = np.random.default_rng(C.SEED)
    root = tmp_path / "bundle"
    p, r = write_bundle(root, rng)
    with pytest.raises(RuntimeError, match="row for row"):
        install.write_block(root, LINE, KEY, plant("vvp"), plant("gdt"),
                            np.arange(N_PERT, dtype=np.int64), p[::-1].copy(), r, K)


def test_a_wrong_width_is_refused(tmp_path):
    rng = np.random.default_rng(C.SEED)
    root = tmp_path / "bundle"
    p, r = write_bundle(root, rng)
    with pytest.raises(RuntimeError, match="axis 1"):
        install.write_block(root, LINE, KEY, plant("vvp")[:, :-1], plant("gdt")[:, :-1],
                            np.arange(N_PERT, dtype=np.int64), p, r, K)


def test_the_manifest_is_not_written_into_features(bundle_root):
    """A ``features/*.npz`` name with no ``__`` kills every arm for the line."""
    fdir = bundle_root / LINE / "features"
    assert [q.name for q in fdir.iterdir()] == [install.block_filename(KEY)]
    assert all("__" in q.stem for q in fdir.glob("*.npz"))
    m = bundle_root / LINE / install.MANIFEST_DIRNAME / f"{KEY}__{C.ANCHOR}.json"
    assert m.exists()
    assert json.loads(m.read_text())["resp_dim"] == 10 * K


# --------------------------------------------------------------------------- #
# the CLI path from an extracted block to an installed one
#
# `extract/run.py` writes the PACKED layout (n_pert, n_dose*k, n_readout) via
# `slots.pack_block`, and until 2026-09-01 `install.source_slots` accepted only
# the unpacked 4-D form or run 5's 3-D norms --- so there was no command-line
# path from a run-6 extraction to an installed block at all, and it was the one
# thing standing between the extracted blocks and a score. It failed loudly, so
# nothing on disk was ever wrong. These tests pin the closure, and in particular
# that the packed layout is decoded from its own labels: shape cannot do it,
# because a packed k=1 block and a stage-0 `norm` block are the same shape.
# --------------------------------------------------------------------------- #
def packed_npz(tmp_path, p, r, k=K, gdt=True):
    """The extractor's own on-disk block, written the way ``extract_line`` writes it."""
    from scfm_hidarm.extract import slots as slotmod

    rng = np.random.default_rng(7)
    n_v = N_DOSE
    arrays = {
        form: rng.standard_normal((N_PERT, n_v, N_READOUT, k)).astype(np.float32)
        for form in (("vvp", "gdt") if gdt else ("vvp",))
    }
    f = tmp_path / "packed.npz"
    np.savez(
        f,
        pert_index=np.arange(N_PERT, dtype=np.int64),
        perturbed_gene_id=p,
        readout_gene_id=r,
        value_cp10k=np.repeat(np.asarray(C.VALUE_GRID_CP10K, dtype=np.float64), k),
        dose_index=np.repeat(np.arange(n_v, dtype=np.int64), k),
        pc_index=np.tile(np.arange(k, dtype=np.int64), n_v),
        **{form: slotmod.pack_block(a, k) for form, a in arrays.items()},
    )
    return f, arrays


def test_unpack_block_inverts_pack_block():
    from scfm_hidarm.extract import slots as slotmod

    rng = np.random.default_rng(3)
    a = rng.standard_normal((N_PERT, N_DOSE, N_READOUT, K)).astype(np.float32)
    packed = slotmod.pack_block(a, K)
    back = install.unpack_block(
        packed,
        np.repeat(np.arange(N_DOSE), K),
        np.tile(np.arange(K), N_DOSE),
    )
    # (n_pert, n_dose, k, n_readout) against pack_block's (n_pert, n_dose, n_readout, k)
    assert back.shape == (N_PERT, N_DOSE, K, N_READOUT)
    assert float(np.max(np.abs(back - a.transpose(0, 1, 3, 2)))) == 0.0


def test_unpack_block_refuses_a_permuted_axis():
    """Run 3 gathers axis 1 whole, so a mis-ordered block scores with a scrambled dose axis."""
    packed = np.zeros((N_PERT, N_DOSE * K, N_READOUT), dtype=np.float32)
    dose = np.tile(np.arange(N_DOSE), K)          # (pc, dose) order, not (dose, pc)
    pc = np.repeat(np.arange(K), N_DOSE)
    with pytest.raises(RuntimeError, match="declared"):
        install.unpack_block(packed, dose, pc)


def test_unpack_block_refuses_a_partially_labelled_axis():
    packed = np.zeros((N_PERT, N_DOSE * K, N_READOUT), dtype=np.float32)
    with pytest.raises(RuntimeError, match="wide but dose_index"):
        install.unpack_block(packed, np.arange(N_DOSE), np.arange(N_DOSE))


def test_cli_installs_the_extractors_packed_block(tmp_path):
    """The gap that blocked every run-6 score, closed and asserted end to end."""
    rng = np.random.default_rng(C.SEED)
    root = tmp_path / "bundle"
    p, r = write_bundle(root, rng)
    src, arrays = packed_npz(tmp_path, p, r)

    rc = install.main([
        "--source", str(src), "--bundle", str(root), "--line", LINE,
        "--model", "scgpt", "--variant", "hid", "--layer", "12", "--k", str(K),
    ])
    assert rc == 0
    z = np.load(install.block_path(root, LINE, KEY), allow_pickle=False)
    for form in ("vvp", "gdt"):
        want = arrays[form].transpose(0, 1, 3, 2).reshape(N_PERT, N_DOSE * K, N_READOUT)
        assert float(np.max(np.abs(z[form] - want))) == 0.0
    assert np.array_equal(z["dose_index"], np.repeat(np.arange(N_DOSE), K))


def test_cli_without_k_still_names_k_in_the_key(tmp_path):
    """A k-less key would make every later join name a block that is not this one."""
    rng = np.random.default_rng(C.SEED)
    root = tmp_path / "bundle"
    p, r = write_bundle(root, rng)
    src, _ = packed_npz(tmp_path, p, r)
    assert install.main([
        "--source", str(src), "--bundle", str(root), "--line", LINE,
        "--model", "scgpt", "--variant", "hid", "--layer", "12",
    ]) == 0
    assert install.block_path(root, LINE, KEY).exists()


def test_cli_k_prefix_is_a_column_prefix_of_the_stored_block(tmp_path):
    """The ladder is a prefix of what is stored and can never be extended upward."""
    rng = np.random.default_rng(C.SEED)
    root = tmp_path / "bundle"
    p, r = write_bundle(root, rng)
    src, arrays = packed_npz(tmp_path, p, r)
    assert install.main([
        "--source", str(src), "--bundle", str(root), "--line", LINE,
        "--model", "scgpt", "--variant", "hid", "--layer", "12", "--k", "1",
    ]) == 0
    key1 = C.model_stem("scgpt", "hid", 12, 1)
    z = np.load(install.block_path(root, LINE, key1), allow_pickle=False)
    assert z["vvp"].shape == (N_PERT, N_DOSE, N_READOUT)
    want = arrays["vvp"][:, :, :, 0]
    assert float(np.max(np.abs(z["vvp"] - want))) == 0.0
    with pytest.raises(RuntimeError, match="column PREFIX"):
        install.main([
            "--source", str(src), "--bundle", str(root), "--line", LINE,
            "--model", "scgpt", "--variant", "hid", "--layer", "12", "--k", str(K + 8),
        ])


def test_a_norm_block_is_not_read_as_a_packed_one(tmp_path):
    """Same shape, different meaning --- only the labels separate them."""
    rng = np.random.default_rng(C.SEED)
    root = tmp_path / "bundle"
    p, r = write_bundle(root, rng)
    norm = rng.standard_normal((N_PERT, N_DOSE, N_READOUT)).astype(np.float32)
    f = tmp_path / "norms.npz"
    np.savez(f, norm=norm)
    slots, k, layout = install.source_slots(np.load(f, allow_pickle=False))
    assert (k, layout) == (1, "norm_3d")
    assert len(slots) == N_DOSE

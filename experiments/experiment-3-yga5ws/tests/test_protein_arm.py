"""The protein arm's two joins: table into bundle order, arm into the sweep."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scfm_pfm import config as C
from scfm_pfm.protein.identity import linear_cka, rsa
from scfm_pfm.protein.install_arm import install
from scfm_pfm.translate import arms as armsmod


def _bundle(tmp_path, readout):
    for line in ("hepg2",):
        d = tmp_path / line
        d.mkdir(parents=True)
        np.savez(d / "targets.npz", readout_gene_id=np.asarray(readout, dtype=np.str_))
    return tmp_path


def _table(path, genes, dim, object_dtype: bool):
    ids = (
        np.asarray(genes, dtype=object) if object_dtype
        else np.asarray(genes, dtype=np.str_)
    )
    rng = np.random.default_rng(0)
    np.savez(
        path,
        embedding=rng.standard_normal((len(genes), dim)).astype(np.float32),
        mapped=np.ones(len(genes), dtype=bool),
        gene_id=ids,
    )


@pytest.mark.parametrize("object_dtype", [False, True])
def test_install_puts_rows_in_the_bundle_order(tmp_path, object_dtype):
    """The arm file must follow the bundle's read-out order, not the table's.

    A silent misalignment here would give the protein arm a real-looking score
    built from the wrong gene's vector, so the ordering is asserted rather than
    assumed. Both key dtypes are covered because a table written from pandas
    carries object-dtype keys that the safe npz loader refuses.
    """
    genes = ["ENSG3", "ENSG1", "ENSG2", "ENSG9"]
    readout = ["ENSG1", "ENSG2", "ENSG3"]
    _table(tmp_path / "esm2_test.npz", genes, 8, object_dtype)
    bundle = _bundle(tmp_path / "bundle", readout)
    report = install(tmp_path / "esm2_test.npz", bundle, ["hepg2"], "esm2_test")

    src = np.load(tmp_path / "esm2_test.npz", allow_pickle=True)["embedding"]
    got = np.load(bundle / "hepg2" / "emb__esm2_test.npz")["embedding"]
    assert got.shape == (3, 8)
    for i, gene in enumerate(readout):
        assert np.array_equal(got[i], src[genes.index(gene)])
    assert report["lines"]["hepg2"]["with_protein"] == 3


def test_missing_gene_becomes_a_zero_row(tmp_path):
    """A read-out gene the table does not carry is a zero row, and is counted."""
    _table(tmp_path / "esm2_test.npz", ["ENSG1"], 4, False)
    bundle = _bundle(tmp_path / "bundle", ["ENSG1", "ENSG2"])
    report = install(tmp_path / "esm2_test.npz", bundle, ["hepg2"], "esm2_test")
    got = np.load(bundle / "hepg2" / "emb__esm2_test.npz")["embedding"]
    assert np.array_equal(got[1], np.zeros(4, dtype=np.float32))
    assert report["lines"]["hepg2"]["zero_rows"] == 1
    assert report["lines"]["hepg2"]["with_protein"] == 1


class _FakeBundle:
    """Enough of a bundle for the arm resolver: a gene table and nothing else."""

    def __init__(self):
        self.emb = {
            "scgpt": torch.zeros(5, 4),
            "esm2_3b": torch.arange(20, dtype=torch.float32).reshape(5, 4),
        }
        self.features = {("scgpt", C.PRIMARY_ANCHOR): {}}
        self.coexpr = {}
        self.node = torch.zeros(5, 2)
        self.self_col = torch.arange(5)
        self.n_readout = 5


def test_embedding_only_model_becomes_an_arm():
    """A model with a gene table and no response features still resolves.

    Only the embedding arm: an arm that needs response features must not appear
    for a model that has none.
    """
    b = _FakeBundle()
    specs = armsmod.resolve_arms(b, ["scgpt"], C.PRIMARY_ANCHOR, False, ["esm2_3b"])
    names = {s.name for s in specs}
    assert "esm2_3b_emb" in names
    assert "esm2_3b_resp" not in names and "esm2_3b_full" not in names
    protein = next(s for s in specs if s.name == "esm2_3b_emb")
    assert protein.family == "protein"
    # the width-matched random null the protein arm is read against
    assert "null_randgene_d8" in names


def test_embedding_only_arm_gathers_both_genes():
    """The arm is the perturbed gene's vector then the read-out gene's."""
    b = _FakeBundle()
    spec = C.ArmSpec("esm2_3b_emb", "protein", ("emb",), model="esm2_3b",
                     anchor=C.PRIMARY_ANCHOR)
    arm = armsmod.ArmFeatures(spec, b)
    assert arm.dim == 8
    x = arm.batch(torch.tensor([1]), torch.tensor([3]))
    assert torch.equal(x[0, :4], b.emb["esm2_3b"][1])
    assert torch.equal(x[0, 4:], b.emb["esm2_3b"][3])


def test_cka_and_rsa_see_through_a_rotation():
    """Both similarity measures must be invariant to how a table is rotated."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 32))
    q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
    idx = np.arange(200)
    assert linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-9)
    assert rsa(x, x @ q, idx) == pytest.approx(1.0, abs=1e-9)
    assert linear_cka(x, rng.standard_normal((200, 32))) < 0.4


def _concat_spec() -> C.ArmSpec:
    return C.ArmSpec(
        "scgpt_esm2_3b_emb", "concat", ("emb", "emb2"),
        model="scgpt", anchor=None, second_model="esm2_3b",
    )


def test_the_two_table_arm_resolves_with_a_null_matched_to_its_width():
    """Wider than either part, so the matched random projection is required."""
    b = _FakeBundle()
    specs = armsmod.resolve_arms(b, ["scgpt"], C.PRIMARY_ANCHOR, False, ["esm2_3b"])
    names = {s.name for s in specs}
    assert "scgpt_esm2_3b_emb" in names
    # 4 + 4 per gene, two genes: the null has to be 16 wide, not 8
    assert "null_randgene_d16" in names
    spec = next(s for s in specs if s.name == "scgpt_esm2_3b_emb")
    assert (spec.family, spec.second_model) == ("concat", "esm2_3b")
    # no anchor: the arm reads two static tables, so response-feature coverage
    # must never decide whether it is scored
    assert spec.anchor is None


def test_the_two_table_arm_is_exactly_its_two_parts_side_by_side():
    """Concatenation, not a re-encoding: each part keeps its own columns."""
    b = _FakeBundle()
    rows, cols = torch.tensor([1]), torch.tensor([3])
    both = armsmod.ArmFeatures(_concat_spec(), b).batch(rows, cols)
    parts = [
        armsmod.ArmFeatures(
            C.ArmSpec(f"{m}_emb", "scfm", ("emb",), model=m, anchor=None), b
        ).batch(rows, cols)
        for m in ("scgpt", "esm2_3b")
    ]
    assert armsmod.ArmFeatures(_concat_spec(), b).dim == 16
    assert torch.equal(both, torch.cat(parts, dim=1))

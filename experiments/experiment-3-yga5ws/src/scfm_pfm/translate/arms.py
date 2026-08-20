"""What each arm feeds the translator, for a batch of gene pairs.

Every arm returns a (batch, dim) tensor for the same pairs; the translator,
its optimizer, its schedule and its epoch count never change. Feature
dimensionality does change between arms, which is exactly why a
matched-dimension random projection is a required null rather than a nicety.
"""

from __future__ import annotations

import torch

from .. import config as C
from .data import LineBundle


class ArmFeatures:
    """A resolved arm: knows its blocks, its dimension, and how to gather."""

    def __init__(self, spec: C.ArmSpec, bundle: LineBundle, coexpr_suffix: str = ""):
        self.spec = spec
        self.bundle = bundle
        self.suffix = coexpr_suffix
        self.model = spec.model
        self.anchor = spec.anchor
        self.entry = bundle.features[(spec.model, spec.anchor)] if spec.model else None
        self.parts: list[str] = list(spec.blocks)
        self.dim = self._dim()

    # ---- dimensions ------------------------------------------------------ #
    def _resp_dim(self) -> int:
        e = self.entry
        d = 0
        if "zero_out" in e:
            d += 1
        for form in ("vvp", "gdt"):
            if form in e:
                d += int(e[form].shape[1])
        return d

    def _dim(self) -> int:
        d = 0
        for part in self.parts:
            if part == "resp":
                d += self._resp_dim()
            elif part == "emb":
                d += 2 * int(self.bundle.emb[self.model].shape[1])
            elif part == "coexp":
                d += 4
            elif part == "node":
                d += 2 * int(self.bundle.node.shape[1])
            elif part == "randpair":
                d += 11
            elif part == "randgene":
                d += 2 * self.spec.rand_dim
            else:
                raise ValueError(f"unknown feature block {part!r}")
        return d

    # ---- gather ---------------------------------------------------------- #
    def batch(self, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """``rows`` index the cell line's in-scope perturbations, ``cols`` the
        read-out genes."""
        b = self.bundle
        out: list[torch.Tensor] = []
        for part in self.parts:
            if part == "resp":
                e = self.entry
                m = e["row_map"][rows]
                if "zero_out" in e:
                    out.append(e["zero_out"][m, cols].unsqueeze(1))
                for form in ("vvp", "gdt"):
                    if form in e:
                        out.append(e[form][m, :, cols])
            elif part == "emb":
                emb = b.emb[self.model]
                out.append(emb[b.self_col[rows]])
                out.append(emb[cols])
            elif part == "coexp":
                for key in ("pearson", "spearman", "mi_bits", "partial"):
                    out.append(b.coexpr[key + self.suffix][rows, cols].unsqueeze(1))
            elif part == "node":
                node = b.node if not self.suffix else b.node_check
                out.append(node[b.self_col[rows]])
                out.append(node[cols])
            elif part == "randpair":
                out.append(b.random_pair_tensor(11)[rows, cols])
            elif part == "randgene":
                rg = b.random_gene_vectors(self.spec.rand_dim)
                out.append(rg[b.self_col[rows]])
                out.append(rg[cols])
        return torch.cat(out, dim=1)


def resolve_arms(
    bundle: LineBundle, models: list[str], anchor: str, include_sensitivity: bool = False
) -> list[C.ArmSpec]:
    """The arm list for one (cell line, scope), given which models cover it."""
    specs: list[C.ArmSpec] = [
        C.ArmSpec("mean_only", "baseline", (), note="predict the training mean effect"),
        C.ArmSpec("coexp", "statistical", ("coexp",)),
        C.ArmSpec("node", "node_only", ("node",)),
        C.ArmSpec("coexp_node", "statistical", ("coexp", "node")),
        C.ArmSpec("null_randpair", "null", ("randpair",)),
    ]
    for width in sorted({int(bundle.emb[m].shape[1]) for m in models if m in bundle.emb}):
        specs.append(
            C.ArmSpec(
                f"null_randgene_d{2 * width}",
                "null",
                ("randgene",),
                rand_dim=width,
                note="fixed random vector per gene, matched to an embedding arm's width",
            )
        )
    for m in models:
        specs += [
            C.ArmSpec(f"{m}_resp", "scfm", ("resp",), model=m, anchor=anchor),
            C.ArmSpec(f"{m}_emb", "scfm", ("emb",), model=m, anchor=anchor),
            C.ArmSpec(f"{m}_full", "scfm", ("resp", "emb"), model=m, anchor=anchor),
            C.ArmSpec(
                f"{m}_hybrid", "hybrid", ("coexp", "node", "resp", "emb"), model=m, anchor=anchor
            ),
        ]
    if include_sensitivity:
        for m in models:
            for other in C.SENSITIVITY_ANCHORS:
                if (m, other) in bundle.features:
                    specs.append(
                        C.ArmSpec(
                            f"{m}_resp@{other}", "scfm_anchor", ("resp",), model=m, anchor=other
                        )
                    )
    return specs

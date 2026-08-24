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
        # ``.get``: an arm may name a model that has a gene embedding and no
        # response features at all -- the protein-model arm is exactly that.
        self.entry = bundle.features.get((spec.model, spec.anchor)) if spec.model else None
        self.parts: list[str] = list(spec.blocks)
        self.dim = self._dim()
        # An additive arm needs to know where the perturbed gene's half of the
        # feature vector ends. Every block an additive arm may use lays down the
        # perturbed gene's features and then the read-out gene's, in that order
        # and with equal width, so the split is the midpoint.
        self.split_at: int | None = None
        if spec.additive:
            per_gene = {"emb", "node", "randgene"}
            if not set(self.parts) <= per_gene:
                raise ValueError(f"additive arm {spec.name!r} needs per-gene blocks only")
            self.split_at = self.dim // 2

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
            elif part == "emb2":
                d += 2 * int(self.bundle.emb[self.spec.second_model].shape[1])
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
            elif part == "emb2":
                emb = b.emb[self.spec.second_model]
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
    bundle: LineBundle,
    models: list[str],
    anchor: str,
    include_sensitivity: bool = False,
    emb_only_models: tuple[str, ...] | list[str] = (),
) -> list[C.ArmSpec]:
    """The arm list for one (cell line, scope), given which models cover it.

    ``emb_only_models`` carry a static gene table and nothing else -- no query,
    no response features. The protein-language-model control is one of these:
    the same ``emb`` block every model embedding arm uses, filled with vectors
    that never saw a cell.
    """
    specs: list[C.ArmSpec] = [
        C.ArmSpec("mean_only", "baseline", (), note="predict the training mean effect"),
        C.ArmSpec("coexp", "statistical", ("coexp",)),
        C.ArmSpec(
            "node", "node_only", ("node",), additive=True,
            note="the two genes' separate control-cell properties, summed, no interaction",
        ),
        C.ArmSpec(
            "node_pairmlp", "node_only", ("node",),
            note="the same per-gene properties with pairwise interactions allowed",
        ),
    ]
    if "pearson_check" in bundle.coexpr:
        # The same features re-estimated on the OTHER half of the control
        # cells. The measured effects were computed against all controls, so if
        # co-expression's score depended on sharing cells with the target, the
        # two halves would not agree.
        specs += [
            C.ArmSpec(
                "coexp_check", "statistical", ("coexp",),
                note="co-expression from the held-out control half",
            ),
            C.ArmSpec(
                "node_check", "node_only", ("node",), additive=True,
                note="node statistics from the held-out control half",
            ),
        ]
    specs += [
        C.ArmSpec("coexp_node", "statistical", ("coexp", "node")),
        C.ArmSpec("null_randpair", "null", ("randpair",)),
    ]
    embedding_models = list(models) + [m for m in emb_only_models if m not in models]
    for width in sorted({int(bundle.emb[m].shape[1]) for m in embedding_models if m in bundle.emb}):
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
    for m in emb_only_models:
        if m in models or m not in bundle.emb:
            continue
        specs.append(
            C.ArmSpec(
                f"{m}_emb",
                "protein" if m in C.PROTEIN_MODELS else "scfm",
                ("emb",),
                model=m,
                anchor=anchor,
                note="static gene table only; this model has no response features here",
            )
        )
    specs += _concat_arm(bundle, {s.name for s in specs})
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


def _concat_arm(bundle: LineBundle, taken: set[str]) -> list[C.ArmSpec]:
    """The two-table arm and the random projection matched to its width.

    ``anchor=None`` on purpose: the arm reads two static gene tables and no
    response features, so it must not be filtered by whether the first model's
    query block covers this scope.
    """
    first, second = C.CONCAT_TABLES
    if first not in bundle.emb or second not in bundle.emb:
        return []
    width = int(bundle.emb[first].shape[1]) + int(bundle.emb[second].shape[1])
    out = [
        C.ArmSpec(
            f"{first}_{second}_emb", "concat", ("emb", "emb2"),
            model=first, anchor=None, second_model=second,
            note="both gene tables concatenated, identical translator",
        )
    ]
    null = f"null_randgene_d{2 * width}"
    if null not in taken:
        out.append(
            C.ArmSpec(
                null, "null", ("randgene",), rand_dim=width,
                note="fixed random vector per gene, matched to the concatenated arm's width",
            )
        )
    return out

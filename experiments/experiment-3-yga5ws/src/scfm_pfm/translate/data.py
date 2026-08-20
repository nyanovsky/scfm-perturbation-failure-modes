"""Load one cell line's bundle and hold every arm's inputs on the GPU.

Everything an arm can ask for is small once it is on the device: the largest
single array is a model's value-value profile at 705 x 5 x 3,247 float32
(46 MB). Batches are therefore built by gathering from resident tensors rather
than materializing a design matrix, which is what makes ~700 translator fits
affordable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .. import config as C
from .. import paths


def _rank_percentile_signed(x: np.ndarray) -> np.ndarray:
    """Sign times the within-query rank percentile of |x|; magnitude discarded.

    Run 2 measured TranscriptFormer's gradient magnitudes as unstable, so its
    GDT features enter as direction and rank only.
    """
    order = np.argsort(np.abs(x), axis=-1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    n = x.shape[-1]
    idx = np.arange(1, n + 1, dtype=np.float32) / n
    np.put_along_axis(ranks, order, np.broadcast_to(idx, x.shape), axis=-1)
    return (np.sign(x) * ranks).astype(np.float32)


class LineBundle:
    """One cell line's targets, control-cell statistics and model features."""

    def __init__(
        self,
        cell_line: str,
        root: Path | None = None,
        device: str = "cuda",
        targets_file: Path | None = None,
    ):
        self.cell_line = cell_line
        self.device = device
        base = paths.line_dir(cell_line, root)
        self.base = base
        t = np.load(base / "targets.npz", allow_pickle=False)
        self.target_source = "run1_all_controls"
        effect, half0, half1 = t["effect"], t["half0"], t["half1"]
        if targets_file is not None:
            # A target recomputed against one half of the control cells, so that
            # the co-expression features and the thing being predicted share no
            # cell. Row and column order have to be identical to the bundle's,
            # because every feature block is addressed by those indices.
            alt = np.load(targets_file, allow_pickle=False)
            if not np.array_equal(alt["pert_target"].astype(str), t["pert_target"].astype(str)):
                raise RuntimeError(f"{targets_file}: perturbation order differs from the bundle")
            if not np.array_equal(
                alt["readout_gene_id"].astype(str), t["readout_gene_id"].astype(str)
            ):
                raise RuntimeError(f"{targets_file}: read-out gene order differs from the bundle")
            effect, half0, half1 = alt["effect"], alt["half0"], alt["half1"]
            self.target_source = Path(targets_file).stem
        self.effect = torch.as_tensor(effect, device=device)
        self.half0 = half0
        self.half1 = half1
        self.pert_target = t["pert_target"].astype(str)
        self.pert_gene_id = t["pert_gene_id"].astype(str)
        self.self_col = torch.as_tensor(t["pert_readout_col"].astype(np.int64), device=device)
        self.self_col_np = t["pert_readout_col"].astype(np.int64)
        self.pert_stratum = t["pert_stratum"].astype(str)
        self.readout_gene_id = t["readout_gene_id"].astype(str)
        self.readout_gene_name = t["readout_gene_name"].astype(str)
        self.n_pert, self.n_readout = self.effect.shape
        self.reference = json.loads((base / "reference.json").read_text())

        self.coexpr: dict[str, torch.Tensor] = {}
        self.node: torch.Tensor | None = None
        self.node_check: torch.Tensor | None = None
        self.node_names: list[str] = []
        cx = base / "coexpr.npz"
        if cx.exists():
            z = np.load(cx, allow_pickle=False)
            for key in ("pearson", "spearman", "mi_bits", "partial"):
                self.coexpr[key] = torch.as_tensor(z[key], device=device)
                if key + "_check" in z.files:
                    self.coexpr[key + "_check"] = torch.as_tensor(z[key + "_check"], device=device)
            self.node = torch.as_tensor(z["node"], device=device)
            if "node_check" in z.files:
                self.node_check = torch.as_tensor(z["node_check"], device=device)
            self.node_names = [str(s) for s in z["node_names"]]

        self.emb: dict[str, torch.Tensor] = {}
        for p in sorted(base.glob("emb__*.npz")):
            model = p.stem.split("__", 1)[1]
            self.emb[model] = torch.as_tensor(
                np.load(p, allow_pickle=False)["embedding"], device=device
            )

        self.features: dict[tuple[str, str], dict] = {}
        fdir = base / "features"
        for p in sorted(fdir.glob("*.npz")) if fdir.exists() else []:
            model, anchor = p.stem.split("__", 1)
            z = np.load(p, allow_pickle=False)
            keep = z["value_keep"].astype(bool)
            entry: dict = {
                "pert_index": z["pert_index"].astype(np.int64),
                "values": z["values"][keep],
                "value_keep": keep,
            }
            if "zero_out" in z.files and bool(keep[np.isclose(z["values"], 0.0)].all()):
                entry["zero_out"] = torch.as_tensor(z["zero_out"], device=device)
            for form in ("vvp", "gdt"):
                if form not in z.files:
                    continue
                arr = z[form][:, keep, :]
                if form == "gdt" and model in C.RANK_ONLY_GDT_MODELS:
                    arr = _rank_percentile_signed(arr)
                    entry["gdt_rank_only"] = True
                entry[form] = torch.as_tensor(np.ascontiguousarray(arr), device=device)
            # scope row -> this block's row, or -1
            row_map = np.full(self.n_pert, -1, dtype=np.int64)
            row_map[entry["pert_index"]] = np.arange(entry["pert_index"].size)
            entry["row_map"] = torch.as_tensor(row_map, device=device)
            entry["row_map_np"] = row_map
            self.features[(model, anchor)] = entry

        gen = torch.Generator(device="cpu").manual_seed(C.SEED)
        self.rand_gene: dict[int, torch.Tensor] = {}
        self.rand_pair: torch.Tensor | None = None
        self._gen = gen

    # ---- nulls ----------------------------------------------------------- #
    def random_gene_vectors(self, dim: int) -> torch.Tensor:
        """Fixed random vector per read-out gene: the matched-dimension null for
        an embedding arm."""
        if dim not in self.rand_gene:
            g = torch.Generator(device="cpu").manual_seed(C.SEED + dim)
            self.rand_gene[dim] = torch.randn(
                self.n_readout, dim, generator=g, dtype=torch.float32
            ).to(self.device)
        return self.rand_gene[dim]

    def random_pair_tensor(self, dim: int) -> torch.Tensor:
        """Fixed random vector per (perturbed gene, read-out gene) pair."""
        if self.rand_pair is None or self.rand_pair.shape[-1] != dim:
            g = torch.Generator(device="cpu").manual_seed(C.SEED + 7919 + dim)
            self.rand_pair = torch.randn(
                self.n_pert, self.n_readout, dim, generator=g, dtype=torch.float32
            ).to(self.device)
        return self.rand_pair

    # ---- scopes ---------------------------------------------------------- #
    def models_present(self, anchor: str) -> list[str]:
        return [m for (m, a) in sorted(self.features) if a == anchor]

    def scope_rows(self, scope: str, anchor: str) -> np.ndarray:
        """The perturbations every arm in this scope is scored on."""
        if scope == "full":
            return np.arange(self.n_pert)
        common = None
        for m in self.models_present(anchor):
            idx = set(self.features[(m, anchor)]["pert_index"].tolist())
            common = idx if common is None else (common & idx)
        return np.array(sorted(common or []), dtype=np.int64)

    def models_for_scope(self, scope: str, anchor: str, rows: np.ndarray) -> list[str]:
        """Models whose block covers every row of the scope."""
        out = []
        for m in self.models_present(anchor):
            covered = self.features[(m, anchor)]["row_map_np"][rows]
            if (covered >= 0).all():
                out.append(m)
        return out


    # ---- cross-cell-line views ------------------------------------------- #
    def restrict_columns(self, gene_ids: np.ndarray) -> "LineBundle":
        """A view of this bundle over an ordered subset of read-out genes.

        Used for the cross-cell-line direction, where the two lines' read-out
        sets differ and the translator has to see one shared column space. A
        perturbed gene whose own column falls outside the shared set keeps a
        ``self_col`` of -1 and is dropped from the transfer rows, because
        neither its embedding nor its self-exclusion is defined there.
        """
        pos = {g: i for i, g in enumerate(self.readout_gene_id)}
        idx = np.array([pos[g] for g in gene_ids], dtype=np.int64)
        tidx = torch.as_tensor(idx, device=self.device)
        view = object.__new__(LineBundle)
        view.cell_line = self.cell_line
        view.device = self.device
        view.base = self.base
        view.effect = self.effect[:, tidx]
        view.half0 = self.half0[:, idx]
        view.half1 = self.half1[:, idx]
        view.pert_target = self.pert_target
        view.pert_gene_id = self.pert_gene_id
        remap = {int(c): i for i, c in enumerate(idx)}
        new_self = np.array(
            [remap.get(int(c), -1) for c in self.self_col_np], dtype=np.int64
        )
        view.self_col_np = new_self
        view.self_col = torch.as_tensor(np.maximum(new_self, 0), device=self.device)
        view.pert_stratum = self.pert_stratum
        view.readout_gene_id = self.readout_gene_id[idx]
        view.readout_gene_name = self.readout_gene_name[idx]
        view.n_pert = self.n_pert
        view.n_readout = int(idx.size)
        view.reference = self.reference
        view.target_source = self.target_source
        view.coexpr = {k: v[:, tidx] for k, v in self.coexpr.items()}
        view.node = None if self.node is None else self.node[tidx]
        view.node_check = None if self.node_check is None else self.node_check[tidx]
        view.node_names = self.node_names
        view.emb = {k: v[tidx] for k, v in self.emb.items()}
        view.features = {}
        for key, e in self.features.items():
            entry = {
                "pert_index": e["pert_index"],
                "values": e["values"],
                "value_keep": e["value_keep"],
                "row_map": e["row_map"],
                "row_map_np": e["row_map_np"],
            }
            if "zero_out" in e:
                entry["zero_out"] = e["zero_out"][:, tidx]
            for form in ("vvp", "gdt"):
                if form in e:
                    entry[form] = e[form][:, :, tidx]
            view.features[key] = entry
        view.rand_gene = {}
        view.rand_pair = None
        return view

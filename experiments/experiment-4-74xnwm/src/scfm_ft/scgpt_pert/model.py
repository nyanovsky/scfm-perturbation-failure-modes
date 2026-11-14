"""Build scGPT's perturbation model with a substitutable gene table.

The model class, its decoder and its forward pass are scGPT's own
``TransformerGenerator`` (scgpt 0.2.4), imported by file path so that
``scgpt/__init__`` -- which pulls scvi-tools, scanpy and torchtext -- never
executes. Run 2 used the same import route for ``TransformerModel``, and the
same two attention substitutions apply here and are recorded the same way: the
published checkpoint's fused FlashMHA projections are renamed onto torch's
packed projection, and the fallback ``torch.nn.TransformerEncoder`` path runs
the attention.

Only one thing is replaced: ``GeneEncoder``'s ``nn.Embedding`` becomes a lookup
into an explicit table plus an optional learned projection into ``d_model``.
With the pretrained rows and no projection that is arithmetically the same
module; it is written this way so that every arm reads its gene table from one
place, saves it in one format, and differs from the others only in what fills
the table and whether it takes gradient.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..sibling import run2

SPECIAL_PAD = "<pad>"


# --------------------------------------------------------------------------- #
# importing scGPT's model code without its package __init__
# --------------------------------------------------------------------------- #
def _load_by_path(fullname: str, path: Path):
    spec = importlib.util.spec_from_file_location(fullname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_SCGPT_ROOT: Path | None = None


def import_module_by_path(fullname: str, relative: str):
    """One installed scGPT module, loaded from its file without the package init."""
    if fullname in sys.modules:
        return sys.modules[fullname]
    return _load_by_path(fullname, scgpt_root() / relative)


def scgpt_root() -> Path:
    """The installed package directory, resolved once.

    Resolved before the synthetic ``scgpt`` module is registered, because a
    stub module has no ``__spec__`` and ``find_spec`` refuses to look past it.
    """
    global _SCGPT_ROOT
    if _SCGPT_ROOT is None:
        spec = importlib.util.find_spec("scgpt")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("scgpt is not installed (pip install --no-deps scgpt==0.2.4)")
        _SCGPT_ROOT = Path(list(spec.submodule_search_locations)[0])
    return _SCGPT_ROOT


def import_generation_model():
    """``scgpt.model.generation_model`` with a stub package around it.

    ``generation_model`` needs three things from the package it lives in: the
    layer definitions in ``scgpt.model.model``, a module logger, and
    ``map_raw_id_to_vocab_id``. The first is loaded from its real file; the
    other two are supplied here, because the real ``scgpt.utils`` imports
    anndata, matplotlib and IPython to provide one array lookup. Nothing in
    scGPT's model source is patched.
    """
    root = scgpt_root()

    if "scgpt" not in sys.modules or not hasattr(sys.modules["scgpt"], "logger"):
        import logging

        pkg = types.ModuleType("scgpt")
        pkg.__path__ = [str(root)]  # type: ignore[attr-defined]
        pkg.logger = logging.getLogger("scgpt")
        sys.modules["scgpt"] = pkg

        utils = types.ModuleType("scgpt.utils")

        def map_raw_id_to_vocab_id(raw_ids, gene_ids):
            """scGPT's own mapping, restated: index the vocabulary id array."""
            if isinstance(raw_ids, torch.Tensor):
                out = torch.as_tensor(
                    np.asarray(gene_ids)[raw_ids.cpu().numpy()], device=raw_ids.device
                )
                return out.to(raw_ids.dtype)
            return np.asarray(gene_ids)[np.asarray(raw_ids)]

        utils.map_raw_id_to_vocab_id = map_raw_id_to_vocab_id
        sys.modules["scgpt.utils"] = utils

        mpkg = types.ModuleType("scgpt.model")
        mpkg.__path__ = [str(root / "model")]  # type: ignore[attr-defined]
        sys.modules["scgpt.model"] = mpkg

    if "scgpt.model.model" not in sys.modules:
        _load_by_path("scgpt.model.model", root / "model" / "model.py")
    if "scgpt.model.generation_model" not in sys.modules:
        _load_by_path("scgpt.model.generation_model", root / "model" / "generation_model.py")
    return sys.modules["scgpt.model.generation_model"]


def scgpt_version() -> str:
    try:
        from importlib.metadata import version

        return version("scgpt")
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
# the substitutable gene table
# --------------------------------------------------------------------------- #
class TableGeneEncoder(nn.Module):
    """Gene identity as an explicit table, optionally projected into d_model.

    ``table`` is (n_tokens, table_dim). With ``project=False`` the table is the
    model's embedding and ``table_dim`` must equal ``d_model``; with
    ``project=True`` a bias-free linear map is learned from ``table_dim`` into
    ``d_model``. ``enc_norm`` is scGPT's own LayerNorm, kept in both cases so
    the vector reaching the transformer is normalised exactly as upstream.
    """

    def __init__(
        self,
        table: torch.Tensor,
        d_model: int,
        trainable_table: bool,
        project: bool,
        pad_index: int,
    ):
        super().__init__()
        table = table.detach().clone().float()
        table[pad_index] = 0.0
        self.table = nn.Parameter(table, requires_grad=bool(trainable_table))
        self.table_dim = int(table.shape[1])
        self.pad_index = int(pad_index)
        if project:
            self.proj: nn.Module = nn.Linear(self.table_dim, d_model, bias=False)
        else:
            if self.table_dim != d_model:
                raise ValueError(
                    f"table width {self.table_dim} != d_model {d_model} and no projection asked for"
                )
            self.proj = nn.Identity()
        self.enc_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.table[x]
        e = self.proj(e)
        return self.enc_norm(e)

    @torch.no_grad()
    def resolved_table(self) -> np.ndarray:
        """The d_model-wide vector each token actually contributes, pre-norm.

        This is the gene table an embedding arm scores: what the model has for
        gene identity after the projection, before the value and perturbation
        embeddings are added.
        """
        return self.proj(self.table).detach().float().cpu().numpy()


# --------------------------------------------------------------------------- #
# assembling one arm's model
# --------------------------------------------------------------------------- #
def load_checkpoint(cache: Path) -> tuple[dict, dict, dict]:
    """The published checkpoint: state dict, build args, vocabulary."""
    ck = Path(cache)
    vocab = json.loads((ck / "vocab.json").read_text())
    args = json.loads((ck / "args.json").read_text())
    sd = torch.load(ck / "best_model.pt", map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    return sd, args, vocab


def build_arm_model(
    arm,
    cache: Path,
    gene_table: np.ndarray | None,
    scgpt_positions: np.ndarray,
    device: str = "cuda",
) -> tuple[nn.Module, dict]:
    """One arm's ``TransformerGenerator``, ready to fine-tune.

    ``scgpt_positions`` is each read-out gene's row in the published vocabulary
    embedding (run 2 resolved it and every later run has used it), so the
    "own table" arms start from exactly the vectors runs 2 and 3 scored.
    ``gene_table`` supplies the substituted table for the other arms, one row
    per read-out gene in read-out order.
    """
    gm = import_generation_model()
    sd, args, vocab = load_checkpoint(cache)
    remap_flash_qkv = run2("models.scgpt_model").remap_flash_qkv
    sd, n_remapped = remap_flash_qkv(sd)

    d_model = int(args["embsize"])
    n_genes = int(np.asarray(scgpt_positions).size)
    pad_index = n_genes
    ntoken = n_genes + 1
    local_vocab = {SPECIAL_PAD: pad_index}

    from .. import config as C

    R = C.RECIPE
    model = gm.TransformerGenerator(
        ntoken=ntoken,
        d_model=d_model,
        nhead=int(args["nheads"]),
        d_hid=int(args["d_hid"]),
        nlayers=int(args["nlayers"]),
        nlayers_cls=int(args.get("n_layers_cls", 3)),
        n_cls=1,
        vocab=local_vocab,
        dropout=R.dropout,
        pad_token=SPECIAL_PAD,
        pad_value=R.pad_value,
        pert_pad_id=R.pert_pad_id,
        use_fast_transformer=False,
        pre_norm=False,
    )

    # the published embedding rows for these genes, in read-out order
    pos = np.asarray(scgpt_positions, dtype=np.int64)
    full_emb = sd["encoder.embedding.weight"]
    if int(pos.max()) >= int(full_emb.shape[0]) or int(pos.min()) < 0:
        raise RuntimeError("a read-out gene's scGPT vocabulary position is outside the table")
    own_rows = full_emb[torch.as_tensor(pos)].clone()
    restricted = torch.zeros(ntoken, d_model, dtype=own_rows.dtype)
    restricted[:n_genes] = own_rows

    # the recipe transfers only these prefixes; the expression decoder is fresh
    prefixes = tuple(R.load_param_prefixes)
    pretrained = {k: v for k, v in sd.items() if k.startswith(prefixes)}
    pretrained["encoder.embedding.weight"] = restricted
    model_dict = model.state_dict()
    shape_ok, shape_bad = {}, {}
    for k, v in pretrained.items():
        if k in model_dict and tuple(model_dict[k].shape) == tuple(v.shape):
            shape_ok[k] = v
        else:
            shape_bad[k] = tuple(v.shape)
    model_dict.update(shape_ok)
    model.load_state_dict(model_dict)

    # swap in the arm's table; every arm goes through the same encoder class
    if arm.table == "scgpt":
        table = restricted.clone()
    else:
        if gene_table is None:
            raise ValueError(f"arm {arm.name!r} needs a substituted gene table")
        wide = torch.as_tensor(np.asarray(gene_table), dtype=torch.float32)
        if wide.shape[0] != n_genes:
            raise ValueError(f"substituted table has {wide.shape[0]} rows, expected {n_genes}")
        table = torch.zeros(ntoken, wide.shape[1], dtype=torch.float32)
        table[:n_genes] = wide
    encoder = TableGeneEncoder(
        table, d_model, trainable_table=arm.trainable_table, project=arm.project, pad_index=pad_index
    )
    with torch.no_grad():
        encoder.enc_norm.weight.copy_(model.encoder.enc_norm.weight)
        encoder.enc_norm.bias.copy_(model.encoder.enc_norm.bias)
    model.encoder = encoder
    model.to(device)

    report = {
        "scgpt_version": scgpt_version(),
        "checkpoint_keys": len(sd),
        "fused_qkv_tensors_remapped": int(n_remapped),
        "pretrained_tensors_loaded": len(shape_ok),
        "pretrained_tensors_shape_mismatch": shape_bad,
        "load_param_prefixes": list(prefixes),
        "ntoken": ntoken,
        "d_model": d_model,
        "n_layers": int(args["nlayers"]),
        "n_heads": int(args["nheads"]),
        "table_source": arm.table,
        "table_dim": int(table.shape[1]),
        "table_trainable": bool(arm.trainable_table),
        "table_projected": bool(arm.project),
        "trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "substitutions": [
            "flash-attention not installed (the published checkpoint was trained with "
            "scGPT's FlashMHA layers, whose API exists only in flash-attn 1.x, which "
            "cannot build against this torch); used scGPT's own fallback "
            "torch.nn.TransformerEncoder path (use_fast_transformer=False), which "
            "changes the attention kernel and therefore the exact numerics",
            f"remapped {int(n_remapped)} fused attention tensors from the checkpoint's "
            "FlashMHA layout (self_attn.Wqkv) onto torch's packed projection "
            "(self_attn.in_proj); both hold [query; key; value] over the same head "
            "partition",
            "imported scgpt.model.generation_model by file path to avoid executing "
            "scgpt/__init__ (scvi-tools / scanpy / torchtext), and supplied "
            "map_raw_id_to_vocab_id and a module logger as stubs; model source unmodified",
            "the gene embedding is restricted to the read-out gene set and read through "
            "an explicit table module rather than nn.Embedding, so that the same code "
            "path serves the substituted tables; with the published rows and no "
            "projection the two are the same function",
        ],
    }
    return model, report


@torch.no_grad()
def verify_table_encoder(model: nn.Module, n_tokens: int, device: str = "cuda") -> dict:
    """Check the table encoder against an nn.Embedding holding the same rows."""
    enc = model.encoder
    if not isinstance(enc, TableGeneEncoder) or not isinstance(enc.proj, nn.Identity):
        return {"applicable": False}
    ref = nn.Embedding(enc.table.shape[0], enc.table.shape[1]).to(device)
    ref.weight.copy_(enc.table)
    idx = torch.arange(min(n_tokens, enc.table.shape[0]), device=device).unsqueeze(0)
    a = enc(idx)
    b = enc.enc_norm(ref(idx))
    diff = float((a - b).abs().max())
    return {"applicable": True, "max_abs_difference": diff, "agrees": diff < 1e-6}

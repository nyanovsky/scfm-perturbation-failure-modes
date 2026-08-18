"""scPRINT (jkobject/scPRINT, medium-v1.5) response interface.

``scprint.model.model`` imports ``scprint.model.utils``, which pulls bionty /
lamindb / scanpy / scdataloader and the GRN and denoising task modules -- an
import chain that needs a configured lamindb instance and is irrelevant to a
forward pass. We install ``scprint`` with ``--no-deps`` and stub exactly those external modules
before importing the model files, so both ``scprint.model.model`` and
``scprint.model.utils`` run as published: ``utils.Attention`` and
``utils._init_weights`` are real code, and only the ontology, atlas and plotting
imports that a forward pass never reaches are replaced.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

from .base import (
    ResponseContract,
    ResponseModel,
    finite_difference_column,
    jacobian_cross_check,
)
from . import sources as S


def _stub(name: str, **attrs) -> types.ModuleType:
    """Register an empty module under ``name``.

    A real ``__spec__`` matters: lightning probes optional dependencies with
    ``importlib.util.find_spec``, which raises on a module whose spec is None.
    """
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__version__ = "0.0.0-stub"
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def import_scprint_model():
    """Import ``scprint.model.model`` with the heavy import chain stubbed out."""
    spec = importlib.util.find_spec("scprint")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("scprint is not installed (pip install --no-deps scprint==2.3.8)")
    root = Path(list(spec.submodule_search_locations)[0])

    pkg = types.ModuleType("scprint")
    pkg.__path__ = [str(root)]
    sys.modules["scprint"] = pkg
    mpkg = types.ModuleType("scprint.model")
    mpkg.__path__ = [str(root / "model")]
    sys.modules["scprint.model"] = mpkg

    # scprint/model/utils.py holds real inference code the model constructor calls
    # (utils.Attention, utils._init_weights), so it must be the genuine module. Only
    # its heavy *external* imports are stubbed: an ontology/atlas stack and plotting
    # that a single forward pass never touches.
    for name in ("bionty", "scanpy", "leidenalg", "louvain"):
        _stub(name)
    scdl = _stub("scdataloader")
    scdl.utils = _stub("scdataloader.utils", translate=lambda *a, **k: {})  # type: ignore[attr-defined]
    def _file_to_list(path: str, strconv=str) -> list:
        """scprint.utils.fileToList, without the package's import chain.

        ``WeightedMasker`` reads the transcription-factor list at class definition
        time, so this has to work during import even though masking is a
        training-time path.
        """
        f = Path(path)
        if not f.exists():
            return []
        return [strconv(line.strip()) for line in f.read_text().splitlines() if line.strip()]

    _stub("scprint.utils", fileToList=_file_to_list)
    tasks = _stub("scprint.tasks")
    tasks.__path__ = []  # type: ignore[attr-defined]
    for name in ("cell_emb", "denoise", "grn"):
        _stub(f"scprint.tasks.{name}")

    for name in ("decoders", "encoders", "fsq", "loss", "utils"):
        sub = importlib.util.spec_from_file_location(
            f"scprint.model.{name}", root / "model" / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(sub)
        sys.modules[f"scprint.model.{name}"] = mod
        sub.loader.exec_module(mod)  # type: ignore[union-attr]

    mspec = importlib.util.spec_from_file_location(
        "scprint.model.model", root / "model" / "model.py"
    )
    mod = importlib.util.module_from_spec(mspec)
    sys.modules["scprint.model.model"] = mod
    mspec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def use_torch_attention() -> str:
    """Route simpler_flash's attention through its own pure-torch implementation.

    ``FlashSelfAttention`` asserts fp16/bf16 and calls a fused CUDA kernel with no
    forward-mode derivative. ``simpler_flash`` ships ``SelfAttention``, the same
    attention written in plain torch, and the transformer picks between them at
    construction time. Sending the flash class's forward to the torch one keeps the
    published weights and the published mathematics in float32.
    """
    from simpler_flash import mha as sfmha

    original = sfmha.FlashSelfAttention.forward

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None,
                cu_seqlens_k=None, max_seqlen_k=None, bias=None, **kwargs):
        inner = getattr(self, "_torch_attention", None)
        if inner is None:
            inner = sfmha.SelfAttention(
                causal=self.causal, softmax_scale=self.softmax_scale, attention_dropout=0.0
            )
            self._torch_attention = inner
        return inner(qkv, causal=causal, bias=bias)

    sfmha.FlashSelfAttention.forward = forward
    return (
        "simpler_flash.FlashSelfAttention.forward delegated to the package's own "
        "pure-torch SelfAttention: the fused kernel asserts fp16/bf16 and has no "
        "forward-mode derivative, so the response and its Jacobian would not be "
        "computable in float32. Same weights, same attention, different kernel "
        f"(upstream forward kept at {original.__qualname__})"
    )


class ScPrintResponse(ResponseModel):
    name = "scprint"

    def load(self) -> None:
        mod = import_scprint_model()
        self.substitutions.append(use_torch_attention())
        ckpt_path = self.cache / "scprint" / "medium-v1.5.ckpt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = dict(ckpt.get("hyper_parameters", {}))
        self.hparams_recorded = {
            k: v for k, v in hp.items() if k != "genes" and isinstance(v, (int, float, str, bool, type(None)))
        }
        self.normalization = str(hp.get("normalization", "sum"))
        hp.pop("precpt_gene_emb", None)  # embedding weights come from the checkpoint
        # Lightning records its own bookkeeping in hyper_parameters; scPrint forwards
        # unknown keyword arguments to the transformer, so they have to go
        dropped = [k for k in hp if k.startswith("_")]
        for k in dropped:
            hp.pop(k)
        self.hparams_dropped = dropped
        model = mod.scPrint(**hp)
        sd = dict(ckpt["state_dict"])
        # The released checkpoint stores the gene embedding table under the older
        # name; scPRINT's own checkpoint loader renames it, and so must this one.
        # Without the rename load_state_dict quietly leaves a randomly initialised
        # embedding table in place and every response is noise.
        renamed = []
        for old, new in (
            ("gene_encoder.0.embedding.weight", "gene_encoder.0.embeddings.weight"),
            ("gene_encoder.embedding.weight", "gene_encoder.embeddings.weight"),
        ):
            if old in sd:
                sd[new] = sd.pop(old)
                renamed.append(f"{old} -> {new}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        self.load_report = {
            "n_missing_keys": len(missing),
            "n_unexpected_keys": len(unexpected),
            "missing_head": list(missing)[:12],
            "unexpected_head": list(unexpected)[:12],
        }
        self.load_report["renamed_checkpoint_keys"] = renamed
        # nothing the forward pass touches may be left at its random initialisation
        allowed_missing = {"gene_encoder.0.embeddings.weight"} if not renamed else set()
        critical = [str(k) for k in missing if str(k) not in allowed_missing]
        self.load_report["missing_critical_keys"] = critical
        if critical:
            raise RuntimeError(
                f"scPRINT weights missing after load; the forward pass would use "
                f"randomly initialised parameters: {critical[:8]}"
            )
        emb = model.gene_encoder.embeddings.weight
        self.load_report["gene_embedding_check"] = {
            "shape": list(emb.shape),
            "l2_norm": float(emb.detach().norm()),
            "loaded_from_checkpoint": bool(renamed),
        }
        self.substitutions.append(
            "stubbed scprint.model.utils / scprint.utils / scprint.tasks.* at import "
            "time to avoid bionty+lamindb+scdataloader (a configured lamindb instance "
            "is not available and is unnecessary for a forward pass); model source unmodified"
        )
        self.genes = [str(g) for g in hp.get("genes", [])]
        self.model = model.eval().to(self.device).to(torch.float32)

    # ---- input ------------------------------------------------------------
    def prepare_input(self, counts: np.ndarray, gene_ids: np.ndarray) -> dict:
        positions, values, seq_index = self.measured_sequence(counts, gene_ids)
        return {
            "gene_pos": positions,
            "counts": values,
            "universe_seq_index": seq_index,
            "n_input_genes": int(getattr(self, "_input_genes_used", 0)),
            "total_counts": float(values.sum()),
        }

    def _forward(self, gene_pos: torch.Tensor, expression: torch.Tensor) -> torch.Tensor:
        # req_depth is one raw total count per cell, shape (batch,); scPRINT applies
        # log2(1 + depth) itself
        depth = expression.sum().reshape(1)
        out = self.model.forward(
            gene_pos=gene_pos.unsqueeze(0),
            expression=expression.unsqueeze(0),
            req_depth=depth,
            get_gene_emb=False,
            do_mvc=False,
            do_class=False,
        )
        mean = out["mean"] if isinstance(out, dict) else out
        return mean.squeeze(0)

    def response(self, prepared: dict) -> np.ndarray:
        gp = torch.as_tensor(prepared["gene_pos"], device=self.device)
        expr = torch.as_tensor(prepared["counts"], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            out = self._forward(gp, expr).float().cpu().numpy()
        self._full_response = out
        return out[prepared["universe_seq_index"]]

    # ---- jacobian ---------------------------------------------------------
    def jacobian_column(self, prepared: dict, source_position: int):
        hit = np.flatnonzero(prepared["gene_pos"] == source_position)
        if hit.size == 0:
            return None, "unavailable", "source gene absent from this model's input sequence"
        seq_pos = int(hit[0])
        gp = torch.as_tensor(prepared["gene_pos"], device=self.device)
        idx = torch.as_tensor(prepared["universe_seq_index"], device=self.device)
        base = torch.as_tensor(prepared["counts"], device=self.device, dtype=torch.float32)

        def fn(e: torch.Tensor) -> torch.Tensor:
            return self._forward(gp, e)[idx]

        tangent = torch.zeros_like(base)
        tangent[seq_pos] = 1.0
        try:
            _, jvp = torch.func.jvp(fn, (base,), (tangent,))
            col = jvp.detach().float().cpu().numpy()
            method = "forward_mode_autodiff_jvp"
        except Exception as exc:  # noqa: BLE001
            self.substitutions.append(
                f"forward-mode AD unavailable ({type(exc).__name__}); used central finite differences"
            )

            def npfn(v: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    return fn(torch.as_tensor(v, device=self.device, dtype=torch.float32)).float().cpu().numpy()

            col, step = finite_difference_column(npfn, prepared["counts"].astype(np.float64), seq_pos)
            method = f"central_finite_difference(step={step:.4g})"

        def npfn2(v: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return fn(torch.as_tensor(v, device=self.device, dtype=torch.float32)).float().cpu().numpy()

        self.jacobian_check = jacobian_cross_check(
            col, method, npfn2, prepared["counts"].astype(np.float64), seq_pos
        )
        self._jacobian_method = method
        return col, method, None

    # ---- contract ---------------------------------------------------------
    def contract(self) -> ResponseContract:
        return ResponseContract(
            model="scprint",
            checkpoint=f"{S.SCPRINT['repo']}:{S.SCPRINT['files']['weights']}",
            checkpoint_revision=S.SCPRINT["revision"],
            checkpoint_md5=None,
            input_normalization=(
                f"raw counts per gene, normalized inside the model "
                f"(hyper_parameter normalization='{getattr(self, 'normalization', 'sum')}': "
                "expression / expression.sum() for 'sum', log2(1+expression) for 'log'), "
                "with a requested-read-depth token prepended (log2(1 + total counts))"
            ),
            output_head="decoders.ExprDecoder -> output['mean'], scaled by the depth multiplier",
            output_semantics=(
                "predicted share of the cell's counts for that gene: the expression "
                "decoder ends in a softmax across the supplied gene positions, "
                "conditioned on the requested read depth. Multiply by the total count "
                "to read it as an expected count; it is not a log fold change, and the "
                "values depend on which genes were supplied together"
            ),
            information_flow="bidirectional flash transformer over the supplied gene set plus cell-embedding tokens",
            ordering_dependent=False,
            response_symmetric=None,
            symmetry_evidence=(
                "gene tokens carry no positional encoding by default, so the response "
                "is permutation-equivariant in the input gene order; pairwise symmetry "
                "of the response was not measured here"
            ),
            differentiable_wrt_input_expression=True,
            jacobian_available=True,
            jacobian_method=getattr(self, "_jacobian_method", "forward_mode_autodiff_jvp"),
            jacobian_unavailable_reason=None,
            jacobian_check=getattr(self, "jacobian_check", {}),
            vocabulary_kind="ensembl",
            vocabulary_size=len(getattr(self, "genes", [])),
            context_length=len(getattr(self, "genes", [])),
            dtype="float32",
            substitutions=self.substitutions,
            notes=(
                "The gene vocabulary is the checkpoint's own hyper_parameters['genes'] "
                "list; gene identity enters through a frozen pretrained (ESM-derived) "
                "gene embedding indexed by position in that list. "
                f"Checkpoint hyper-parameters recorded: {getattr(self, 'hparams_recorded', {})}."
            ),
        )

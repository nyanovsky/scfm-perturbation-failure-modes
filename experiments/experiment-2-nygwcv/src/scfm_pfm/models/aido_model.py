"""AIDO.Cell / GB.Cell-100M response interface.

The two HuggingFace-style modeling files shipped in the ModelGenerator sdist are
used unmodified; only ``transformers`` and ``torch`` are needed, not the
``modelgenerator`` stack. The input transform is copied exactly from
``GenBioCellFoundation.forward`` (read-depth-aware scFoundation preprocessing):

    rd = max(log10(total_counts), 5) ; x = log1p(x / total * 1e4) ;
    x = concat([x, rd, rd]) ; clamp(x, max=20)

Attention: the shipped ``BertSelfFlashAttention`` requires flash-attention 2,
which supports only fp16/bf16 and exposes no forward-mode derivative. Its
non-CUDA fallback branch is *numerically wrong* -- it hands
``scaled_dot_product_attention`` tensors shaped ``(batch*seq, heads, head_dim)``,
so the softmax runs across heads instead of across the sequence. We therefore
override that one ``forward`` with a shape-correct SDPA implementation (same
mathematics, same weights, explicit 1/sqrt(head_dim) scale) rather than using
either branch, and record it as a substitution. When flash-attention 2 is
importable we additionally run the published kernel and record how far the two
response vectors differ.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import (
    ResponseContract,
    ResponseModel,
    finite_difference_column,
    jacobian_cross_check,
)
from . import sources as S

N_GENES = 19264


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def import_cellfoundation(src_dir: Path):
    """Import the two self-contained modeling files without modelgenerator."""
    pkg = types.ModuleType("cellfoundation_local")
    pkg.__path__ = [str(src_dir)]
    sys.modules["cellfoundation_local"] = pkg
    cfg = _load_module(
        "cellfoundation_local.configuration_cellfoundation",
        src_dir / "configuration_cellfoundation.py",
    )
    # modeling file uses `from .configuration_cellfoundation import ...`
    mdl = _load_module(
        "cellfoundation_local.modeling_cellfoundation",
        src_dir / "modeling_cellfoundation.py",
    )
    return cfg, mdl


def sdpa_self_attention_forward(self, hidden_states, attention_mask=None, *args, **kwargs):
    """Shape-correct scaled dot-product attention for BertSelfFlashAttention."""
    bsz, q_len, _ = hidden_states.size()
    q = self._reshape(self.query(hidden_states), -1, bsz).transpose(1, 2)
    k = self._reshape(self.key(hidden_states), -1, bsz).transpose(1, 2)
    v = self._reshape(self.value(hidden_states), -1, bsz).transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
        scale=1.0 / math.sqrt(q.shape[-1]),
    )
    return (out.transpose(1, 2).reshape(bsz, q_len, -1),)


class AidoCellResponse(ResponseModel):
    name = "aido_cell"

    def load(self) -> None:
        src = self.cache / "aido_cell" / "cellfoundation"
        cfg_mod, mdl_mod = import_cellfoundation(src)
        ck = self.cache / "aido_cell"
        config = cfg_mod.CellFoundationConfig.from_pretrained(ck / "config.json")
        self.config_dict = json.loads((ck / "config.json").read_text())
        self.flash_available = importlib.util.find_spec("flash_attn") is not None
        self._mdl_mod = mdl_mod

        self._upstream_forward = mdl_mod.BertSelfFlashAttention.forward
        mdl_mod.BertSelfFlashAttention.forward = sdpa_self_attention_forward
        self.substitutions.append(
            "replaced BertSelfFlashAttention.forward with a shape-correct SDPA "
            "implementation: the shipped non-flash fallback softmaxes across heads "
            "instead of across the sequence, and the flash-attention path is fp16/bf16 "
            "only with no forward-mode derivative. Weights and attention mathematics "
            "are unchanged; the kernel and dtype are not."
        )
        model = mdl_mod.CellFoundationForMaskedLM(config)
        from safetensors.torch import load_file

        sd = load_file(str(ck / "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        self.load_report = {
            "n_missing_keys": len(missing),
            "n_unexpected_keys": len(unexpected),
            "missing_head": list(missing)[:12],
            "unexpected_head": list(unexpected)[:12],
        }
        # cls.predictions.decoder.bias is the *same parameter object* as
        # cls.predictions.bias (HF ties them), so the checkpoint stores it once and
        # load_state_dict reports the alias as missing. Accept that only if the tie
        # actually holds in this build.
        head = model.cls.predictions
        tied = head.decoder.bias is head.bias
        allowed = {"cls.predictions.decoder.bias"} if tied else set()
        critical = [str(k) for k in missing if str(k).startswith("cls.") and str(k) not in allowed]
        self.load_report["decoder_bias_tied_to_prediction_bias"] = bool(tied)
        self.load_report["missing_critical_keys"] = critical
        if critical:
            raise RuntimeError(f"per-gene MLM head weights missing from checkpoint: {critical[:8]}")
        self.model = model.eval().to(self.device).to(torch.float32)

    # ---- input ------------------------------------------------------------
    def prepare_input(self, counts: np.ndarray, gene_ids: np.ndarray) -> dict:
        x = self.vocab_input_vector(counts, gene_ids, size=N_GENES)
        return {
            "counts": x,
            "universe_positions": self.universe_positions(),
            "n_input_genes": int(getattr(self, "_input_genes_used", 0)),
            "total_counts": float(x.sum()),
        }

    def _transform(self, counts: torch.Tensor) -> torch.Tensor:
        total = counts.sum(dim=1, keepdim=True)
        rd = torch.maximum(torch.log10(total), torch.tensor(5.0, device=counts.device))
        x = torch.log1p(counts / total * 1e4)
        x = torch.cat((x, rd, rd), dim=1)
        return torch.clamp(x, max=20.0)

    def _forward_counts(self, counts: torch.Tensor) -> torch.Tensor:
        x = self._transform(counts.unsqueeze(0))
        logits = self.model(input_ids=x).logits
        return logits[0, :-2, 0]

    def response(self, prepared: dict) -> np.ndarray:
        counts = torch.as_tensor(prepared["counts"], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            out = self._forward_counts(counts).float().cpu().numpy()
        self._full_response = out
        if self.flash_available:
            self._flash_check = self._published_kernel_check(prepared)
        return out[prepared["universe_positions"]]

    def _published_kernel_check(self, prepared: dict) -> dict:
        """Run the published flash-attention bf16 path and compare."""
        try:
            import copy

            mod = self._mdl_mod
            saved = mod.BertSelfFlashAttention.forward
            # restore upstream forward on a bf16 copy of the model
            mod.BertSelfFlashAttention.forward = self._upstream_forward
            model_bf16 = copy.deepcopy(self.model).to(torch.bfloat16)
            counts = torch.as_tensor(prepared["counts"], device=self.device, dtype=torch.float32)
            x = self._transform(counts.unsqueeze(0)).to(torch.bfloat16)
            with torch.no_grad():
                out = model_bf16(input_ids=x).logits[0, :-2, 0].float().cpu().numpy()
            mod.BertSelfFlashAttention.forward = saved
            up = prepared["universe_positions"]
            return {
                "ran": True,
                "dtype": "bfloat16",
                "pearson_vs_sdpa_fp32": float(
                    np.corrcoef(out[up], self._full_response[up])[0, 1]
                ),
                "median_abs_difference": float(
                    np.median(np.abs(out[up] - self._full_response[up]))
                ),
                "response_sd": float(np.std(self._full_response[up])),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ran": False, "error": f"{type(exc).__name__}: {exc}"}

    # ---- jacobian ---------------------------------------------------------
    def jacobian_column(self, prepared: dict, source_position: int):
        up = torch.as_tensor(prepared["universe_positions"], device=self.device)
        base = torch.as_tensor(prepared["counts"], device=self.device, dtype=torch.float32)

        def fn(c: torch.Tensor) -> torch.Tensor:
            return self._forward_counts(c)[up]

        tangent = torch.zeros_like(base)
        tangent[source_position] = 1.0
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

            col, step = finite_difference_column(npfn, prepared["counts"].astype(np.float64), source_position)
            method = f"central_finite_difference(step={step:.4g})"

        def npfn2(v: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return fn(torch.as_tensor(v, device=self.device, dtype=torch.float32)).float().cpu().numpy()

        self.jacobian_check = jacobian_cross_check(
            col, method, npfn2, prepared["counts"].astype(np.float64), source_position
        )
        self._jacobian_method = method
        return col, method, None

    # ---- contract ---------------------------------------------------------
    def contract(self) -> ResponseContract:
        return ResponseContract(
            model="aido_cell",
            checkpoint=f"{S.AIDO_CELL['repo']} (plan alias {S.AIDO_CELL['plan_alias']}, upstream rename)",
            checkpoint_revision=S.AIDO_CELL["revision"],
            checkpoint_md5=None,
            input_normalization=(
                "raw counts scattered onto the fixed 19,264-gene order (unmeasured "
                "genes = 0), then read-depth-aware scFoundation transform: "
                "rd = max(log10(total), 5); x = log1p(x / total * 1e4); "
                "x = concat([x, rd, rd]); clamp(x, max=20). Sequence length 19,266."
            ),
            output_head="CellFoundationForMaskedLM.cls.predictions -> Linear(hidden, 1)",
            output_semantics=(
                "one scalar per gene position (the model's masked-language-model "
                "regression target, i.e. predicted high-read-depth expression on the "
                "log1p CP10K scale); the last two positions are read-depth tokens and "
                "are dropped"
            ),
            information_flow="bidirectional encoder (18 layers) over the full fixed gene order",
            ordering_dependent=False,
            response_symmetric=None,
            symmetry_evidence=(
                "bidirectional attention plus absolute position embeddings over a fixed "
                "gene order; every gene sees every other gene, but pairwise symmetry was "
                "not measured here"
            ),
            differentiable_wrt_input_expression=True,
            jacobian_available=True,
            jacobian_method=getattr(self, "_jacobian_method", "forward_mode_autodiff_jvp"),
            jacobian_unavailable_reason=None,
            jacobian_check=getattr(self, "jacobian_check", {}),
            vocabulary_kind="symbol",
            vocabulary_size=N_GENES,
            context_length=N_GENES + 2,
            dtype="float32",
            substitutions=self.substitutions,
            notes=(
                "The gene order is a model constraint, not a choice: position i of the "
                "input is gene i of OS_scRNA_gene_index.19264.tsv. Expression enters "
                "through GeneEmbedding's auto-discretization (Linear(1, 100) -> softmax "
                "-> bin-embedding mixture), which is smooth in the input value, so no "
                "binning has to be bypassed. "
                f"flash-attention 2 importable in this environment: {getattr(self, 'flash_available', False)}; "
                f"published-kernel cross-check: {getattr(self, '_flash_check', {'ran': False})}."
            ),
        )

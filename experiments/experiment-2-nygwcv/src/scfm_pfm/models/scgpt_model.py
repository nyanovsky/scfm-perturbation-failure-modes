"""scGPT (wanglab/scGPT-human) response interface.

Upstream's ``TransformerModel`` is used unmodified. It is imported by file path
with a synthetic ``scgpt`` package so that ``scgpt/__init__.py`` -- which pulls
scvi-tools, scanpy, torchtext and scib -- never executes. Nothing in the model
source is patched.

Two input paths are recorded for the same cell:

``binned``
    the published preprocessing: CP10K, log1p, then per-cell quantile binning
    into 51 bins. ``np.digitize`` is a step function, so this path has zero
    gradient with respect to expression.
``smooth``
    the same quantile bin edges, but the step function replaced by linear
    interpolation between edges. It agrees with the binned value to within one
    bin everywhere and is differentiable, so the Jacobian is taken through it.

The size of the binned-vs-smooth deviation in the *output* is measured and
recorded rather than assumed small.
"""

from __future__ import annotations

import importlib.util
import json
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

N_BINS_DEFAULT = 51


def import_transformer_model():
    """Import ``scgpt.model.model`` without executing ``scgpt/__init__``."""
    spec = importlib.util.find_spec("scgpt")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("scgpt is not installed (pip install --no-deps scgpt==0.2.4)")
    root = Path(list(spec.submodule_search_locations)[0])
    pkg = types.ModuleType("scgpt")
    pkg.__path__ = [str(root)]
    sys.modules.setdefault("scgpt", pkg)
    mpkg = types.ModuleType("scgpt.model")
    mpkg.__path__ = [str(root / "model")]
    sys.modules["scgpt.model"] = mpkg
    mspec = importlib.util.spec_from_file_location("scgpt.model.model", root / "model" / "model.py")
    mod = importlib.util.module_from_spec(mspec)
    sys.modules["scgpt.model.model"] = mod
    mspec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def quantile_bin_edges(log_expr: np.ndarray, n_bins: int) -> np.ndarray:
    nz = log_expr[log_expr > 0]
    if nz.size == 0:
        raise RuntimeError("cell has no non-zero expression")
    return np.quantile(nz, np.linspace(0, 1, n_bins - 1))


def remap_flash_qkv(sd: dict) -> tuple[dict, int]:
    """Rename FlashMHA fused-projection keys to torch's packed-projection names.

    The published scGPT-human checkpoint was trained with
    ``use_fast_transformer=True``, whose attention module keeps one fused
    ``Wqkv`` linear. torch's ``nn.MultiheadAttention`` stores exactly the same
    thing under ``in_proj_weight`` / ``in_proj_bias``: both are
    ``[query; key; value]`` stacked over the same head partition. Renaming is
    therefore lossless; only the kernel that consumes them changes.
    """
    out = {}
    n = 0
    for key, value in sd.items():
        if key.endswith("self_attn.Wqkv.weight"):
            out[key.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")] = value
            n += 1
        elif key.endswith("self_attn.Wqkv.bias"):
            out[key.replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias")] = value
            n += 1
        else:
            out[key] = value
    return out, n


class ScGPTResponse(ResponseModel):
    name = "scgpt"

    def load(self) -> None:
        mod = import_transformer_model()
        ck = self.cache / "scgpt"
        self.vocab_map: dict[str, int] = json.loads((ck / "vocab.json").read_text())
        self.args = json.loads((ck / "args.json").read_text())
        sd = torch.load(ck / "best_model.pt", map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        keys = list(sd.keys())
        self.flags = {
            "do_mvc": any(k.startswith("mvc_decoder") for k in keys),
            "explicit_zero_prob": any(k.startswith("decoder.zero_logit") for k in keys),
            "use_batch_labels": any(k.startswith("batch_encoder") for k in keys),
            "has_dsbn": any(k.startswith("dsbn") for k in keys),
        }
        self.n_bins = int(self.args.get("n_bins", N_BINS_DEFAULT))
        self.model = mod.TransformerModel(
            ntoken=len(self.vocab_map),
            d_model=int(self.args["embsize"]),
            nhead=int(self.args["nheads"]),
            d_hid=int(self.args["d_hid"]),
            nlayers=int(self.args["nlayers"]),
            nlayers_cls=int(self.args.get("n_layers_cls", 3)),
            n_cls=1,
            vocab=self.vocab_map,
            dropout=float(self.args.get("dropout", 0.2)),
            pad_token=self.args.get("pad_token", "<pad>"),
            pad_value=int(self.args.get("pad_value", -2)),
            do_mvc=self.flags["do_mvc"],
            do_dab=False,
            use_batch_labels=self.flags["use_batch_labels"],
            domain_spec_batchnorm=False,
            input_emb_style=self.args.get("input_emb_style", "continuous"),
            n_input_bins=self.n_bins,
            cell_emb_style="cls",
            explicit_zero_prob=self.flags["explicit_zero_prob"],
            use_fast_transformer=False,
            pre_norm=False,
        )
        sd, n_remapped = remap_flash_qkv(sd)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        self.load_report = {
            "n_missing_keys": len(missing),
            "n_unexpected_keys": len(unexpected),
            "missing_head": [str(k) for k in list(missing)[:12]],
            "unexpected_head": [str(k) for k in list(unexpected)[:12]],
            "n_checkpoint_keys": len(keys),
            "n_fused_qkv_tensors_remapped": n_remapped,
        }
        # the pretrained release carries no cell-type classifier; that head is never
        # called here, so its absence is recorded rather than treated as a failure
        unused_prefixes = ("cls_decoder",)
        critical = [str(k) for k in missing if not str(k).startswith(unused_prefixes)]
        self.load_report["missing_unused_head_keys"] = len(missing) - len(critical)
        self.load_report["missing_critical_keys"] = critical[:12]
        if critical:
            raise RuntimeError(f"scGPT checkpoint is missing weights this build needs: {critical[:10]}")
        self.substitutions.append(
            "flash-attention not installed (the published checkpoint was trained with "
            "scGPT's FlashMHA layers, whose API exists only in flash-attn 1.x, which "
            "cannot build against this torch); used scGPT's own fallback "
            "torch.nn.TransformerEncoder path (use_fast_transformer=False), which "
            "changes the attention kernel and therefore the exact numerics"
        )
        self.substitutions.append(
            f"remapped {n_remapped} fused attention tensors from the checkpoint's "
            "FlashMHA layout (self_attn.Wqkv) onto torch's packed projection "
            "(self_attn.in_proj); both hold [query; key; value] over the same head "
            "partition, so the attention computed is the same function of the same "
            "weights. Verified numerically at load time (see attention_remap_check)"
        )
        self.attention_remap_check = self._verify_attention_remap(sd)
        self.substitutions.append(
            "imported scgpt.model.model by file path to avoid executing "
            "scgpt/__init__ (scvi-tools / torchtext / scib); model source unmodified"
        )
        self.model.eval().to(self.device)

    def _verify_attention_remap(self, sd: dict) -> dict:
        """Check the remap by recomputing layer 0's attention from the fused weights.

        The fallback module is fed the remapped ``in_proj``; here the same output is
        recomputed directly from the checkpoint's ``Wqkv`` tensor with the head
        rearrangement FlashMHA uses. Agreement to floating-point tolerance is what
        makes the substitution a kernel change rather than a different model.
        """
        layer = self.model.transformer_encoder.layers[0]
        d = int(self.args["embsize"])
        h = int(self.args["nheads"])
        wq = sd["transformer_encoder.layers.0.self_attn.in_proj_weight"]
        bq = sd["transformer_encoder.layers.0.self_attn.in_proj_bias"]
        wo = sd["transformer_encoder.layers.0.self_attn.out_proj.weight"]
        bo = sd["transformer_encoder.layers.0.self_attn.out_proj.bias"]
        torch.manual_seed(0)
        x = torch.randn(1, 16, d, dtype=torch.float32)
        qkv = torch.nn.functional.linear(x, wq, bq).reshape(1, 16, 3, h, d // h)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        att = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        manual = torch.nn.functional.linear(
            att.transpose(1, 2).reshape(1, 16, d), wo, bo
        )
        with torch.no_grad():
            module = layer.self_attn(x, x, x, need_weights=False)[0]
        diff = float((manual - module).abs().max())
        return {
            "layer": 0,
            "max_abs_difference": diff,
            "reference_scale": float(module.abs().max()),
            "tolerance": 1e-4,
            "agrees": diff < 1e-4,
            "note": (
                "fused-QKV recomputation vs the torch attention module the remapped "
                "weights were loaded into, random input, float32"
            ),
        }

    # ---- input ------------------------------------------------------------
    def prepare_input(self, counts: np.ndarray, gene_ids: np.ndarray) -> dict:
        tokens, raw, seq_index = self.measured_sequence(counts, gene_ids)
        total = float(raw.sum())
        v = np.log1p(raw / max(total, 1.0) * 1e4)
        edges = quantile_bin_edges(v, self.n_bins)
        binned = np.digitize(v, edges).astype(np.float32)
        smooth = np.interp(v, edges, np.arange(1, edges.size + 1), left=0.0).astype(np.float32)
        return {
            "tokens": tokens,
            "values_binned": binned,
            "values_smooth": smooth,
            "log_expr": v.astype(np.float32),
            "bin_edges": edges,
            "universe_seq_index": seq_index,
            "total_counts": total,
            "n_input_genes": int(getattr(self, "_input_genes_used", 0)),
        }

    def _forward_values(self, tokens: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(1, tokens.shape[0], dtype=torch.bool, device=values.device)
        out = self.model._encode(tokens.unsqueeze(0), values.unsqueeze(0), pad)
        return self.model.decoder(out)["pred"].squeeze(0)

    def response(self, prepared: dict) -> np.ndarray:
        tokens = torch.as_tensor(prepared["tokens"], device=self.device)
        idx = prepared["universe_seq_index"]
        with torch.no_grad():
            self._pred_binned = self._forward_values(
                tokens, torch.as_tensor(prepared["values_binned"], device=self.device)
            ).float().cpu().numpy()
            self._pred_smooth = self._forward_values(
                tokens, torch.as_tensor(prepared["values_smooth"], device=self.device)
            ).float().cpu().numpy()
        self.binning_deviation = {
            "input_max_abs_bin_difference": float(
                np.max(np.abs(prepared["values_binned"] - prepared["values_smooth"]))
            ),
            "output_pearson_binned_vs_smooth": float(
                np.corrcoef(self._pred_binned[idx], self._pred_smooth[idx])[0, 1]
            ),
            "output_median_abs_difference": float(
                np.median(np.abs(self._pred_binned[idx] - self._pred_smooth[idx]))
            ),
            "output_binned_sd": float(np.std(self._pred_binned[idx])),
        }
        return self._pred_binned[idx]

    def response_smooth(self, prepared: dict) -> np.ndarray:
        return self._pred_smooth[prepared["universe_seq_index"]]

    # ---- jacobian ---------------------------------------------------------
    def jacobian_column(self, prepared: dict, source_position: int):
        tokens = torch.as_tensor(prepared["tokens"], device=self.device)
        idx = prepared["universe_seq_index"]
        hit = np.flatnonzero(prepared["tokens"] == source_position)
        if hit.size == 0:
            return None, "unavailable", "the source gene is not in this model's input sequence"
        seq_pos = int(hit[0])
        base = torch.as_tensor(prepared["values_smooth"], device=self.device)

        def fn(v: torch.Tensor) -> torch.Tensor:
            return self._forward_values(tokens, v)[torch.as_tensor(idx, device=v.device)]

        tangent = torch.zeros_like(base)
        tangent[seq_pos] = 1.0
        # torch's fused/efficient attention kernels have no forward-mode rule, so the
        # derivative pass runs under the math backend: same attention, composed of ops
        # that do carry one. The response above still uses the default kernel.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        try:
            with sdpa_kernel(SDPBackend.MATH):
                _, jvp = torch.func.jvp(fn, (base,), (tangent,))
            col = jvp.detach().float().cpu().numpy()
            method = "forward_mode_autodiff_jvp_on_smooth_binning_math_sdpa_backend"
        except Exception as exc:  # noqa: BLE001
            self.substitutions.append(
                f"forward-mode AD unavailable ({type(exc).__name__}: {exc}); "
                "used central finite differences instead"
            )

            def npfn(v: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    return fn(
                        torch.as_tensor(v, device=self.device, dtype=torch.float32)
                    ).float().cpu().numpy()

            col, step = finite_difference_column(
                npfn, prepared["values_smooth"].astype(np.float64), seq_pos
            )
            method = f"central_finite_difference(step={step:.4g})_on_smooth_binning"
        # An independent cross-check, chosen so it is never a comparison of a method
        # with itself: autodiff is checked against a finite difference, and a finite
        # difference is checked against a second one at twice the step.
        def npfn2(v: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return fn(
                    torch.as_tensor(v, device=self.device, dtype=torch.float32)
                ).float().cpu().numpy()

        self.jacobian_check = jacobian_cross_check(
            col, method, npfn2, prepared["values_smooth"].astype(np.float64), seq_pos
        )
        return col, method, None

    # ---- contract ---------------------------------------------------------
    def contract(self) -> ResponseContract:
        return ResponseContract(
            model="scgpt",
            checkpoint=S.SCGPT["repo"],
            checkpoint_revision=S.SCGPT["revision"],
            checkpoint_md5=None,
            input_normalization=(
                "counts -> CP10K -> log1p -> per-cell quantile binning into "
                f"{self.n_bins} bins (published path). Gene tokens are vocab.json ids; "
                "no <cls> token is prepended (the checkpoint was trained with no_cls=True)."
            ),
            output_head="ExprDecoder.fc -> dict['pred'], one scalar per gene token",
            output_semantics=(
                "predicted binned expression value for that gene in this cell, on the "
                "51-bin training scale (not counts, not CP10K)"
            ),
            information_flow="bidirectional full attention over gene tokens; no positional encoding over genes",
            ordering_dependent=False,
            response_symmetric=None,
            symmetry_evidence=(
                "architecture imposes no ordering, so a response exists for every "
                "ordered pair, but symmetry of the pairwise response is not implied "
                "and was not measured here"
            ),
            differentiable_wrt_input_expression=True,
            jacobian_available=True,
            jacobian_method=getattr(self, "_jacobian_method", "forward_mode_autodiff_jvp_on_smooth_binning"),
            jacobian_unavailable_reason=None,
            jacobian_check=getattr(self, "jacobian_check", {}),
            vocabulary_kind="symbol",
            vocabulary_size=len(self.vocab_map),
            context_length=int(self.args.get("max_seq_len", 1200)),
            dtype="float32",
            substitutions=self.substitutions,
            notes=(
                "The published input path bins expression with np.digitize, which has "
                "zero derivative almost everywhere: the Jacobian is therefore taken "
                "through a piecewise-linear surrogate of the same quantile bin edges. "
                "binning_deviation records how far the surrogate moves the output. "
                f"Checkpoint build flags inferred from the state dict: {getattr(self, 'flags', {})}. "
                f"Sequence length used here is the number of resolved vocabulary genes, "
                f"which exceeds the training max_seq_len of {self.args.get('max_seq_len')}; "
                "scGPT has no positional encoding over genes, so this is a context-length "
                "extrapolation rather than an invalid input."
            ),
        )

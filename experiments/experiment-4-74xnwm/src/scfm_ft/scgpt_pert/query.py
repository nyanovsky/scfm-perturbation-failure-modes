"""Two ways to ask the fine-tuned model what one gene does to another.

``ScGPTPertQuery`` implements run 2's ``QueryModel`` interface, so run 2's
``extract_anchor`` drives it exactly as it drove the four frozen models: change
the perturbed gene's input *value*, read every read-out gene's predicted
expression, and assemble the same zero-out, value-value-profile and
gradient-trajectory forms. That is the query the frozen scGPT was read with in
runs 2 and 3, which is what makes "before fine-tuning" and "after fine-tuning"
the same measurement.

``pert_flag_response`` asks the other question, the one the fine-tuning
objective is defined on: leave the input cell alone, set the model's own
perturbation flag on the gene, and read the difference. A model fine-tuned to
predict perturbation responses should answer this one well if it answers any.

Unlike the pretrained scGPT of run 2, this model takes continuous expression
rather than quantile bins, so the derivative is taken through the real input
with no smoothing surrogate.
"""

from __future__ import annotations

import numpy as np
import torch

from ..sibling import run2


def _base():
    return run2("models.queries").QueryModel, run2("models.queries").QueryLayout


class ScGPTPertQuery(run2("models.queries").QueryModel):  # type: ignore[misc]
    """scGPT's perturbation model behind run 2's query interface."""

    name = "scgpt_ft"
    batch_size = 16
    self_response_defined = True
    ordering_dependent = False
    #: the model predicts log1p CP10K, so expm1 puts it back on a linear scale
    linear_scale = "expm1"

    def __init__(self, model, n_readout: int, device: str = "cuda", amp: bool = False):
        self.model = model
        self.n_readout = int(n_readout)
        self.device = device
        # Deliberately full precision, unlike the training step. A response is a
        # difference between two forward passes that agree to within about 1e-4
        # on outputs of order 1; in half precision that difference is below the
        # representable gap and the feature would be rounding noise. Measured on
        # this model before it was turned off: the exact derivative and the
        # finite difference of the same quantity agreed at Pearson 0.93 with
        # autocast on, and far closer with it off.
        self.amp = bool(amp)
        self.substitutions: list[str] = []
        self._src = torch.arange(self.n_readout, device=device).unsqueeze(0)
        self._zero_flags = torch.zeros(1, self.n_readout, dtype=torch.long, device=device)
        self.model.eval()

    # ---- run 2's interface ------------------------------------------------ #
    def configure(self, readout, anchor_cp10k, cell_line: str):
        _, QueryLayout = _base()
        if int(np.asarray(anchor_cp10k).size) != self.n_readout:
            raise ValueError("anchor profile width does not match the read-out set")
        return QueryLayout(
            model=self.name,
            cell_line=cell_line,
            n_readout=self.n_readout,
            n_supplied=self.n_readout,
            supplied_mode="readout",
            n_passes=1,
            context_block=self.n_readout,
            notes={
                "input_scale": "log1p(CP10K), the scale the perturbation recipe trains on",
                "perturbation_flags": "all zero for this query family: only the input value moves",
                "output_head": "AffineExprDecoder -> mlm_output, one predicted log1p CP10K per gene",
            },
        )

    def to_native(self, cp10k) -> np.ndarray:
        return np.log1p(np.asarray(cp10k, dtype=np.float64))

    def _forward(self, x: torch.Tensor, flags: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        f = self._zero_flags.expand(b, -1) if flags is None else flags
        pad = torch.zeros(b, self.n_readout, dtype=torch.bool, device=self.device)
        return self.model(
            self._src.expand(b, -1), x, f, src_key_padding_mask=pad
        )["mlm_output"]

    def forward_batch(self, x_native, perturbed_col=None, mask=None) -> np.ndarray:
        if mask is not None:
            raise NotImplementedError("this adapter has no masking regime")
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
        out = np.zeros((x.shape[0], self.n_readout), dtype=np.float32)
        with torch.no_grad():
            for start, chunk in self._batches(x):
                t = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)
                with torch.autocast("cuda", enabled=self.amp):
                    out[start : start + chunk.shape[0]] = self._forward(t).double().cpu().numpy()
        return out

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        """The vector gene identity contributes, after any learned projection.

        Run 2 read scGPT's gene table as ``encoder.embedding.weight`` before the
        encoder's LayerNorm; the same point in the graph is taken here, which
        for a substituted table is the projected row.
        """
        table = self.model.encoder.resolved_table()[: self.n_readout]
        return (
            np.asarray(table, dtype=np.float32),
            "TransformerGenerator.encoder: the gene table after its projection into d_model, "
            "before the encoder LayerNorm",
        )

    def exact_jvp(self, x_native: np.ndarray, col: int) -> np.ndarray | None:
        """Forward-mode derivative of every output with respect to gene ``col``."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        base = torch.as_tensor(
            np.asarray(x_native, dtype=np.float32).reshape(-1), device=self.device
        )
        tangent = torch.zeros_like(base)
        tangent[int(col)] = 1.0

        def fn(v: torch.Tensor) -> torch.Tensor:
            return self._forward(v.unsqueeze(0)).squeeze(0)

        try:
            # torch's fused attention kernels carry no forward-mode rule, so the
            # derivative pass runs under the math backend: the same attention,
            # composed of ops that do. Run 2 recorded the same substitution.
            with sdpa_kernel(SDPBackend.MATH):
                _, jvp = torch.func.jvp(fn, (base,), (tangent,))
            return jvp.detach().float().cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            self._jvp_error = f"{type(exc).__name__}: {exc}"
            return None


@torch.no_grad()
def pert_flag_response(
    q: ScGPTPertQuery, anchor_native: np.ndarray, cols: np.ndarray, batch: int = 16
) -> dict:
    """The model's own perturbation query, one response vector per perturbed gene.

    ``anchor_native`` is the anchor cell on the model's input scale; ``cols``
    are the perturbed genes' read-out columns. Returns the raw difference and
    the standardized form run 3 reads, plus the baseline for the record.
    """
    x = torch.as_tensor(
        np.asarray(anchor_native, dtype=np.float32).reshape(1, -1), device=q.device
    )
    with torch.autocast("cuda", enabled=q.amp):
        base = q._forward(x).double().cpu().numpy()[0]

    cols = np.asarray(cols, dtype=np.int64)
    out = np.zeros((cols.size, q.n_readout), dtype=np.float64)
    for start in range(0, cols.size, batch):
        chunk = cols[start : start + batch]
        b = chunk.size
        flags = torch.zeros(b, q.n_readout, dtype=torch.long, device=q.device)
        flags[torch.arange(b, device=q.device), torch.as_tensor(chunk, device=q.device)] = 1
        with torch.autocast("cuda", enabled=q.amp):
            pred = q._forward(x.expand(b, -1), flags).double().cpu().numpy()
        out[start : start + b] = pred - base[None, :]
    return {"delta": out, "baseline": base.astype(np.float32)}

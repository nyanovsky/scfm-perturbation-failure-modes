"""Perturbation discrimination: is a prediction closer to its own perturbation
than to the other held-out ones?

Two implementations, kept apart on purpose.

``cell_eval_discrimination`` calls Arc Institute's own ``cell-eval`` at a pinned
version on effect-scale profiles. That is the Virtual Cell Challenge metric as
its authors implemented it, and the version is recorded in the manifest.

``own_discrimination`` is the same definition written here, so it can also be
run on the residual vectors the translator actually emits. It is reported as
our variant, never as the challenge metric.
"""

from __future__ import annotations

import numpy as np


def own_discrimination(real, pred, metric: str = "l1", device: str | None = None) -> dict:
    """Normalized rank of the true match, 1.0 best and 0.0 worst.

    For perturbation p, the distance from its prediction to every real profile
    is ranked; the score is 1 - rank(own) / (n - 1). Same definition as the
    challenge metric, computed here so it can also run on residual vectors.
    """
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    r = torch.as_tensor(np.asarray(real, dtype=np.float32), device=dev)
    p = torch.as_tensor(np.asarray(pred, dtype=np.float32), device=dev)
    n = r.shape[0]
    if n < 3:
        return {"per_perturbation": np.full(n, np.nan), "mean": float("nan"),
                "median": float("nan"), "metric": metric}
    d = torch.cdist(p, r, p=1.0 if metric == "l1" else 2.0)
    own = d.diagonal()
    rank = (d < own.unsqueeze(1)).sum(dim=1).float()
    scores = (1.0 - rank / (n - 1)).cpu().numpy()
    return {
        "per_perturbation": scores,
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "metric": metric,
    }


def cell_eval_discrimination(
    real: np.ndarray,
    pred: np.ndarray,
    perturbations: np.ndarray,
    gene_names: np.ndarray,
    metric: str = "l1",
) -> dict:
    """Arc Institute's ``cell-eval`` discrimination score on effect profiles.

    One pseudobulk profile per perturbation plus a zero-effect control row,
    which is what a control profile is on the effect scale.
    """
    import anndata as ad
    import cell_eval
    import pandas as pd
    from cell_eval._types._anndata import PerturbationAnndataPair
    from cell_eval.metrics import discrimination_score

    control = "non-targeting"
    labels = np.concatenate([np.asarray(perturbations, dtype=object), np.array([control])])
    obs = pd.DataFrame({"perturbation": pd.Categorical(labels)})
    obs.index = [f"cell{i}" for i in range(labels.size)]
    var = pd.DataFrame(index=np.asarray(gene_names, dtype=str))

    def build(x):
        m = np.vstack([np.asarray(x, dtype=np.float32), np.zeros((1, x.shape[1]), np.float32)])
        return ad.AnnData(X=m, obs=obs.copy(), var=var.copy())

    pair = PerturbationAnndataPair(
        real=build(real), pred=build(pred), pert_col="perturbation", control_pert=control
    )
    out = discrimination_score(pair, metric=metric, exclude_target_gene=True)
    values = np.array([v for v in out.values()], dtype=float)
    try:
        from importlib.metadata import version as _pkg_version

        installed = _pkg_version("cell-eval")
    except Exception:  # noqa: BLE001
        installed = getattr(cell_eval, "__version__", "unknown")
    return {
        "package_version": installed,
        "metric": metric,
        "per_perturbation": out,
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "n": int(values.size),
    }

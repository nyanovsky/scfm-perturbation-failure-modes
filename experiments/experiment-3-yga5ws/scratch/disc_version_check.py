"""Does the discrimination number depend on which cell-eval version computes it?

The pre-registration pins ``cell-eval==0.5.19``. That version was never
released: PyPI publishes 0.1.0 and then 0.5.33 onward, and the upstream tag list
jumps 0.5.18 -> 0.5.21. The sweep therefore resolved 0.8.2. This script scores
the *same* saved prediction matrices under 0.8.2 and under the two real versions
either side of the pinned string, so the deviation can be reported with a
measured size rather than waved through.

Run once per version, each with its own --target directory on PYTHONPATH:

    PYTHONPATH=/tmp/ce_0_8_2 python3 scratch/disc_version_check.py --tag 0.8.2
    PYTHONPATH=/tmp/ce_0_5_18 <py3.11> scratch/disc_version_check.py --tag 0.5.18

then merge the per-version JSONs with --merge.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
PRED_DIR = REPO / "scratch" / "dl" / "predictions"
OUT_DIR = REPO / "results" / "discrimination"


def _load_discrimination_module():
    """Load discrimination.py by path: the package __init__ pulls in torch,
    which the 3.11 interpreter used for the old cell-eval does not have."""
    path = REPO / "src" / "scfm_pfm" / "discrimination.py"
    spec = importlib.util.spec_from_file_location("_disc", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def score_one(path: pathlib.Path, disc) -> dict:
    z = np.load(path, allow_pickle=False)
    real = z["measured_effect"].astype(np.float32)
    pred = z["prediction_effect_scale"].astype(np.float32)
    out = disc.cell_eval_discrimination(
        real=real,
        pred=pred,
        perturbations=z["perturbation"].astype(str),
        gene_names=z["gene_name"].astype(str),
    )
    return {
        "file": path.name,
        "n_perturbations": int(real.shape[0]),
        "n_genes": int(real.shape[1]),
        "mean": out["mean"],
        "median": out["median"],
        "package_version": out["package_version"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="", help="label for this version's output file")
    ap.add_argument("--merge", action="store_true", help="merge the per-version files")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        rows: list[dict] = []
        for p in sorted(OUT_DIR.glob("cell_eval_*.json")):
            rec = json.loads(p.read_text())
            for r in rec["runs"]:
                rows.append({"asked_for": rec["tag"], **r})
        by_file: dict[str, dict[str, float]] = {}
        for r in rows:
            by_file.setdefault(r["file"], {})[r["package_version"]] = r["mean"]
        spread = {
            f: {
                "values": v,
                "max_abs_difference": (max(v.values()) - min(v.values())) if len(v) > 1 else 0.0,
            }
            for f, v in by_file.items()
        }
        payload = {
            "pinned_in_preregistration": "0.5.19",
            "pinned_version_exists": False,
            "why": (
                "PyPI publishes cell-eval 0.1.0 then 0.5.33 onward; the upstream tag list "
                "jumps 0.5.18 -> 0.5.21. 0.5.19 was never released, so the pin was "
                "unattainable when it was written."
            ),
            "resolved_in_the_sweep": "0.8.2",
            "runs": rows,
            "per_file": spread,
            "max_abs_difference_any_file": max(
                (s["max_abs_difference"] for s in spread.values()), default=0.0
            ),
        }
        dest = REPO / "results" / "discrimination_versions.json"
        dest.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload["per_file"], indent=2))
        print(f"-> {dest}")
        return 0

    disc = _load_discrimination_module()
    runs = [score_one(p, disc) for p in sorted(PRED_DIR.glob("*.npz"))]
    for r in runs:
        print(f"  {r['file']}: {r['mean']:.6f} (cell-eval {r['package_version']})", flush=True)
    tag = args.tag or "unknown"
    (OUT_DIR / f"cell_eval_{tag.replace('.', '_')}.json").write_text(
        json.dumps({"tag": tag, "python": sys.version.split()[0], "runs": runs}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

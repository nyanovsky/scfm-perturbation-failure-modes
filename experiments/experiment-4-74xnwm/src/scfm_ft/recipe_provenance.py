"""Check every recipe setting against scGPT's published perturbation tutorial.

    curl -sSo /tmp/pert.ipynb \
        https://raw.githubusercontent.com/bowang-lab/scGPT/main/tutorials/Tutorial_Perturbation.ipynb
    PYTHONPATH=src apy -m scfm_ft.recipe_provenance --notebook /tmp/pert.ipynb

``config.Recipe`` claims to be scGPT's published perturbation recipe. That claim
is worth being able to check rather than assert, and it is easy to get wrong:
several of these settings differ between the ``TransformerGenerator`` class
signature and the tutorial that constructs it, and the tutorial is the recipe.
``dropout`` defaults to 0.5 in the class and the tutorial passes 0;
``pert_pad_id`` defaults to 2 in the class and the tutorial passes 0.

So this reads the tutorial notebook's own settings cell, parses the literal
assignments out of it, and compares them field by field with what this run used.
The notebook is not vendored: the URL and the sha256 of the file the comparison
was run against are recorded in the output, so a re-reader can fetch the same
bytes and repeat it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from . import config as C
from . import paths

TUTORIAL_URL = (
    "https://raw.githubusercontent.com/bowang-lab/scGPT/main/"
    "tutorials/Tutorial_Perturbation.ipynb"
)
#: the file this comparison was run against
TUTORIAL_SHA256 = "a997191d88e4c05969103e777f1323ebf4263238e64520877ae3a0f6aed4bdbb"

#: each recipe field beside the tutorial variable that fixes it. A field with no
#: tutorial variable is listed with why, so the table has no silent gaps.
FIELDS: tuple[tuple[str, str | None, str], ...] = (
    ("lr", "lr", ""),
    ("batch_size", "batch_size", ""),
    ("eval_batch_size", "eval_batch_size", ""),
    ("epochs", "epochs", ""),
    ("schedule_interval", "schedule_interval", ""),
    ("early_stop", "early_stop", ""),
    ("dropout", "dropout", "the class default is 0.5; the tutorial passes this"),
    ("pad_token", "pad_token", ""),
    ("pad_value", "pad_value", ""),
    ("pert_pad_id", "pert_pad_id", "the class default is 2; the tutorial passes this"),
    ("include_zero_gene", "include_zero_gene", ""),
    ("max_seq_len", "max_seq_len", ""),
    ("amp", "amp", ""),
    ("load_param_prefixes", "load_param_prefixs", "spelled load_param_prefixs upstream"),
    ("schedule_gamma", None, "passed inline to StepLR(..., gamma=0.9), not a named variable"),
    ("grad_clip", None, "the tutorial clips to 1.0 inline in its training loop"),
)


def settings_cell(nb: dict) -> str:
    """The tutorial's settings cell: the one that fixes the training budget.

    Located by content rather than index, so a cell inserted above it upstream
    does not silently make this compare against the wrong source.
    """
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "early_stop" in src and "epochs" in src and "pert_pad_id" in src:
            return src
    raise RuntimeError("the tutorial's settings cell was not found")


def training_loop_cell(nb: dict) -> str:
    """The tutorial's training-loop cell: the one that fixes the selection rule.

    Located by content for the same reason the settings cell is.
    """
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "best_model" in src and "patience" in src and "for epoch in range" in src:
            return src
    raise RuntimeError("the tutorial's training-loop cell was not found")


def selection_rule(src: str) -> dict:
    """Which validation quantity the tutorial keeps a checkpoint on, and which way.

    Read out of the loop cell rather than asserted, because it is the one recipe
    choice that is not a named setting and it decides which weights every
    downstream number is read off. Three things are extracted: what ``val_score``
    is assigned, the comparison that guards the ``best_model`` copy, and whether
    the early-stopping counter is reset by the same comparison.
    """
    tree = ast.parse(src)
    metric, compare, resets = None, None, False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "val_score"
        ):
            metric = ast.unparse(node.value)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            body = "".join(ast.unparse(s) for s in node.body)
            if "best_model" not in body:
                continue
            compare = ast.unparse(node.test)
            resets = "patience = 0" in body
    if metric is None or compare is None:
        raise RuntimeError("the tutorial's loop does not assign val_score or guard best_model")
    higher = isinstance(ast.parse(compare, mode="eval").body.ops[0], ast.Gt)  # type: ignore[attr-defined]
    return {
        "val_score_expression": metric,
        "best_model_guard": compare,
        "higher_is_better": higher,
        "early_stopping_counter_reset_by_the_same_guard": resets,
        "metric_key": metric.split('"')[1] if '"' in metric else metric.split("'")[1],
    }


#: what this run's training loop does, stated so the comparison above is a check
#: and not a description. ``scgpt_pert.train.finetune`` keeps the epoch with the
#: highest ``tutorial_pearson`` and resets its patience counter on the same
#: comparison; ``tutorial_pearson`` restates
#: ``compute_perturbation_metrics(...)["pearson"]``.
OUR_SELECTION = {
    "metric_key": "pearson",
    "higher_is_better": True,
    "early_stopping_counter_reset_by_the_same_guard": True,
}


def literals(src: str) -> dict[str, object]:
    """Every ``name = <literal>`` assignment in the cell, ignoring the rest."""
    out: dict[str, object] = {}
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            out[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return out


def _same(ours: object, theirs: object) -> bool:
    if isinstance(ours, tuple) or isinstance(theirs, (list, tuple)):
        return list(ours) == list(theirs)  # type: ignore[arg-type]
    if isinstance(ours, bool) or isinstance(theirs, bool):
        return bool(ours) is bool(theirs)
    if isinstance(ours, (int, float)) and isinstance(theirs, (int, float)):
        return float(ours) == float(theirs)
    return ours == theirs


def build(notebook: Path, out_dir: Path | None = None) -> Path:
    raw = Path(notebook).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    nb = json.loads(raw)
    cell = settings_cell(nb)
    theirs = literals(cell)
    recipe = C.RECIPE
    loop = training_loop_cell(nb)
    theirs_rule = selection_rule(loop)
    rule_matches = {k: theirs_rule[k] == v for k, v in OUR_SELECTION.items()}

    rows = []
    for field, name, note in FIELDS:
        ours = getattr(recipe, field)
        if name is None:
            rows.append({"field": field, "ours": ours, "tutorial_variable": None,
                         "tutorial": None, "matches": None, "note": note})
            continue
        if name not in theirs:
            raise RuntimeError(f"{name} is not assigned in the tutorial's settings cell")
        rows.append({
            "field": field,
            "ours": ours,
            "tutorial_variable": name,
            "tutorial": theirs[name],
            "matches": _same(ours, theirs[name]),
            "note": note,
        })

    compared = [r for r in rows if r["matches"] is not None]
    doc = {
        "source": TUTORIAL_URL,
        "sha256": digest,
        "sha256_matches_recorded": digest == TUTORIAL_SHA256,
        "settings_cell": cell,
        "n_compared": len(compared),
        "n_matching": sum(1 for r in compared if r["matches"]),
        "mismatches": [r["field"] for r in compared if not r["matches"]],
        "not_named_in_the_cell": [r["field"] for r in rows if r["matches"] is None],
        "fields": rows,
        "selection": {
            "tutorial": theirs_rule,
            "ours": OUR_SELECTION,
            "matches": all(rule_matches.values()),
            "per_key": rule_matches,
            "training_loop_cell": loop,
        },
    }
    if not doc["selection"]["matches"]:
        doc["mismatches"] = [*doc["mismatches"], "selection_rule"]
    dest = Path(out_dir or paths.results_dir()) / "recipe_provenance.json"
    dest.write_text(json.dumps(doc, indent=2, default=str))
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook", required=True, help=f"a local copy of {TUTORIAL_URL}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dest = build(Path(a.notebook), Path(a.out) if a.out else None)
    doc = json.loads(dest.read_text())
    for r in doc["fields"]:
        mark = {True: "ok", False: "DIFFERS", None: "n/a"}[r["matches"]]
        print(f"{mark:>7}  {r['field']:<22} ours={r['ours']!r:<50} tutorial={r['tutorial']!r}")
    sel = doc["selection"]
    print(f"\n{'ok' if sel['matches'] else 'DIFFERS':>7}  selection rule           "
          f"tutorial keeps the epoch maximising {sel['tutorial']['val_score_expression']}")
    print(f"\n{doc['n_matching']}/{doc['n_compared']} settings match; "
          f"sha256 matches recorded: {doc['sha256_matches_recorded']}")
    print(f"wrote {dest}")
    return 0 if not doc["mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

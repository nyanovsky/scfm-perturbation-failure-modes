"""Re-read every claim in claims_manifest.json from its named source and key.

The manifest keys are human-readable locators; this resolves the grammars the
contract actually emits:

  parquet : ``col=value, ..., column=<metric>`` with optional ``arm=<glob>``
            plus ``statistic=max|min|count``
  json    : ``field=<dotted path>`` optionally followed by row selectors
            (``model=``, ``anchor=``, ``version=``, ``file=``, ``row=``,
            ``block=``, ``check=``) and a final ``statistic=<field>``

Prints one short line per claim that cannot be resolved or that disagrees with
the stored value, then a count.
"""

from __future__ import annotations

import fnmatch
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SELECTORS = (
    "model",
    "anchor",
    "version",
    "file",
    "row",
    "block",
    "check",
    "cell_line",
    "arm",
    "reference",
    "variant",
    "line",
    "name",
    "asked_for",
)


def parse_key(key: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in key.split(","):
        part = part.strip()
        if "=" in part:
            name, _, value = part.partition("=")
            out[name.strip()] = value.strip()
    return out


def coerce(series: pd.Series, value: str):
    if series.dtype.kind in "ifu":
        try:
            return float(value)
        except ValueError:
            return value
    if series.dtype.kind == "b":
        return value.lower() in {"true", "1", "yes"}
    return value


def from_table(df: pd.DataFrame, key: str):
    parts = parse_key(key)
    column = parts.pop("column", None)
    statistic = parts.pop("statistic", None)
    parts.pop("scope_note", None)
    parts.pop("note", None)
    sub = df
    for name, value in parts.items():
        col = name[:-1] if name.endswith("s") and name[:-1] in sub.columns else name
        if col not in sub.columns:
            return None, f"no column {name!r}"
        if any(ch in value for ch in "*?"):
            mask = sub[col].astype(str).map(lambda v, p=value: fnmatch.fnmatch(v, p))
        else:
            mask = sub[col] == coerce(sub[col], value)
        sub = sub[mask]
    if column is None or column not in sub.columns:
        return None, f"no metric column {column!r}"
    vals = sub[column].dropna()
    if statistic in {"max", "min", "median", "mean", "count", "sum"}:
        if vals.empty:
            return None, "no rows match"
        return getattr(vals, statistic)(), None
    uniq = vals.unique()
    if len(uniq) != 1:
        return None, f"{len(uniq)} rows match"
    return uniq[0], None


def dotted(obj, path: str):
    """Walk a dotted path.  ``[*]`` fans out and the rest is applied to each item."""
    cur = obj
    tokens = [t for t in path.replace("/", ".").split(".") if t]
    for i, token in enumerate(tokens):
        if "[" in token:
            token, _, idx = token.partition("[")
            idx = idx.rstrip("]")
            if token:
                if isinstance(cur, dict) and token in cur:
                    cur = cur[token]
                else:
                    return None, f"no key {token!r}"
            if idx == "*":
                items = list(cur.values()) if isinstance(cur, dict) else cur
                if not isinstance(items, list):
                    return None, "not a list"
                rest = ".".join(tokens[i + 1 :])
                if not rest:
                    return items, None
                out = []
                for item in items:
                    got, err = dotted(item, rest)
                    if err:
                        return None, err
                    out.extend(got if isinstance(got, list) else [got])
                return out, None
            if not isinstance(cur, list) or int(idx) >= len(cur):
                return None, f"bad index {idx}"
            cur = cur[int(idx)]
            continue
        if isinstance(cur, dict) and token in cur:
            cur = cur[token]
        else:
            return None, f"no key {token!r}"
    return cur, None


ALIASES = {"version": "values", "check": "checks", "block": "blocks", "row": "rows"}

#: aggregates a json locator may take over a list it fanned out to
AGGREGATES = {
    "max": max, "min": min, "sum": sum, "count": len,
    "mean": lambda v: sum(v) / len(v),
    "max_abs": lambda v: max(abs(x) for x in v),
}

COMPOSITE_KEYS = (
    ("model", "cell_line", "anchor"),
    ("network", "dataset"),
    ("model", "anchor"),
    ("cell_line", "arm"),
    ("arm", "reference"),
    ("asked_for", "file"),
)


def _composite_ids(item: dict) -> set[str]:
    out = set()
    for cols in COMPOSITE_KEYS:
        if all(c in item for c in cols):
            out.add("_".join(str(item[c]) for c in cols))
    return out


def pick_row(items: list, wanted: dict[str, str]):
    hits = list(items)
    for name, value in wanted.items():
        exact = [
            item
            for item in hits
            if isinstance(item, dict)
            and (str(item.get(name, "")) == value or value in {str(v) for v in item.values()})
        ]
        if not exact:
            exact = [
                item
                for item in hits
                if isinstance(item, dict) and value in _composite_ids(item)
            ]
        hits = exact
    return hits[0] if len(hits) == 1 else None


def _apply_selectors(cur, selectors: dict[str, str]):
    guard = 0
    while selectors and guard < 8:
        guard += 1
        if isinstance(cur, list):
            hit = pick_row(cur, selectors)
            if hit is None:
                for name, value in selectors.items():
                    one = pick_row(cur, {name: value})
                    if one is not None:
                        cur, hit = one, one
                        selectors.pop(name)
                        break
                else:
                    return None, "no matching row for " + ", ".join(
                        f"{k}={v}" for k, v in selectors.items()
                    )
            else:
                cur = hit
                selectors.clear()
            continue
        if not isinstance(cur, dict):
            return None, "selector on a scalar"
        for name, value in list(selectors.items()):
            if value in cur:
                cur = cur[value]
                selectors.pop(name)
                break
            container = next(
                (c for c in (name + "s", "per_" + name, ALIASES.get(name, "")) if c and c in cur),
                None,
            )
            if container is not None:
                inner = cur[container]
                if isinstance(inner, dict) and value in inner:
                    cur = inner[value]
                    selectors.pop(name)
                    break
                cur = inner
                break
        else:
            return None, "no " + ", ".join(f"{k}={v}" for k, v in selectors.items())
    return cur, None


def _resolve(obj, field, selectors, statistic, selectors_first: bool):
    cur = obj
    if selectors_first:
        cur, err = _apply_selectors(cur, dict(selectors))
        if err:
            return None, err
        if field is not None:
            cur, err = dotted(cur, field)
            if err:
                return None, err
    else:
        if field is not None:
            cur, err = dotted(cur, field)
            if err:
                return None, err
        cur, err = _apply_selectors(cur, dict(selectors))
        if err:
            return None, err
    if statistic:
        if isinstance(cur, dict) and statistic in cur:
            return cur[statistic], None
        if isinstance(cur, list) and statistic in AGGREGATES:
            if statistic == "count":
                return len(cur), None
            vals = [float(v) for v in cur if isinstance(v, (int, float))]
            if not vals:
                return None, f"no numbers to take {statistic!r} of"
            return AGGREGATES[statistic](vals), None
        return None, f"no statistic {statistic!r}"
    return cur, None


def from_json_key(obj, key: str):
    parts = parse_key(key)
    statistic = parts.pop("statistic", None)
    field = parts.pop("field", None)
    selectors = {k: v for k, v in parts.items() if k in SELECTORS}
    first = _resolve(obj, field, selectors, statistic, selectors_first=True)
    if first[1] is None:
        return first
    second = _resolve(obj, field, selectors, statistic, selectors_first=False)
    return second if second[1] is None else first


def main() -> int:
    manifest = json.loads((ROOT / "claims_manifest.json").read_text())
    claims = manifest["claims"]
    cache: dict[str, object] = {}
    bad: list[str] = []
    ok = 0
    derived = 0
    for claim in claims:
        src = ROOT / claim["source"]
        label = f"{claim['source']} [{claim['key']}]"
        if not src.exists():
            bad.append(f"MISSING SOURCE  {label}")
            continue
        if str(src) not in cache:
            if src.suffix == ".parquet":
                cache[str(src)] = pd.read_parquet(src)
            elif src.suffix == ".csv":
                cache[str(src)] = pd.read_csv(src)
            elif src.suffix == ".json":
                cache[str(src)] = json.loads(src.read_text())
            else:
                cache[str(src)] = None
        if "derived=" in claim["key"]:
            derived += 1
            continue
        obj = cache[str(src)]
        if obj is None:
            bad.append(f"UNREADABLE      {label}")
            continue
        if isinstance(obj, pd.DataFrame):
            got, err = from_table(obj, claim["key"])
        else:
            got, err = from_json_key(obj, claim["key"])
        if err is not None:
            bad.append(f"UNRESOLVED      {label} :: {err}")
            continue
        want = claim["value"]
        if isinstance(got, list) and got and all(isinstance(v, bool) for v in got):
            got = f"{sum(got)} of {len(got)}"
        if isinstance(want, (int, float)) and not isinstance(want, bool):
            if not isinstance(got, (int, float, bool)):
                bad.append(f"NOT A NUMBER    {label} :: got {type(got).__name__}")
                continue
            if not math.isclose(float(got), float(want), rel_tol=2e-3, abs_tol=1e-6):
                bad.append(f"MISMATCH        {label} :: got {got!r} want {want!r}")
                continue
        elif str(got)[:80] != str(want)[:80]:
            bad.append(f"MISMATCH        {label} :: got {str(got)[:80]!r} want {str(want)[:80]!r}")
            continue
        ok += 1
    for line in bad:
        print(line[:300])
    print(f"{ok} of {len(claims) - derived} re-readable claims matched their source "
          f"({derived} more are differences between two claimed rows and state their formula "
          f"in the key)")
    kinds: dict[str, int] = {}
    for line in bad:
        kinds[line.split()[0]] = kinds.get(line.split()[0], 0) + 1
    for kind, count in sorted(kinds.items()):
        print(f"  {kind}: {count}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

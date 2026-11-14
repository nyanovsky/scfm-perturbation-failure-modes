"""Import run 2's and run 3's code without modifying either.

Both earlier runs ship a package literally called ``scfm_pfm``, in two
different experiment directories. Putting both on ``PYTHONPATH`` would make one
shadow the other, so each is registered under its own alias with its real
directory as the package ``__path__``. Relative imports inside the package
(``from .. import config``) then resolve within that alias, and both trees stay
importable in one process, byte-for-byte as they were merged.

Run 2 supplies the model adapters, the anchor construction and the in-scope
perturbation set; run 3 supplies the retrieval score, the response groups and
the folds. Run 5 adds no scientific code to either: it subclasses run 2's query
adapters to expose hidden states, which is why run 2 is imported and never
edited --- its shipped blocks, manifests and reports must stay exactly as the
paper cites them.

Copied from run 4's ``scfm_ft/sibling.py`` rather than imported from it, because
run 4 is a sibling and not a dependency; the alias trick is the shared thing,
not the module.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

RUN2_DIRNAME = "experiment-2-nygwcv"
RUN3_DIRNAME = "experiment-3-yga5ws"
RUN2_ALIAS = "scfm_pfm_r2"
RUN3_ALIAS = "scfm_pfm_r3"


def experiments_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _register(alias: str, dirname: str) -> types.ModuleType:
    if alias in sys.modules:
        return sys.modules[alias]
    root = experiments_dir() / dirname / "src" / "scfm_pfm"
    if not root.is_dir():
        raise RuntimeError(f"sibling package not found: {root}")
    pkg = types.ModuleType(alias)
    pkg.__path__ = [str(root)]  # type: ignore[attr-defined]
    pkg.__package__ = alias
    sys.modules[alias] = pkg
    return pkg


def run2(module: str | None = None):
    """Run 2's package, or one of its submodules (``"features.extract"``)."""
    _register(RUN2_ALIAS, RUN2_DIRNAME)
    if module is None:
        return sys.modules[RUN2_ALIAS]
    return importlib.import_module(f"{RUN2_ALIAS}.{module}")


def run3(module: str | None = None):
    """Run 3's package, or one of its submodules (``"translate.runner"``)."""
    _register(RUN3_ALIAS, RUN3_DIRNAME)
    if module is None:
        return sys.modules[RUN3_ALIAS]
    return importlib.import_module(f"{RUN3_ALIAS}.{module}")


def run2_dir() -> Path:
    return experiments_dir() / RUN2_DIRNAME


def run3_dir() -> Path:
    return experiments_dir() / RUN3_DIRNAME

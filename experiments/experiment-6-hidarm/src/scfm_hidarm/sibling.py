"""Import runs 2, 3, 4 and 5 without modifying any of them.

Runs 2 and 3 both ship a package literally called ``scfm_pfm``, in two different
experiment directories, so putting both on ``PYTHONPATH`` would make one shadow
the other. Each is registered under its own alias with its real directory as the
package ``__path__``; relative imports inside the package (``from .. import
config``) then resolve within that alias, and both trees stay importable in one
process, byte-for-byte as they were merged. Runs 4 and 5 have distinct package
names and only need their ``src`` on the path.

Why import rather than copy, given that runs 1-3 copied each other's code: run 6
is scored by run 3's rule and its entire claim is that the rule did not change.
A copy of ``metrics.py`` or ``translate/`` that drifted by one line would make
run 6's numbers incomparable with the ~2,300 already scored, and the drift would
be silent. Run 4 made the same call for the same reason.

The consequence is the one run 4 already carries: a change to run 3's
``metrics.py`` or ``translate/`` now reaches run 4 AND run 6. Check all three
before editing any of them --- and run 6 edits none.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

RUN2_DIRNAME = "experiment-2-nygwcv"
RUN3_DIRNAME = "experiment-3-yga5ws"
RUN4_DIRNAME = "experiment-4-74xnwm"
RUN5_DIRNAME = "experiment-5-hidprb"

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


def _add_src(dirname: str) -> Path:
    src = experiments_dir() / dirname / "src"
    if not src.is_dir():
        raise RuntimeError(f"sibling source tree not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


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


def run4(module: str | None = None):
    """Run 4's ``scfm_ft`` package, or one of its submodules (``"score"``).

    Run 4's own ``sibling.py`` registers runs 2 and 3 under the same two aliases
    this module uses, so importing run 4 after run 6 has already registered them
    reuses the same module objects rather than building a second copy.
    """
    _add_src(RUN4_DIRNAME)
    if module is None:
        return importlib.import_module("scfm_ft")
    return importlib.import_module(f"scfm_ft.{module}")


def run5(module: str | None = None):
    """Run 5's ``scfm_probe`` package, or a submodule (``"models.hidden"``)."""
    _add_src(RUN5_DIRNAME)
    if module is None:
        return importlib.import_module("scfm_probe")
    return importlib.import_module(f"scfm_probe.{module}")


def run2_dir() -> Path:
    return experiments_dir() / RUN2_DIRNAME


def run3_dir() -> Path:
    return experiments_dir() / RUN3_DIRNAME


def run4_dir() -> Path:
    return experiments_dir() / RUN4_DIRNAME


def run5_dir() -> Path:
    return experiments_dir() / RUN5_DIRNAME

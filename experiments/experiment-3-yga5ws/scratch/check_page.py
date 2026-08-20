"""Check the built page for leaked environment paths, refs, or tokens."""

from __future__ import annotations

import os
import pathlib
import re

path = pathlib.Path(os.environ["SILICO_APP_DIR"]) / "index.html"
s = path.read_text()
patterns = {
    "SILICO_ env var": r"SILICO_[A-Z_]+",
    "artifact ref": r"artifact://",
    "silico.run url": r"silico\.run",
    "state path": r"/srv/silico-" + "state",
    "home path": r"/home/[a-z0-9_-]+",
    "token-ish": r"(?i)(api[_-]?key|secret|bearer\s)",
}
for name, pat in patterns.items():
    hits = re.findall(pat, s)
    print(f"{name:16s} {len(hits)} {sorted(set(hits))[:4]}")
print(f"bytes {len(s)}")

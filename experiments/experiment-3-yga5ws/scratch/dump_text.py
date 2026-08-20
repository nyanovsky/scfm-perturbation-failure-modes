"""Print the visible text of the built report page, for proof-reading."""

from __future__ import annotations

import html
import os
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["SILICO_APP_DIR"]) 
if path.is_dir():
    path = path / "index.html"
s = path.read_text()
s = re.sub(r"<script.*?</script>", " ", s, flags=re.S)
s = re.sub(r"<style.*?</style>", " ", s, flags=re.S)
s = re.sub(r"</(p|li|h1|h2|h3|tr|figcaption|div|section|td|th)>", "\n", s)
s = re.sub(r"<(li|tr)>", "- ", s)
s = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", s)
txt = re.sub(r"<[^>]+>", " ", s)
txt = html.unescape(txt)
txt = re.sub(r"[ \t]+", " ", txt)
txt = re.sub(r"\n\s*\n+", "\n", txt)
print(txt.strip())

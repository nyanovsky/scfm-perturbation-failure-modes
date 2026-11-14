# figure-pdf

Static PDF exports of any tracked figure bundle, with **no experiment re-run**.

Only run 2's builder ever wrote PDFs — `report/figures.py` calls `write_image` for PDF and
PNG at `PNG_SCALE = 3`. Runs 1 and 3 stop at the interactive HTML. This tool gives the other
two runs static exports without going near the builders.

## Why it does not need the data

Each bundle's `<name>.html` embeds the whole Plotly figure — traces, layout and the house
template — in its `Plotly.newPlot` call. Rebuilding from that needs no effect matrices, no
feature blocks, no artifact store and no GPU.

That matters twice over:

- **Run 1 cannot be rebuilt any other way.** Its `report.figures --root` reads the artifact
  mirror, which is gitignored and absent off-platform.
- **Going through the builders would overwrite the tracked figures.** Run 2's
  `report.build` calls `figmod.build_all(bundles, repo)` with no flag to redirect it. Reading
  the committed HTML instead exports the figures *as the platform rendered them*, not as a
  local rebuild would.

## Setup

Kaleido 1.x needs a Chrome binary, which this machine did not have. It installs user-local,
no root:

```sh
uv pip install --python <venv>/bin/python plotly kaleido certifi

# the uv-managed CPython ships no CA bundle, so the download fails on SSL without this
export SSL_CERT_FILE=$(<venv>/bin/python -c "import certifi;print(certifi.where())")
<venv>/bin/plotly_get_chrome -y --path /tmp/chrome

# Kaleido only looks in choreographer's deps dir, and expects the chrome-linux64/ layout
DEST=$(<venv>/bin/python -c "from choreographer.cli import _cli_utils as c; print(c.default_download_path)")
mkdir -p "$DEST" && cp -r /tmp/chrome/chrome-linux64 "$DEST/chrome-linux64"
```

Chrome currently lives at `~/.local/share/choreographer/deps/chrome-linux64/` (~150 MB).

## Use

```sh
# everything, writing <name>.pdf beside each <name>.html
python tools/figure-pdf/export_pdfs.py experiments/*/figures/*/*.html

# one run, into a staging directory instead
python tools/figure-pdf/export_pdfs.py \
    experiments/experiment-3-yga5ws/figures/*/*.html --out /tmp/staged
```

An existing PDF is **skipped** unless `--force` is passed, so run 2's nine platform-rendered
PDFs are not clobbered by a careless invocation. All 30 figures export in about 70 seconds.

## Validation

Run 2's nine PDFs were rendered on the platform, so they are ground truth for this method.
Round-tripping them — HTML → figure → PDF — and comparing against the tracked originals gives
**9 of 9 matching on MediaBox and page count, 0 geometry mismatches**:

```sh
python - <<'PY'
import re, pathlib
def info(p):
    b = pathlib.Path(p).read_bytes()
    return re.search(rb'/MediaBox\s*\[([^\]]*)\]', b).group(1), b.count(b'/Type /Page')
base = pathlib.Path("experiments/experiment-2-nygwcv/figures")
for d in sorted(x for x in base.iterdir() if x.is_dir()):
    print(d.name, info(d / f"{d.name}.pdf"))
PY
```

## Hover-only data, and what is done about it

A PDF has no hover layer. Ten of the thirty bundles reference `%{text}` in their
`hovertemplate`, so a naive export silently drops whatever that text carried — the points
are still plotted, but nothing says which is which.

Most of the ten lose nothing a reader would miss: they are scatters of 3.7k–30k gene-level
points where no one reads an individual label. Five were degraded enough to matter, and
`export_pdfs.py` handles them through the explicit `LABEL_RULES` table:

| figure | run | treatment |
|---|---|---|
| `split_gap` | 3 | labels drawn; null cloud suppressed; redundant `"<line>, "` prefix stripped |
| `metric_check` | 3 | labels drawn; null cloud suppressed |
| `topk_overlap_grid` | 1 | labels drawn (4 strata per panel) |
| `feature_scale` | 2 | **left HTML-only** — see below |
| `transfer_mode_agreement` | 3 | **left HTML-only** — 284 labelled points, illegible at any size |

`feature_scale` stacks four labels per model on a log axis where the medians land almost on
top of one another; the result was unreadable. It loses least by being skipped, because its
marker *symbol* already separates median from 99th percentile — only the cell line is
hover-only.

Labels are placed per point, left or right of the marker according to which half of its own
subplot the point sits in, because Plotly autoscales to markers and not to the text beside
them, so a label on an extreme point would otherwise be clipped. Linear axes are then
widened to make room; **log axes are left alone**, since their `range` is in log10 units and
widening one in data units drives the real data off the canvas.

**The three labelled PDFs therefore differ from the HTML they came from.** That is the
deliberate trade: the static file is self-contained instead of faithful. The other 27 are
exported exactly as drawn, and `--no-labels` does that for all of them.

## Known differences from the platform's PDFs

Both are cosmetic; the rendered pages are geometrically identical.

**File size.** Some bar-chart figures come out much larger — `wiring_checks` is 73 KB as
committed versus ~1.25 MB regenerated. This is font *embedding*, not data or geometry: the
figure has only 48 points. The committed PDF references `DejaVuSans` in 73 KB; today's Chrome
writes a full font program. Pinning the layout font to DejaVu Sans explicitly changes the
font name but not the size, so the renderer, not the font stack, is the cause.

**Typeface.** The house stack is `'Suisse Intl', -apple-system, BlinkMacSystemFont, Arial,
sans-serif`. `Suisse Intl` is installed on neither machine, so both fall back — the platform
to DejaVu Sans, this machine to Liberation Sans via the stack's `Arial` entry.

Neither affects what the figures show. If exact visual parity with run 2's committed PDFs
ever matters, set `fig.update_layout(font_family="DejaVu Sans, sans-serif")` before export.

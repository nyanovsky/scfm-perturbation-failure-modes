# silico-figures-local

A local stand-in for the platform's `silico-figures`, so runs 2 and 3 can rebuild their
reports off-platform.

Every report builder in this repository does `from silico_figures import ...` at module
level. On Goodfire's Silico platform that package was a local path dependency at
`/opt/silico/plugins/figures`. That path does not exist here, which is the caveat recorded
in `experiments/experiment-3-yga5ws/results/verification/report_rebuild.json`:

> The house figure-style package is a local path dependency supplied by the platform, so a
> clone taken off-platform needs that package made available first.

This is that package made available. **It is a reconstruction, not the original.**

## What is reconstructed, and how faithfully

The theme in `src/silico_figures/_theme.json` was not invented. It was extracted verbatim
from the Plotly `layout.template` embedded in a figure bundle the *real* package produced —
`experiments/experiment-3-yga5ws/figures/arm_scores/arm_scores.html` — so the colourway,
fonts, margins, axis styling, hover labels, legend placement, modebar and colorscales are
the platform's own values. `PLOTLY_CONFIG` was read off the same bundle.

`add_reference_line`'s guide-line and label geometry was likewise read off tracked bundles
and is verified against them: the shim's emitted shape and annotation dictionaries are
compared field-for-field with exemplars taken from `figures/arm_scores` (an x line on a
two-panel grid) and `figures/step_stability` (a y line striped across `row="all",
col="all"`). All nine checks pass.

What is **not** guaranteed to match is anything the real package did that leaves no trace in
a saved figure: accessibility linting, colourblind pair checking, static-export defaults, or
any layout tweak applied conditionally.

## What this does and does not affect

Report **prose and numbers do not pass through this package at all** — they are read from
the results tables by each run's `report/build.py`. The strongest evidence for that: run 2's
rebuilt `feature_extraction_report.md` and `no_op_queries.csv` are **byte-identical** to the
copies committed under `results/report/`, which is exactly the equality check
`report_rebuild.json` records.

Only the rendered figures differ, and only in ways a `to_html` fragment can differ. Rebuilt
HTML sizes come out close to, but not equal to, the platform's:

| report | platform | rebuilt here |
|---|---|---|
| run 2 `feature_extraction_report.html` | 4,999,781 | 5,002,785 |
| run 3 `index.html` | 136,401 | 189,383 |

Run 3's page grew because the run did: the platform figure is from before the protein control
and the two-table arm. Run 2's rebuild reproduces byte-for-byte at 5,002,785.

**Pin `plotly<7`.** plotly.py 7 emits a `plotly.js` 4.x CDN reference where the platform used
3.6.0, and changes the rendered byte sizes above. 6.x emits 3.7.0, which is the closest match
available.

## Rebuilding the reports

Build from a tracked-files-only export, which is the procedure `report_rebuild.json`
records, and which also keeps the builders from overwriting the tracked, house-styled figure
bundles in the working tree — **run 2's builder writes figures to `repo_dir()/figures` with
no flag to redirect it.**

```sh
cd /usr/users/nyanovsky/scfm-perturbation-failure-modes
SCRATCH=$(mktemp -d)

git archive HEAD | tar -x -C "$SCRATCH"

uv venv "$SCRATCH/venv" --python 3.12
uv pip install --python "$SCRATCH/venv/bin/python" \
    "pandas>=2.2" "pyarrow>=17" "numpy<3" "plotly>=6,<7"
uv pip install --python "$SCRATCH/venv/bin/python" -e tools/silico-figures-local

cd "$SCRATCH/experiments/experiment-2-nygwcv"
PYTHONPATH=src "$SCRATCH/venv/bin/python" -m scfm_pfm.report.build \
    --bundles results/report/bundles --out "$SCRATCH/out2"

cd "$SCRATCH/experiments/experiment-3-yga5ws"
PYTHONPATH=src "$SCRATCH/venv/bin/python" -m scfm_pfm.report.build \
    --summary results/summary --out "$SCRATCH/out3"

cd "$SCRATCH/experiments/experiment-4-74xnwm"
PYTHONPATH=src "$SCRATCH/venv/bin/python" -m scfm_ft.report.build \
    --summary results/summary --alt-summary results/summary_masked_mse --out "$SCRATCH/out4"
```

Run 4 builds through this package too. It needs no extra dependency: its `sibling.py` imports
run 3's package from the same export, so the export must be of the whole repository, not of one
run directory.

Then copy only the rendered pages back — not the figure directories.

Both builds are CPU-only, need no artifact-store credential, and touch no GPU.

`kaleido` is not installed above, so run 2's per-figure PDF/PNG exports fail and are skipped
with a printed reason. That path is already wrapped in `try/except` upstream; add `kaleido`
to the venv if you want the static exports.

## Verifying a rebuild

```sh
# run 2: prose and data must be byte-identical to the committed copies
md5sum "$SCRATCH/out2/feature_extraction_report.md" \
       experiments/experiment-2-nygwcv/results/report/feature_extraction_report.md

# run 3: the page must leak no platform paths, refs or tokens
cd "$SCRATCH/experiments/experiment-3-yga5ws"
SILICO_APP_DIR="$SCRATCH/out3" "$SCRATCH/venv/bin/python" scratch/check_page.py

# run 3: every claim re-read from its named source (expect 389 of 400 resolvable)
PYTHONPATH=src "$SCRATCH/venv/bin/python" scratch/verify_claims.py

# runs 3 and 4: the contract must regenerate byte-identically to the committed copies
PYTHONPATH=src "$SCRATCH/venv/bin/python" -m scfm_pfm.report.contract --summary results/summary
```

Run 4 has no `verify_claims.py` of its own and its `scratch/` is gitignored, so it is not in the
export. Regenerating its contract and diffing against the committed
`report_summary.json` / `claims_manifest.json` is the equivalent check.

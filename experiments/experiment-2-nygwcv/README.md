# Response features for four models over three query anchors

## What this is

Run 2 of the feature-swap thread. For each of scGPT, AIDO.Cell (GB.Cell-100M), scPRINT and
TranscriptFormer it extracts, at three query anchors and in four feature forms, the response of every
read-out gene to a change in every perturbed gene, plus the evidence that those numbers are
numerically trustworthy (finite-difference step stability, softmax coupling, read-out order alignment).

It answers no scientific question and compares no models. Run 3 trains a translator over these
blocks, swapping only the feature.

## Layout

| path | what |
|---|---|
| `src/scfm_pfm/config.py` | every grid, threshold and scope constant the run commits to |
| `src/scfm_pfm/paths.py` | the three roots: durable outputs, worktree handoff tables, checkpoint cache |
| `src/scfm_pfm/data/floor_sweep.py` | step 1: ceiling vs expression floor, and the read-out gene set |
| `src/scfm_pfm/data/demo_cells.py` | real non-targeting control cells (run 1 code, Jurkat added here) |
| `src/scfm_pfm/models/` | the four response adapters (run 1) extended with a batched query interface |
| `src/scfm_pfm/features/` | anchors, the extraction driver, step-stability sweep, alignment checks |
| `src/scfm_pfm/verify_features.py` | per-block verification (`--block-dir`) and the roll-up (`--all`) |
| `src/scfm_pfm/report/` | the nine figures and the report the artifacts assemble themselves into |
| `results/handoff/` | small tables that travel with the worktree into a job |
| `results/floor_sweep/` | the floor sensitivity sweep's outputs |
| `results/report/bundles/` | one verification record per feature block; the report reads only these |
| `results/verification/` | the roll-up: `verify_all.txt`, `verify_all.json`, per-block summaries |
| `figures/<name>/` | one directory per figure: interactive HTML, static PDF and PNG, `data.json` |

## How to reproduce

```sh
# step 1 (CPU, pod): needs run 1's persisted split-half vectors
python -m scfm_pfm.data.floor_sweep --vectors-dir <dir with shv_<line>.npz>

# checkpoints onto one job volume, reused read-only by every GPU job
python3 -m scfm_pfm.models.prefetch --cache _ckpt

# per (model, cell line) extraction job (GPU; mounts the checkpoint volume)
python -m scfm_pfm.features.extract --model <model> --cell-line <line>

# per-block verification (CPU; mounts that extraction job's volume)
python -m scfm_pfm.verify_features --block-dir auto --out <bundles dir>

# roll-up over the per-block records, then the report
python -m scfm_pfm.verify_features --all --bundles results/report/bundles
python -m scfm_pfm.report.build --bundles results/report/bundles
```

A block is verified inside a job that mounts the volume the extraction wrote, against that block's
own `npz` files. The pod never sees the feature arrays; it reads only the per-block records, which is
why the report can be rebuilt anywhere.

## Outputs

Everything durable lives under one prefix in the artifact store:

```
artifact://fundacion-instituto-leloir-553557/experiments/exp_01m0c7ecnnf3fvxskxj5nygwcv/
```

The worktree keeps only what is small enough to read and diff: the per-block verification
records, the roll-up, the figures, and the report's Markdown copy. The feature arrays and the
inline-Plotly HTML report are in the store.

### Feature blocks — `features/<model>/<cell_line>/`

One directory per (model, cell line). Inside:

| file | what |
|---|---|
| `anchor=<anchor>/vvp.npz` | value-value profile: response at each of the 5 grid values, `(n_perturbed, 5, n_readout)` |
| `anchor=<anchor>/gdt.npz` | gradient trajectory: derivative at each of the 5 base values, same shape |
| `anchor=<anchor>/zero_out.npz` | single knock-to-zero response, `(n_perturbed, n_readout)` |
| `anchor=<anchor>/query_summary.parquet` | one row per (perturbed gene, cell, value): cross-gene mean and sd, max abs, own-entry delta and rank |
| `gene_embeddings.npz` | the model's own gene embedding table, restricted to the read-out set |
| `manifest.json` | the block's whole contract: layout, anchors, value grid, sign convention, mask spec, step selection, wiring checks, checkpoint identity |
| `perturbed_genes.parquet`, `readout_genes.parquet` | the row and column keys; **matrix rows follow the first, columns the second** |
| `step_selection.json`, `step_sweep.parquet` | the finite-difference step chosen per anchor and the sweep behind it |
| `environment.json` | image, package versions, GPU, wall clock |

Each `npz` carries `raw` and `standardized`, plus `recentred` where the model's head ends in a
softmax. Anchors are `uniform_virtual`, `mean_control`, `individual_cells`, and
`mean_control_masked` for the two bidirectional models. `scgpt/<line>/context_comparison.npz`
additionally holds the three-context side check described in the report.

Per-gene values sit on four different output scales and are not comparable across models without
an explicit transformation. The median absolute zero-out response spans about three orders of
magnitude between models; the report's scale figure gives the numbers.

### Report and verification

| path | what |
|---|---|
| `report/feature_extraction_report.html` (store) | the report, nine figures inline, self-contained |
| `results/report/feature_extraction_report.md` (worktree) | the same text as a reading copy |
| `results/report/bundles/report_bundle_<model>_<line>.json` | one verification record per block; the report reads only these |
| `results/report/no_op_queries.csv` | per (model, cell line, anchor, value) count of queries the model could not see |
| `results/verification/verify_all.{txt,json}` | the roll-up over the whole artifact tree |
| `results/floor_sweep/` | the expression-floor sensitivity sweep and the resulting read-out gene set |
| `figures/<name>/` | interactive HTML, static PDF and PNG, and the `data.json` behind each figure |

### Provenance

Checkpoints, as recorded in each block's `manifest.json`:

| model | checkpoint | revision |
|---|---|---|
| scGPT | `wanglab/scGPT-human` | `a24c237737a40f3720f75abb555489e9fe753be6` |
| scPRINT | `jkobject/scPRINT:medium-v1.5.ckpt` | `8e1ad1a1bc40c1fc2f4f1700fe27923eea9a2bed` |
| TranscriptFormer | `s3://czi-transcriptformer/weights/tf_sapiens.tar.gz` | unversioned public bucket object |
| AIDO.Cell | `genbio-ai/GB.Cell-100M` (the plan's `AIDO.Cell-100M`, renamed upstream) | `a983b388e1025860f2c413ecef60e76078253c24` |

All GPU work ran as single-H100 on-demand jobs from the `job-core` image, one job per
(model, cell line), with the checkpoints mounted read-only from a shared prefetch volume.
Verification ran as CPU-only `python:3.11-slim` jobs mounting each extraction job's volume.
No external model API was used and no researcher credential was needed.

Every model needed some change to its shipped code to run here, 7 in total; each block's
`manifest.json` records its own verbatim, and the report's provenance section groups them. Three
are import plumbing, three swap a fused half-precision attention kernel for the same attention in
float32, and one remaps scGPT's fused query-key-value tensors onto torch's packed projection
(verified numerically at load time).

Cost, on one H100, for the three anchors of one (model, cell line): scGPT 89-114 s, scPRINT
167-176 s, TranscriptFormer about 1,085 s, AIDO.Cell about 10,573 s. Size run 3 against the last
number, not the first.

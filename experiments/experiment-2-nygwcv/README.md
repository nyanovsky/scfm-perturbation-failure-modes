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
| `src/scfm_pfm/paths.py` | the roots: durable outputs, worktree handoff tables, checkpoint cache |
| `src/scfm_pfm/forms.py` | the two array-level operations every block shares, torch-free |
| `src/scfm_pfm/data/floor_sweep.py` | step 1: ceiling vs expression floor, and the read-out gene set |
| `src/scfm_pfm/data/demo_cells.py` | real non-targeting control cells (run 1 code, Jurkat added here) |
| `src/scfm_pfm/models/` | the four response adapters (run 1) extended with a batched query interface |
| `src/scfm_pfm/features/scope.py` | which perturbed genes a block covers, and the coverage record |
| `src/scfm_pfm/features/extract.py` | anchors, the extraction driver, step-stability sweep, alignment checks |
| `src/scfm_pfm/features/hidden_readout.py` | visible / mask token / median level at one anchor |
| `src/scfm_pfm/features/cost_probe.py` | measured throughput, full-coverage projection, forward-mode probe |
| `src/scfm_pfm/features/annotate_coverage.py` | adds the coverage record to a block extracted before it existed |
| `src/scfm_pfm/features/feature_index.py` | the canonical block per (model, cell line) |
| `src/scfm_pfm/verify_features.py` | per-block verification (`--block-dir`) and the roll-up (`--all`) |
| `src/scfm_pfm/report/` | the figures and the report the artifacts assemble themselves into |
| `tests/` | the hidden-readout estimator against a plain reference on a toy model |
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

# per (model, cell line) extraction job (GPU; mounts the checkpoint volume).
# --full-scope covers every in-scope perturbation and writes to features_full_scope/
python -m scfm_pfm.features.extract --model <model> --cell-line <line> [--full-scope]

# the visible / mask-token / median-level comparison at the mean-control anchor (GPU)
python -m scfm_pfm.features.hidden_readout --model <model> --cell-line <line>

# what full coverage would cost for one model, and whether forward-mode autodiff runs (GPU)
python -m scfm_pfm.features.cost_probe --model aido_cell --batch-sizes 1 2 4 8

# per-block verification (CPU; mounts that extraction job's volume)
python -m scfm_pfm.verify_features --block-dir auto --out <bundles dir>

# roll-up over the per-block records, the block index, then the report
python -m scfm_pfm.verify_features --all --bundles results/report/bundles
python -m scfm_pfm.features.feature_index --bundles results/report/bundles
python -m scfm_pfm.report.build --bundles results/report/bundles

# the estimator behind the masking comparison, on a toy model (CPU, pod)
PYTHONPATH=src apy -m pytest tests -q
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

### Feature blocks — two trees, and an index that names the canonical one

Coverage differs by model, so there are two trees with the same layout:

| tree | what | which models |
|---|---|---|
| `features_full_scope/<model>/<cell_line>/` | every in-scope perturbation, at every anchor and in the masked variant | scGPT, scPRINT, TranscriptFormer |
| `features/<model>/<cell_line>/` | the first wave's 250 perturbed genes | AIDO.Cell (canonical); the other three are superseded |
| `feature_index/blocks.json` | one row per (model, cell line): the block to read, its coverage, what it supersedes | — |

"In scope" is not the same as the plan's strata. The S3 and S4 strata hold 991 perturbations in
HepG2 and 1,100 in Jurkat, but a perturbed gene has to be a column of the supplied set to be
perturbed at all, so only the 524 and 705 whose target is itself a read-out gene are reachable. Of
those, AIDO.Cell runs 250 because its measured cost is about 0.9 s per forward pass; the projection
for full coverage is in `report/cost_probe/` and in the report. Every manifest records its own
`coverage` block and the roll-up fails if one does not.

Blocks at different coverage share the smaller one's perturbed genes but **not** its row order.
Join on `perturbed_gene_id`, never on row index, and do not compare a per-block aggregate across
two coverages.

Inside a block:

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
`mean_control_masked` for the three bidirectional models. `scgpt/<line>/context_comparison.npz`
additionally holds the three-context side check described in the report.

### Hiding the read-out gene two ways — `features/<model>/<cell_line>/hidden_readout/`

For the three bidirectional models, the same 60-gene subset at the mean-control anchor under three
conditions: the gene's own input `visible`, replaced by the model's `masked` convention, and
replaced by one uninformative in-distribution level (`median_substituted`, the median pooled-control
level over read-out genes). This is what separates information removal from a mask-token artefact.
`manifest.json` carries the per-condition medians, the ratios, the per-gene agreement and the
substitute level; `condition=<c>/{zero_out,vvp}.npz` carry the arrays and a `read_coverage` array.

The estimator differs from the shipped masked block in one way, deliberately. A mask block that
also hides the perturbed gene overwrites the perturbation before the forward pass, so that pass
contributes exactly zero; the shipped block averaged those zeros in, shrinking its magnitudes by
1/(blocks per cycle) — 25% for scGPT, 14% for scPRINT and AIDO.Cell. Here they are skipped, and
`abs_median_run2_diluted_estimator` reproduces the shipped estimator exactly.

Per-gene values sit on four different output scales and are not comparable across models without
an explicit transformation. The median absolute zero-out response spans about three orders of
magnitude between models; the report's scale figure gives the numbers.

### Report and verification

| path | what |
|---|---|
| `report/feature_extraction_report.html` (store) | the report, figures inline, self-contained |
| `results/report/feature_extraction_report.md` (worktree) | the same text as a reading copy |
| `results/report/bundles/report_bundle_<model>_<line>.json` | the canonical verification record per block; the report reads only these |
| `results/report/bundles_subset250/` | the first wave's records for the six blocks now superseded |
| `results/report/hidden_readout/<model>_<line>.json` | the visible / mask-token / median-level comparison and its checks |
| `results/report/cost_probe/cost_probe_<model>.json` | measured throughput, the full-coverage projection, the forward-mode probe |
| `results/report/no_op_queries.csv` | per (model, cell line, anchor, value) count of queries the model could not see |
| `results/feature_index/blocks.{json,parquet}` | the canonical block per (model, cell line), with its coverage |
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

All GPU work ran as single-H100 on-demand jobs from the public `python:3.11-slim` image with
`envs/model_stack.sh` installing the pinned model stack at job start (torch 2.5.1+cu121, scgpt
0.2.4, scprint 2.3.8, transcriptformer 0.6.1); each block's `environment.json` records the exact
resolved package list. One job per (model, cell line), with the checkpoints mounted read-only from
a shared prefetch volume. Verification ran as CPU-only jobs from the same image mounting each
extraction job's volume. No external model API was used and no researcher credential was needed.

Every model needed some change to its shipped code to run here, 7 in total; each block's
`manifest.json` records its own verbatim, and the report's provenance section groups them. Three
are import plumbing, three swap a fused half-precision attention kernel for the same attention in
float32, and one remaps scGPT's fused query-key-value tensors onto torch's packed projection
(verified numerically at load time).

Cost, on one H100, for the three anchors of one (model, cell line), at the coverage each block
shipped: scGPT 372 s (524 perturbed genes) and 643 s (705), scPRINT 714 s and 1,196 s,
TranscriptFormer 4,621 s and 6,232 s, AIDO.Cell 10,575 s and 10,571 s at 250 genes. The masked
variant costs roughly as much again as one anchor. Size run 3 against the last number, not the
first: AIDO.Cell runs about 0.88 s per forward pass at every batch size tried, so full coverage
for it would be 44,484 s on HepG2 and 59,874 s on Jurkat.

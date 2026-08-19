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

Filled in at completion.

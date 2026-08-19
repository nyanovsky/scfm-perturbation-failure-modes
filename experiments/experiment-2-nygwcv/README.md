# Response features for four models over three query anchors

## What this is

Run 2 of the feature-swap thread. For each of scGPT, AIDO.Cell (GB.Cell-100M), scPRINT and
TranscriptFormer it extracts, at three query anchors and in four feature forms, the response of every
read-out gene to a change in every perturbed gene, plus the evidence that those numbers are
numerically trustworthy (finite-difference step stability, softmax coupling, wiring checks).

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
| `src/scfm_pfm/features/` | anchors, the extraction driver, step-stability sweep, wiring checks |
| `src/scfm_pfm/verify_features.py` | `--all`: the verification contract |
| `results/handoff/` | small tables that travel with the worktree into a job |
| `results/floor_sweep/` | the floor sensitivity sweep's outputs |
| `figures/<name>/` | one directory per figure: interactive HTML, static PDF and PNG, `data.json` |

## How to reproduce

```sh
# step 1 (CPU, pod): needs run 1's persisted split-half vectors
python -m scfm_pfm.data.floor_sweep --vectors-dir <dir with shv_<line>.npz>

# checkpoints onto one job volume, reused read-only by every GPU job
python3 -m scfm_pfm.models.prefetch --cache _ckpt

# per (model, cell line) extraction job
python -m scfm_pfm.features.extract --model <model> --cell-line <line>

# verification over the artifact tree
python -m scfm_pfm.verify_features --all
```

## Outputs

Filled in at completion.

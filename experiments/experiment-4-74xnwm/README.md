# Run 4 — does fine-tuning for perturbation prediction install the regulatory signal?

Run 3 found that a frozen single-cell foundation model's **queried response features** never leave
the shuffled-label band, while its **static gene table** carries everything. The obvious objection
is that nobody runs these models frozen. This run tests it: scGPT is fine-tuned on the Nadig screen
with its own published perturbation recipe, and the fine-tuned checkpoints are pushed through run
3's feature swap unchanged.

Five fine-tuning arms per cell line, varying only the gene table and whether it updates:

| arm | gene table | table during fine-tuning | what it isolates |
|---|---|---|---|
| `own_trainable` | scGPT's own | trainable | the standard published recipe |
| `own_frozen` | scGPT's own | frozen | gain attributable to the readout alone |
| `esm2_trainable` | ESM2-3B substituted | trainable | whether protein provenance is a better start |
| `esm2_frozen` | ESM2-3B substituted | frozen | the same, with the table held fixed |
| `rand_frozen` | random, ESM2's width | frozen | capacity-matched null for the two ESM2 arms |

## Layout

| path | what |
|---|---|
| `src/scfm_ft/config.py` | the pre-registration: recipe, arms, which folds the model may see |
| `src/scfm_ft/sibling.py` | registers run 2's and run 3's `scfm_pfm` packages under separate aliases so both are importable unmodified |
| `src/scfm_ft/ingest.py` | one streaming pass over the raw GEO screen, building the per-cell matrices the recipe trains on |
| `src/scfm_ft/scgpt_pert/` | scGPT's perturbation model with a substitutable gene table, the training loop, and the query adapter run 2's extractor drives |
| `src/scfm_ft/score.py` | run 3's translator and scoring rule, restricted to the folds the fine-tuned model never saw |
| `jobs/` | the job scripts, one per stage |

## Reproduce

```sh
# 1. per-cell matrices and the fold assignment, one CPU job per cell line
sh jobs/ingest.sh <hepg2|jurkat>

# 2. one GPU job per (arm, cell line): fine-tune, extract both query families, score
sh jobs/train_arm.sh <cell_line> <arm>

# 3. each model's own prediction, the whole-screen pooling, the last-epoch reading
sh jobs/followup.sh
sh jobs/rescore.sh <cell_line>          # re-read a tagged run from its saved weights
sh jobs/crossfit.sh <cell_line>         # pool the complementary-fold models

# 4. merge, figures, page (pod)
sh jobs/summarise.sh
PYTHONPATH=src apy -m scfm_ft.compare_run3          # run 4's re-read of run 3's arms vs run 3
PYTHONPATH=src uv run --no-sync python -m scfm_ft.report.figures --summary results/summary
PYTHONPATH=src uv run --no-sync python -m scfm_ft.report.build --summary results/summary --out "$SILICO_APP_DIR"
PYTHONPATH=src uv run --no-sync python -m scfm_ft.report.contract
```

The job image is `docker.io/library/python:3.11-slim` with `jobs/env_gpu.sh` installing run 2's
pinned stack (torch 2.5.1+cu121, numpy 1.26.4, scgpt 0.2.4 with `--no-deps`). The catalogue image
is not used here because it ships no pip.

## Outputs

All durable outputs live under
`artifact://fundacion-instituto-leloir-553557/experiments/exp_01m0trpxqyf0g8pq29k774xnwm/`.

| prefix | what | how to load |
|---|---|---|
| `cells/` | per-cell-line training matrices and the fold assignment, built from GSE264667 | `np.load`, see `src/scfm_ft/ingest.py` |
| `checkpoints/<line>__<run>/` | 16 fine-tuned models: `model_best.pt` and `model_final.pt` (bare state dicts for `scfm_ft.scgpt_pert.model.build_arm_model`), `gene_table_{init,final}.npz` (resolved table including any learned projection), `train_config.json`, `loss_history.json`, `direct_prediction.npz` | registered as `scfm-run4-ft-<line>-<arm>` |
| `scores/` | per training run: `restricted_`, `contrasts_`, `runs_`, `direct_`, `per_pert_` tables and `summary_` json | `pd.read_parquet` |
| `rescore/<line>/` | the tagged control runs re-read from saved weights after the naming fix, and the whole-screen `crossfit_` pooling | `pd.read_parquet` |
| `followup/` | each model's own predicted response scored against both baselines, plus the training metadata | `pd.read_parquet` |
| `summary.tar.gz` | the merged tables the report reads, unpacked locally to `results/summary` | `tar -xzf` |

`results/run3_comparison.parquet` is not in the artifact store: it is computed pod-side by
`src/scfm_ft/compare_run3.py` against run 3's own `results/summary/runs.parquet`, and holds run 3's
published whole-screen scores beside run 4's re-reading of the same arms.

Models and credentials: no external model API is used. Every number comes from `wanglab/scGPT-human`
at the pinned revision, fine-tuned on this cluster, plus run 3's ESM2-3B gene tables.


Source experiments: run 1 `exp_01m09d588jez38j6trqk30499z` (effect matrices), run 2
`exp_01m0c7ecnnf3fvxskxj5nygwcv` (response-feature extractor), run 3
`exp_01m0emwktvfe892x6ebqyga5ws` (bundle, translator, scoring rule, ESM2 gene tables).
Base model: `wanglab/scGPT-human` at revision `a24c237737a40f3720f75abb555489e9fe753be6`.
Seed 42.

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

## Result

All 10 fine-tuned models (5 arms x 2 cell lines) leave the queried response features inside run 3's
shuffled-label band. The highest reaches 0.045629 against the same model's static gene table at
0.227871 in HepG2 and 0.231806 in Jurkat, so 5 to 43 times below it. Fine-tuning does work on its
own terms — the predicted profile correlates with the measured effect at 0.545115 and 0.344712 — but
that agreement is not perturbation-specific: measured against the model's own unflagged output it
falls to 0.081793 and 0.001952, and only 0.120229 and 0.055233 of a predicted profile's variance
survives subtracting the mean prediction, against 0.744403 and 0.869399 for the measurement itself.
Reading the same checkpoints through the model's own perturbation-flag query instead is noisier and
not a clean null: 2 of those 10 readings, both in Jurkat, sit above the band and above that query's
own pooled shuffled draws. This is a bound at the achieved resolution, not a demonstration that
fine-tuning cannot install the signal; on these two folds co-expression itself is not separated from
the band.

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

# 2. one GPU job per (arm, cell line): fine-tune, extract both query families, score.
#    SCFM_FT_RUN_SUFFIX names the execution and suffixes every output directory.
SCFM_FT_RUN_SUFFIX=_pearson sh jobs/train_arm.sh <cell_line> <arm>

# 3. merge the sixteen per-run archives, pool the complementary folds, write the tables
SCFM_FT_RUN_SUFFIX=_pearson sh jobs/finalise.sh

# 4. figures, page, claims contract (pod)
PYTHONPATH=src apy -m scfm_ft.compare_run3 --summary results/summary
PYTHONPATH=src uv run --no-sync python -m scfm_ft.report.figures --summary results/summary
PYTHONPATH=src uv run --no-sync python -m scfm_ft.report.build --summary results/summary \
    --alt-summary results/summary_masked_mse --out "$SILICO_APP_DIR"
PYTHONPATH=src uv run --no-sync python -m scfm_ft.report.contract --summary results/summary \
    --alt-summary results/summary_masked_mse
```

### Two executions

The sixteen arms were run twice. The reported execution selects checkpoints the way scGPT's
tutorial does: the epoch with the highest mean Pearson correlation, across all genes, between
each held-out perturbation's measured and predicted mean expression profile. Its outputs carry
the `_pearson` suffix. An earlier execution selected on held-out masked MSE and predates the
three faithfulness controls, so it differs in random initialisation as well as in the rule; its
tables are kept at `results/summary_masked_mse` and read as an independent replicate. Every run
of the reported execution also tracks the masked-MSE rule with its own patience counter, which
is the clean within-run comparison.

The job image is `docker.io/library/python:3.11-slim` with `jobs/env_gpu.sh` installing run 2's
pinned stack (torch 2.5.1+cu121, numpy 1.26.4, scgpt 0.2.4 with `--no-deps`). The catalogue image
is not used here because it ships no pip.

## Outputs

All durable outputs live under
`artifact://fundacion-instituto-leloir-553557/experiments/exp_01m0trpxqyf0g8pq29k774xnwm/`.

| prefix | what | how to load |
|---|---|---|
| `cells/` | per-cell-line training matrices and the fold assignment, built from GSE264667 | `np.load`, see `src/scfm_ft/ingest.py` |
| `checkpoints_pearson/<line>__<run>/` | the 16 reported models: `model_best.pt` (the epoch the recipe's rule keeps), `model_best_masked_mse.pt` (the epoch the alternative rule keeps), `model_final.pt` — all bare state dicts for `scfm_ft.scgpt_pert.model.build_arm_model` — plus `gene_table_{init,final}.npz` (resolved table including any learned projection), `train_config.json`, `loss_history.json` (per-epoch scores under both rules and each rule's selected and stopping epoch), `provenance.json`, `direct_prediction.npz`, `prediction_spread.json`, `forward_pass_sample.json`, `query_summary_{init,final}.parquet` | registered as `scfm-run4-ft-<line>-<arm>-pearson` |
| `packs_pearson/pack_<line>__<run>.tar.gz` | one archive per run: that run's score tables and its training metadata, which is what the merge job stages | `tar -xzf` |
| `summary_pearson.tar.gz` | the merged tables the report reads, unpacked locally to `results/summary` | `tar -xzf` |
| `checkpoints/<line>__<run>/` | the earlier masked-MSE execution's 16 models, same directory contract minus the second selected checkpoint | registered as `scfm-run4-ft-<line>-<arm>` |
| `scores/`, `rescore/<line>/`, `followup/`, `summary.tar.gz` | the earlier execution's per-run tables, whole-screen pooling and merged summary, unpacked locally to `results/summary_masked_mse` | `pd.read_parquet`, `tar -xzf` |

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

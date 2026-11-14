# When perturbation queries fail

Code, small result tables and reports for **"When perturbation queries fail: static gene and
protein embeddings outperform simulated scFM perturbation responses"** (Nicolas Yanovsky and
Ariel Chernomoretz).

**The question.** Earlier work extracts regulatory structure from a pretrained single-cell
foundation model (scFM) by changing one gene's input expression and reading how the model's
output changes. Here we ask whether those *perturbation queries* carry information about what
actually happens when that gene is knocked down in a real cell.

**The answer, in short.** We tested scGPT, scPRINT and TranscriptFormer on two genome-scale
CRISPRi Perturb-seq screens (HepG2 and Jurkat). Their simulated responses to perturbation
predict the measured knockdown response barely better than plain co-expression. The same
models' static gene embedding tables do much better. A table built from protein sequence alone
(ESM2) does as well as the single-cell tables. Fine-tuning scGPT on the screen itself does not
change this. Inside the models, the identity of the perturbed gene can be read from the
hidden states, and responses read from the hidden states beat responses read from the output
head. They still fall short of a static gene table.

---

## Contents

1. [What each experiment does](#1-what-each-experiment-does)
2. [Main results](#2-main-results)
3. [Data sources](#3-data-sources)
4. [Models](#4-models)
5. [Reproducing the work](#5-reproducing-the-work)
6. [Pitfalls](#6-pitfalls)
7. [Known gaps](#7-known-gaps)
8. [Citation](#8-citation)

---

## 1. What each experiment does

The work is six runs. Each run is a self-contained directory under `experiments/` with its own
`README.md` (purpose, design, commands, outputs), `manifest.yaml` (headline numbers in
machine-readable form), `src/`, `results/` and `pyproject.toml`. Each run reads the earlier
runs' outputs through files, never a shared process.

| run | directory | question | package |
|---|---|---|---|
| 1 | [`experiment-1-30499z`](experiments/experiment-1-30499z/) | What is the measured knockdown response, and how reproducible is it? Builds effect matrices, noise ceilings and a gene universe shared by all models. | `scfm_pfm` |
| 2 | [`experiment-2-nygwcv`](experiments/experiment-2-nygwcv/) | What does each model predict when you change one gene's input? Extracts zero-out, virtual-value and gradient-trajectory features, plus each model's static gene table. | `scfm_pfm` |
| 3 | [`experiment-3-yga5ws`](experiments/experiment-3-yga5ws/) | Do those features beat co-expression? One fixed translator scores every representation on held-out perturbation groups. Also has the ESM2 protein control and the checkpoint provenance audit. | `scfm_pfm` |
| 4 | [`experiment-4-74xnwm`](experiments/experiment-4-74xnwm/) | Does fine-tuning scGPT on the screen, with its own published recipe, change the answer? | `scfm_ft` |
| 5 | [`experiment-5-hidprb`](experiments/experiment-5-hidprb/) | Does the input intervention reach the hidden states at all? Decodes the perturbed gene's identity from the other genes' hidden states. | `scfm_probe` |
| 6 | [`experiment-6-hidarm`](experiments/experiment-6-hidarm/) | Do responses read from the hidden states, instead of the output head, predict the measured response? | `scfm_hidarm` |

How the runs connect:

```
run 1 (measurement) ──► run 2 (model queries) ──► run 3 (scoring: translator, folds, metric)
                                   │                         │
                                   ├──────────► run 4 (fine-tuned scGPT, scored by run 3)
                                   └──────────► run 5 (hidden-state probe)
                                                run 6 (hidden-state arm: run 5's hooks, run 4's checkpoints, run 3's scoring)
```

Runs 1–3 each copied the previous run's code instead of importing it. Runs 4–6 **import**
earlier runs unmodified, through each run's `sibling.py`, so that they are scored by run 3's
actual rule and queried by run 2's actual model adapters, not by copies that could drift.
Because of this, a change to run 2's or run 3's code also changes runs 4–6.

AIDO.Cell was queried in runs 1–3 but left out of the paper: it was run on a restricted
perturbation scope and cost about 40 times scGPT per forward pass.

## 2. Main results

Every score is a weighted, disattenuated Pearson correlation on held-out perturbation groups,
so **1.0 means the measurement's own replicate reproducibility**, not perfection. All of them
use run 3's **pre-registered** metric (`residual__weighted__disattenuated`). Each run also
reports a fold-centred version as a post-hoc diagnostic. Intervals are bootstraps over
response groups, not over individual perturbations.

| finding | HepG2 | Jurkat | source |
|---|---|---|---|
| Best scFM static gene table | 0.2719 (scGPT) | 0.2371 (TranscriptFormer) | `experiment-3-yga5ws/manifest.yaml` |
| Raw ESM2-3B protein table | 0.2331 | 0.2642 | same |
| Control-cell co-expression | 0.04879 | 0.04462 | same |
| Shuffled-label ceiling | 0.02764 | 0.02351 | same |
| Fine-tuned scGPT, queried response (all 10 checkpoints) | inside the shuffled-label band | inside the band | `experiment-4-74xnwm/manifest.yaml` |
| Perturbed-gene identity from other genes' hidden states, held-out cells, knockdown dose, last layer | 0.457 (scGPT), 0.189 (scPRINT) | 0.409, 0.164 | `experiment-5-hidprb/README.md` |
| Best hidden-state response arm | 0.1440 | 0.1067 | `experiment-6-hidarm/results/summary/` |

Things these numbers do **not** show:

- **Run 4 is a bound, not a null.** On the two folds that fine-tuning never saw, co-expression
  itself is not separated from the band either.
- **Run 5's split-half score is 1.000** at every block layer, in all three models and both cell
  lines. That estimator reads both halves inside one anchor profile. The held-out-cell numbers
  in the table are the ones that test transfer between cells. Decodable identity is also not
  the same thing as a correct response.
- **Run 6 covers scGPT, scPRINT and two fine-tuned scGPT checkpoints.** TranscriptFormer and
  the `k = 16` rung are not yet run. Within the same job, the hidden-state arm reaches about
  53% (HepG2) to 64% (Jurkat) of the static scGPT table, and it beats the static table nowhere.

Each run's `README.md` has the full findings, the caveats that go with them, and what is
still open. Runs 3 and 4 also have a `claims_manifest.json` that maps every number in their
report to a file and a row.

## 3. Data sources

| what | source | used by |
|---|---|---|
| **HepG2 and Jurkat CRISPRi Perturb-seq screens** (Nadig et al. 2025) | GEO [**GSE264667**](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667). Files: [`GSE264667_hepg2_raw_singlecell_01.h5ad`](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_hepg2_raw_singlecell_01.h5ad) (5,614,460,941 bytes, md5 `dfc676e4186b8c8c173ff14e1194cc8a`) and [`GSE264667_jurkat_raw_singlecell_01.h5ad`](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_jurkat_raw_singlecell_01.h5ad) (about 8.8 GB) | run 1, then every run through run 1's outputs |
| Paper describing the screens | Nadig, Replogle et al., *Transcriptome-wide analysis of differential expression in perturbation atlases*, Nature Genetics 57, 1228–1237 (2025), [doi:10.1038/s41588-025-02169-3](https://doi.org/10.1038/s41588-025-02169-3) | — |
| Published differential-expression reference, used as an external cross-check | Hugging Face dataset [`tahoebio/replogle-nadig-de-rhaister`](https://huggingface.co/datasets/tahoebio/replogle-nadig-de-rhaister) (`pdex/all_pdex.parquet`, `cell_eval/all_delta.parquet`) | run 1 (`data/crosscheck.py`) |
| Gene-identifier map | [HGNC complete set](https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt) | run 1 (`genes/universe.py`) |
| Protein sequences for the ESM2 control | [UniProt REST API](https://rest.uniprot.org/uniprotkb/search), reviewed human entries (release 2026_02, 20,431 entries) | run 3 (`protein/uniprot.py`) |

The downloader in `experiment-1-30499z/src/scfm_pfm/data/sources.py` fetches the h5ads and
checks them against GEO's md5 table. Both files are dense float32 in the legacy AnnData layout
and are read in chunks by `data/h5ad_io.py`. Loading a whole file takes about 5 GB (HepG2) or
9 GB (Jurkat) of memory.

## 4. Models

Revisions are pinned in `experiment-2-nygwcv/src/scfm_pfm/models/sources.py` (single-cell
models) and `experiment-3-yga5ws/src/scfm_pfm/protein/sources.py` (ESM2).

| model | source | pinned revision | used in |
|---|---|---|---|
| scGPT (whole-human) | [`wanglab/scGPT-human`](https://huggingface.co/wanglab/scGPT-human) (`best_model.pt`, `vocab.json`, `args.json`) | `a24c237737a40f3720f75abb555489e9fe753be6` | runs 1–6 |
| scPRINT `medium-v1.5` | [`jkobject/scPRINT`](https://huggingface.co/jkobject/scPRINT) (`medium-v1.5.ckpt`) | `8e1ad1a1bc40c1fc2f4f1700fe27923eea9a2bed` | runs 1–3, 5, 6 |
| TranscriptFormer `tf_sapiens` | [`tf_sapiens.tar.gz`](https://czi-transcriptformer.s3.amazonaws.com/weights/tf_sapiens.tar.gz) on CZI's public S3 bucket | unversioned; 1,819,975,447 bytes is the only handle | runs 1–3, 5 |
| AIDO.Cell-100M (now `GB.Cell-100M`) | [`genbio-ai/GB.Cell-100M`](https://huggingface.co/genbio-ai/GB.Cell-100M), with the gene order from the `modelgenerator 0.1.3.post0` sdist | `a983b388e1025860f2c413ecef60e76078253c24` | runs 1–3 only; not in the paper |
| ESM2 650M | [`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | — | run 3 (protein control) |
| ESM2 3B | [`facebook/esm2_t36_3B_UR50D`](https://huggingface.co/facebook/esm2_t36_3B_UR50D) | — | runs 3, 4 |

## 5. Reproducing the work

There are three levels, and each needs much more compute than the one before.

### 5.1 Read the results (no setup)

Everything a claim rests on is committed:

- **Reports.** Runs 1, 3 and 4 each have a self-contained HTML page:
  `experiment-1-30499z/results/data_preparation_report.html`, and
  `results/report/index.html` in runs 3 and 4. Run 2's page is not committed because it
  inlines about 5 MB of plotly.js; rebuild it with the command in §5.2.
- **Tables.** Each run has `results/summary/` (runs 3, 4, 6) or `results/probe/` (run 5), and
  a `manifest.yaml` with the headline numbers.
- **Checks.** Each run's `verify` harness prints every check and the number it checked.
  The captured output is committed under each run's `results/` (for example
  `results/verify_all.txt` in run 1 and `results/verification/verify_all.txt` in runs 2, 5 and 6).

### 5.2 Rebuild reports and run the tests (CPU only, tracked files only)

Runs 1–4 pin their dependencies with `uv`; Python ≥ 3.12. The packages are not installed, so
put `src/` on the path:

```sh
cd experiments/experiment-3-yga5ws
uv sync
PYTHONPATH=src uv run python -m scfm_pfm.report.build --summary results/summary --out /tmp/run3_report
PYTHONPATH=src uv run pytest tests -q
```

Run 2: `python -m scfm_pfm.report.build --bundles results/report/bundles --out OUT`.
Run 4: `python -m scfm_ft.report.build --summary results/summary --alt-summary results/summary_masked_mse --out OUT`.

The report builders import a house figure-style package, `silico_figures`, from the platform
runs 1–4 were executed on. `tools/silico-figures-local/` reconstructs it. Put
`tools/silico-figures-local/src` on `PYTHONPATH` (or install it), and read its `README.md`
before trusting a rebuilt figure. Pin `plotly` to 6.x: plotly 7 emits a plotly.js 4.x CDN
reference, and the original pages used 3.6.0.

Tests exist for runs 2, 3, 5 and 6 (`PYTHONPATH=src pytest tests -q` inside each run). They
are torch-free wherever the logic allows. They include toy models with hand-computable
responses, no-op checks, and null controls that can fail.

### 5.3 Regenerate from raw data (GPU)

Follow the runs in order. Each run's `README.md` has the full command list. Outline:

| run | what to run | compute |
|---|---|---|
| 1 | `data.prepare --cell-line all`, `data.ceilings_grid`, `data.crosscheck`, `genes.universe`, `models.demo --model <m>`, then `verify --all` | CPU with high memory; one short GPU job per model |
| 2 | `models.prefetch`, then `features.extract --model <m> --cell-line <line>` for each model and line, then `verify_features` | GPU. Per (model, line) on one H100: scGPT 6–11 min, scPRINT 12–20 min, TranscriptFormer 77–104 min |
| 3 | `ingest.build` (builds one bundle from runs 1 and 2), then `translate.runner` shards, then `analysis`, `report.contract`, `report.build` | CPU for the sweep; GPU for the ESM2 embeddings |
| 4 | `jobs/ingest.sh <line>`, `SCFM_FT_RUN_SUFFIX=_pearson jobs/train_arm.sh <line> <arm>` for 5 arms × 2 lines, then `jobs/finalise.sh` | one H100 per (arm, line) |
| 5 | `jobs/smoke.sh` first, then `jobs/sweep_mean_control.sh` and `jobs/sweep_individual_cells.sh` | one RTX 3090 (24 GB) |
| 6 | `jobs/smoke.sh`, `stage0.sh`, `fit_basis.sh`, `extract.sh`, `stage1.sh`, `restrict.sh` | one RTX 3090 (24 GB) |

**Where outputs go.** The heavy outputs are not in this repository: effect matrices, feature
blocks, sweep shards, hidden-state blocks and fine-tuned checkpoints. Every run resolves paths
in its `paths.py`, in this order: an explicit environment variable, then a platform variable,
then a local fallback. Set the variable to a data directory:

| run | variables |
|---|---|
| 1 | `SCFM_PFM_DATA_ROOT` (write root), `SCFM_PFM_RAW_DIR` (raw h5ads, ~15 GB), `SCFM_PFM_INPUT_ROOT` |
| 2 | `SCFM_PFM_DATA_ROOT`, `SCFM_PFM_CKPT_DIR` (checkpoint cache), `SCFM_PFM_HANDOFF_ROOT` |
| 3 | `SCFM_PFM3_DATA_ROOT`, `SCFM_PFM3_BUNDLE` (the bundle `ingest.build` writes) |
| 4 | `SCFM_FT_DATA_ROOT`, `SCFM_FT_BUNDLE`, `SCFM_FT_CELLS`, `SCFM_FT_CKPT`, `SCFM_FT_ESM2`, `SCFM_FT_RUN_SUFFIX` |
| 5 | `SCFM_PFM5_DATA_ROOT`, `SCFM_PFM5_BUNDLE`, `SCFM_PFM3_BUNDLE`, `SCFM_PFM5_R2_FEATURES`, `SCFM_PFM_CKPT_DIR` |
| 6 | `SCFM_PFM6_DATA_ROOT`, `SCFM_PFM6_BUNDLE`, `SCFM_PFM3_BUNDLE`, `SCFM_PFM6_RUN5_PROBE`, `SCFM_PFM6_FT_CKPT`, `SCFM_PFM_CKPT_DIR` |

**Where runs 1–4 were executed.** They ran on a hosted compute platform: on-demand H100 jobs
and a private artifact store. Their job scripts and some defaults assume that platform's
environment variables (`SILICO_*`). With those unset, the `paths.py` fallbacks and the
variables above apply. Run 3's `ingest.build` pulls run 1's and run 2's outputs through a
manifest of download links (`--urls`). It was written for presigned HTTPS links, so off the
platform you supply a manifest that points at your own copies. Runs 5 and 6 were written and
run off-platform, on a single workstation.

**Environments for runs 5 and 6.** Each model runs in its own conda environment, because the
three checkpoints were verified against three different torch builds:

| model | python | torch | package |
|---|---|---|---|
| scGPT | 3.10 | 2.9.1+cu128 | `scgpt 0.2.4` |
| TranscriptFormer | 3.11 | 2.5.1+cu124 | `transcriptformer 0.6.1` |
| scPRINT | 3.11 | 2.5.1+cu124 | `scprint 2.3.8`, installed with `--no-deps`, plus `simpler-flash==1.0.7` |

`experiment-5-hidprb/envs/README.md` has the details and the import check to run.

## 6. Pitfalls

None of these raises an error. Each one silently changes a number.

- **scPRINT loading.** Install `scprint==2.3.8` with `--no-deps`: its metadata pins
  `torch==2.2.0` and would downgrade the CUDA build. Set `precpt_gene_emb=None`, because the
  checkpoints point at a stale path that is not needed. Keep the
  `gene_encoder.0.embedding.weight → …embeddings.weight` rename: **skipping it leaves a randomly
  initialised gene embedding, and every response is noise that looks like a result.** Do not
  test the install with `import scprint`; run 2 imports only `scprint.model.model`. Runs 1–6
  all used `medium-v1.5`. Run 2's manifest labels it "large", which is a documentation error.
- **scGPT's `args.json`** embeds the authors' cluster paths. Override `vocab_path` with the
  local `vocab.json`.
- **TranscriptFormer's intervention is a different operation.** Gene counts enter as an
  attention-score modifier, not through the residual stream. Counts are also clipped at a
  training-time 30 CP10K, so the 16 and 64 CP10K grid points both saturate, and its effective
  dose axis is `(0, 1, 4, 30, 30)`. See `experiment-5-hidprb/README.md`.
- **Model keys must not contain `__`.** Run 3's bundle reader splits feature file names on the
  first `__`. A key containing it makes a block unreadable, and its arms drop out of the scored
  set without an error. Use `scfm_ft.config.model_stem` / `run_of_stem`.
- **Build reports from a `git archive` export, not in place.** Run 2's report builder writes
  figures to `<run>/figures` with no flag to redirect it, and would overwrite the committed
  figure bundles.
- **Run 4 needs `SCFM_FT_RUN_SUFFIX=_pearson`.** Without it, run 4 writes into the earlier
  masked-MSE execution's tree. The reported execution selects checkpoints on validation
  Pearson, as scGPT's tutorial does. The masked-MSE execution is kept as an independent
  replicate in `results/summary_masked_mse/`.
- **Directory names are load-bearing.** Build scripts, manifests and `sibling.py` refer to
  them. Do not rename or move a run.
- **Name the metric cell.** The verdict is on the pre-registered
  `residual__weighted__disattenuated`. The fold-centred cell is a post-hoc secondary used to
  order arms. Quote both, and do not let either confirm the other.

## 7. Known gaps

- Run 6: TranscriptFormer, the `k = 16` rung, the random split and a `claims_manifest.json`
  are not yet done (see run 6's README, §0).
- Run 4 ships no `tests/`.
- The READMEs for runs 5 and 6 refer in places to internal working notes (`docs/INDEX.md`,
  `docs/METRICS.md`, `PROGRESS.md`) that are not part of this repository. The facts those
  pointers lead to that you need for reproduction are repeated in §4–§6 above.
- The heavy intermediate outputs of runs 1–4 live in the original platform's private store and
  are not public. Everything needed to regenerate them from the public data and checkpoints is
  in this repository.

## 8. Citation

```bibtex
@misc{yanovsky2026perturbationqueries,
  title  = {When perturbation queries fail: static gene and protein embeddings outperform
            simulated {scFM} perturbation responses},
  author = {Yanovsky, Nicolas and Chernomoretz, Ariel},
  year   = {2026},
  url    = {https://github.com/nyanovsky/scfm-perturbation-failure-modes}
}
```

If you use the screens, please also cite Nadig et al. (2025). If you use the models, please
cite their original papers.

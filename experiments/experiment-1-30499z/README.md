# Run 1 — Nadig effect matrices and the four-model response contract

Two auditable artifacts that every later run in this thread reads through a
**file contract** (never a shared Python process):

1. **Effect matrices** for the Replogle–Nadig HepG2 and Jurkat CRISPRi
   Perturb-seq screens (GEO GSE264667): per-perturbation expression deltas and
   log2 fold changes against batch-matched non-targeting controls, with
   on-target knockdown, differential-expression counts, effect-size strata and
   replicate-split noise ceilings.
2. **Response contract** for four single-cell foundation models (scGPT,
   AIDO.Cell / GB.Cell-100M, scPRINT, TranscriptFormer): a harmonized gene
   universe, an audited identifier map, per-model environment manifests, and a
   demonstrated per-gene response vector plus Jacobian column on one real
   filtered control cell.

## Layout

| path | what |
|---|---|
| `src/scfm_pfm/config.py` | every threshold and constant this run commits to |
| `src/scfm_pfm/data/` | download, filtering funnel, effect estimation, noise ceilings, external cross-check |
| `src/scfm_pfm/genes/` | gene universe + identifier-map audit |
| `src/scfm_pfm/models/` | per-model response interfaces (later runs add extractors against `models/base.py`) |
| `src/scfm_pfm/verify.py` | `--all` asserts and prints the whole done-condition |
| `figures/` | report figures (plot script + plot-ready data + rendered bundle) |
| `results/` | small reviewable outputs and the pod-side mirror of the artifact store |
| `scratch/` | probes and one-off checks |

## How to reproduce

```bash
# data track (CPU, high memory; ~15 GB of raw h5ads land on the job volume)
python3 -m scfm_pfm.data.prepare --cell-line all
python3 -m scfm_pfm.data.crosscheck            # vs the tahoebio reference release

# model track
python3 -m scfm_pfm.genes.universe              # four vocabularies -> shared universe
python3 -m scfm_pfm.models.demo --model scgpt   # one short H100 job per model

# done condition
uv run python -m scfm_pfm.verify --all
```

## Outputs

All durable outputs live under the experiment artifact store, rooted at
`artifact://fundacion-instituto-leloir-553557/experiments/exp_01m09d588jez38j6trqk30499z/` (readable from any fabric). Small reviewable copies are tracked in
`results/`.

| output | location | what it is |
|---|---|---|
| effect matrices | `effects/{hepg2,jurkat}/effects.npz` | delta and log2FC matrices vs batch-matched non-targeting controls, both estimators |
| per-perturbation table | `effects/<cl>/perturbations.parquet` | knockdown, DE counts, effect-size strata, split-half and split-by-batch ceilings |
| gene tables | `effects/<cl>/genes.parquet` | pooled control CP10K and the expressed-gene flag |
| ceiling summary | `effects/<cl>/noise_ceiling_summary.parquet` | per-stratum medians and quartiles for both splits and both metrics |
| filter/QC record | `effects/<cl>/qc.json` | funnel counts, config, retained counts, environment |
| gene universe | `gene_universe/gene_universe.parquet` | 7,444 genes measured in both lines and present in all four vocabularies |
| model vocabularies | `gene_universe/vocabulary_<model>.parquet` | position, token, resolved Ensembl id, resolution route |
| model responses | `models/<model>/response_vector.parquet` | per-gene response over the universe on one real control cell |
| model Jacobians | `models/<model>/jacobian_column.parquet` | d response / d source gene count for that cell |
| model contracts | `models/<model>/contract.json` | output semantics, information flow, ordering, symmetry, substitutions |
| model environments | `models/<model>/environment.json` | pinned packages, image, device, checkpoint revision |
| reference accumulation | `crosscheck/reference_accumulated.npz` | pooled sums from the 29.8 GB tahoebio release |
| cross-check result | `results/crosscheck/{summary.json,*.parquet}` | pooled Spearman 0.9251 over 8,464,986 entries |
| report | `results/data_preparation_report.md` | the data-prep report, every number read from a file above |
| provenance | `results/provenance.json` | sources, checksums, checkpoint revisions, job ids, file contract |

Source data: GEO GSE264667 (HepG2 h5ad md5 `dfc676e4186b8c8c173ff14e1194cc8a`,
5,614,460,941 bytes) and `tahoebio/replogle-nadig-de-rhaister`. Model
checkpoints: `wanglab/scGPT-human` @ `a24c2377`, `genbio-ai/GB.Cell-100M` @
`a983b388` (published as AIDO.Cell-100M), scPRINT `medium-v1.5.ckpt` @
`8e1ad1a1`, TranscriptFormer `tf_sapiens`. One pinned stack
(`envs/model_stack.sh`, torch 2.5.1+cu121) on `python:3.11-slim` + H100; each
model still runs in its own job and records its own resolved package set. No
external model API or LLM was used. Seed 42. Done condition:
`uv run python -m scfm_pfm.verify --all` -> 84/84 checks passed
(2026-08-18).

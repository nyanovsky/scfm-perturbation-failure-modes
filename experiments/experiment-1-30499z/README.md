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

Filled in at completion; see `manifest.yaml` and the provenance manifest under
the experiment artifacts directory.

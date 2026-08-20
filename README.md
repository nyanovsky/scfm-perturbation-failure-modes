# Do single-cell foundation models know how a cell responds to a knockdown?

Three runs against the Replogle-Nadig CRISPRi Perturb-seq screens (GEO GSE264667, HepG2 and
Jurkat). Run 1 builds the measurement, run 2 extracts what four models say about it, run 3 asks
whether any of that beats ordinary co-expression. Each run is a self-contained directory with its
own `README.md`, `manifest.yaml`, `src/`, `results/` and `pyproject.toml`; later runs read earlier
ones through files, never a shared process.

**Directory names are referenced by build scripts and manifests. Do not rename or move them.**

## The three runs

### Run 1 — `experiments/experiment-1-30499z` (`exp_01m09d588jez38j6trqk30499z`)

*What is the measurement, and how reproducible is it?* Effect matrices for both screens
(per-perturbation log2 fold changes against batch-matched non-targeting controls), a harmonized
7444-gene universe with per-model vocabularies, and a demonstrated response vector plus Jacobian
column for each of the four models on one real control cell.

Key outputs: `effects/` and `gene_universe/` in the artifact store; `results/ceiling_grid/`,
`results/crosscheck/` and `results/provenance.json` in the repository.

Report: **committed** at `results/data_preparation_report.html`.

Headline: 1982 HepG2 and 2201 Jurkat perturbations retained, median on-target knockdown 0.8976 and
0.8866, pooled Spearman 0.9251 against the published differential-expression reference, 102 of 102
checks passed. The replicate-split noise ceiling computed here is the scale every later score is
read against, so a correlation of 1.0 is the measurement's own reproducibility, not perfection.

### Run 2 — `experiments/experiment-2-nygwcv` (`exp_01m0c7ecnnf3fvxskxj5nygwcv`)

*What does each model predict when you ask it to change one gene?* Response features for scGPT,
AIDO.Cell, scPRINT and TranscriptFormer across both cell lines, at three query anchors plus a
masked variant for the three bidirectional models: eight feature blocks, each with a zero-out
response, a five-value virtual-value profile and a gradient trajectory, plus the model's static
gene embedding.

Key outputs: `features/` and `features_full_scope/` in the artifact store, indexed by
`feature_index/blocks.json`; verification bundles under `results/report/bundles/`.

Rebuild the report (verified from tracked files only, CPU, no artifact-store fetch):

```sh
cd experiments/experiment-2-nygwcv
uv sync
PYTHONPATH=src uv run python -m scfm_pfm.report.build --bundles results/report/bundles --out OUT
```

Headline: 328 of 352 block-level checks passed. Masking a read-out gene is not the same as giving
it an uninformative value, in any of the six bidirectional blocks. Raw feature scales differ
1,900-fold between models, so downstream work must use the standardized arrays.

### Run 3 — `experiments/experiment-3-yga5ws` (`exp_01m0emwktvfe892x6ebqyga5ws`)

*Do those response features beat co-expression at predicting the perturbation-specific response?*
One fixed translator maps a gene-pair feature to the residual log2 fold change of a read-out gene;
only the input feature changes between arms. Held-out perturbations on grouped and random splits,
both cell lines, plus HepG2-to-Jurkat transfer, a control-cell overlap test, a query-anchor sweep
and a reproduction of the source paper's own BEELINE baseline.

Key outputs: `bundle/`, `sweep/`, `transfer2/` and the other job shards in the artifact store;
merged tables in `results/summary/`; `report_summary.json` and `claims_manifest.json`.

Rebuild the report (verified from tracked files only, CPU, no artifact-store fetch):

```sh
cd experiments/experiment-3-yga5ws
uv sync
PYTHONPATH=src uv run python -m scfm_pfm.report.build --summary results/summary --out OUT
```

Headline: the static gene embedding beats co-expression in both lines — 0.1089 against 0.005176 in
HepG2 (shuffled-label ceiling 0.02860) and 0.1726 against 0.03364 in Jurkat (ceiling 0.02040) — and
the harvested response features add nothing on top of it. Three caveats travel with those numbers.
The scoring cell is post hoc: the pre-registered cell fails its own shuffled-label control, so the
verdict is read from the fold-centred cell the pre-registration also defined. The second
pre-registered axis, perturbation discrimination, is version-dependent and inconclusive. Absolute
scores stay under 0.18 on a scale where 1.0 is the measurement's reproducibility limit.

## What the thread found, and what is still open

A single-cell foundation model does carry usable information about which genes respond to a
knockdown, but it sits in the static gene-identity embedding rather than in the model's simulated
reaction to a perturbation. Querying the model — the mechanism the source paper proposes — adds
nothing beyond the null band on this task.

**Open: whether "foundation model" is the right word for the winning arms.** scPRINT's and
TranscriptFormer's gene tables are ESM2-derived, so their advantage may belong to a protein
language model rather than to single-cell pretraining. scGPT's table is learned from expression
data and is the only such arm above the noise band. A raw ESM2 arm through the identical translator
and a per-model audit of whether each gene table was updated during pretraining are in progress in
run 3; until they land, read the headline as being about the models' gene embeddings, not about
single-cell pretraining.

## Rebuilding, in general

Each run pins its own dependencies in its own `pyproject.toml` and `uv.lock`; run `uv sync` inside
the run directory. The house figure-style package is a local path dependency supplied by the
platform, so a clone taken off-platform needs that package made available before the report
builders will run. Regenerating the underlying results — not the reports — needs GPU jobs and the
artifact store; each run's `README.md` gives those commands.

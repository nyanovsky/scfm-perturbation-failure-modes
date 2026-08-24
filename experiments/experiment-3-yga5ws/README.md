# Run 3 — do scFM response features beat co-expression at predicting perturbation response?

One fixed translator maps a **gene-pair feature** to the **residual** (perturbation-specific)
log2 fold change of a read-out gene under CRISPRi knockdown of a perturbed gene. Only the input
feature varies across arms. The arms that matter are the four single-cell foundation models'
response features against plain statistical co-expression computed from the same control cells,
the control the UGRN paper never ran.

**What it found.** The response features add nothing to the model's static gene embedding; the
embedding is what beats co-expression, on the pre-registered scoring rule and with every interval
resampling the response groups the split holds out together (0.2719 for scGPT's embedding in HepG2
against 0.04879 for co-expression; paired +0.1846 [0.1501, 0.2408]). Reading the four released
checkpoints then shows that two of the four gene-identity tables are protein-derived rather than
learned from cells, and a table built directly from UniProt protein sequence with a public ESM2
checkpoint, run through the same translator and folds, beats co-expression in both lines and ties
the single-cell arms in seven of eight paired comparisons, beating scGPT's in Jurkat. On genes
never perturbed in the training screen, only the protein-derived tables keep a score clear of
zero. Concatenating scGPT's table with the raw ESM2-3B table into one arm does not beat the better
of the two parts, and that contrast was underpowered for the effect on offer: in all four cells
the difference against the better part is smaller than what the comparison resolves (+0.02233
against 0.1366, +0.02134 against 0.04688, +0.004363 against 0.02620, +0.01529 against 0.03364).
The informative half is the agreement statistic read against its calibration: the two parts agree
on which perturbations they score well at Spearman 0.2503 and 0.2394, against 0.6816 and 0.6727
for two sizes of one protein model and 0.2843 and 0.2295 for scGPT against co-expression. scGPT's
table agrees with a protein model about as weakly as with co-expression and far below the
same-family reference, which points toward two tables holding different information rather than
two routes to one quantity; whether that difference converts into a gain is open, not null. `manifest.yaml` carries the
numbers; the published report carries the argument.

The fold-centred version of the score, which removes the held-out fold's own per-gene mean, is
reported beside the pre-registered one everywhere. It uses held-out outcomes and was written after
seeing the shuffled-label draws, so it is a diagnostic and never a substitute for the verdict.

## Layout

| path | what |
|---|---|
| `src/scfm_pfm/config.py` | the pre-registration: scoring rule, splits, translator, arms |
| `src/scfm_pfm/metrics.py` | reliability weights, residual ceiling, weighted correlation, the group (cluster) bootstrap every interval uses, power |
| `src/scfm_pfm/groups.py` | recovers the response groups from the bundle's effect matrix, so any scored table can be re-bootstrapped over groups |
| `src/scfm_pfm/fetch.py` | pulls run 1 / run 2 artifacts into a job from a signed-link manifest |
| `src/scfm_pfm/ingest/` | the one bundle every later job reads: targets, co-expression, features |
| `src/scfm_pfm/translate/` | arms, splits, the translator, the sweep runner |
| `src/scfm_pfm/protein/` | the protein-model control: checkpoint provenance audit, gene-to-protein mapping, ESM2 tables, table similarity |
| `src/scfm_pfm/report/` | figures and the report |
| `results/handoff/` | small tables that travel with the worktree into a job |

## How to reproduce

```sh
# 1. mint signed links for the run 1 / run 2 objects (pod-side, see note below),
#    then build the bundle in one CPU job
python -m scfm_pfm.ingest.build --urls results/handoff/urls/artifact_urls.json --work ./_work

# 2. the translator sweep (GPU job, mounts the ingest job's volume)
python -m scfm_pfm.translate.runner --shard <k> --n-shards <n>

# 3. the protein-model control (see `src/scfm_pfm/protein/`)
python -m scfm_pfm.protein.audit --cache _ckpt --out results/provenance   # CPU
python -m scfm_pfm.protein.uniprot --out results/protein                  # CPU, network
python -m scfm_pfm.protein.esm_embed --genes results/protein --out results/protein  # GPU
python -m scfm_pfm.protein.install_arm --tables results/protein --bundle <bundle>
python -m scfm_pfm.translate.runner --bundle <bundle> --arms esm2_650m_emb,esm2_3b_emb,...

# 3b. the two-table arm: the same installed ESM2-3B table, concatenated with scGPT's
python -m scfm_pfm.translate.runner --bundle <bundle> \
  --arms scgpt_esm2_3b_emb,null_randgene_d6144 --scopes full --splits grouped,random

# 4. scoring, figures, report
python -m scfm_pfm.report.build
```

### Reading run 1 and run 2 from inside a job

A fresh on-demand job has no artifact-store credential, and run 2's job volumes have passed their
retention, so the feature blocks cannot be mounted. The supported path is a short-lived signed
download link per object, minted pod-side with the artifact-store tool and written to
`results/handoff/urls/artifact_urls.json`, which travels in the untracked part of the worktree
snapshot. Both that file and the helper that composes it embed live one-hour AWS session
credentials and are deliberately **not** committed; the durable `artifact://` refs they point at
are listed in `config.py` (`RUN1`, `RUN2`, `BLOCK_PREFIX`). The ingest job consolidates everything
into one bundle, so every later job reads the bundle rather than the store.

## Outputs

Durable outputs live under this experiment's artifact root
`artifact://fundacion-instituto-leloir-553557/experiments/exp_01m0emwktvfe892x6ebqyga5ws/`,
written by on-demand jobs whose output flow-back uploads `$SILICO_EXPERIMENT_ARTIFACTS_DIR`.
Small reviewable copies live in this worktree.

| output | location | what it is |
|---|---|---|
| bundle | `bundle/<line>/` (store) | targets (effect + both half-estimates), co-expression and node features, per-model response features, `reference.json` |
| main sweep | `sweep/` (store), 8 shards | 374 runs: both lines x full and 250-gene common scopes x grouped and random splits, every arm, both translators |
| common + random | `sweep_common_random/` (store) | the common-scope random split, added after review |
| node controls | `sweep_node/`, `sweep_node_common/` (store) | additive node, free node, shuffled-gene node |
| co-expression half check | `leakcheck/` (store) | `coexp_check` on the other control half |
| transfer | `transfer2/`, `transfer_unseen/` (store) | cross-line transfer, three reference modes, full / common / unseen-gene scopes |
| control-split targets | `ctrl_split_targets/` (store) | effect recomputed against each control half and against all control cells, plus `recompute_report.json` |
| control-overlap scoring | `alt_target_ctrl0/`, `alt_target_ctrl1/`, `alt_target_ctrlall/` (store) | the key arms rescored against each of those targets |
| anchor sensitivity | `sensitivity/` (store) | HepG2 response arms at the uniform-virtual, individual-cells and masked anchors, with each model's mean-control arm re-run as the in-job reference |
| gene-table provenance | `provenance/audit.json`, `provenance/audit_tables.json` (store), same under `results/provenance/` (worktree) | per model, which tensor is the gene-identity table and whether pretraining updated it, read from the released checkpoints |
| protein tables | `protein/` (store): `esm2_650m.npz`, `esm2_3b.npz`, `gene_to_protein.parquet`, `identity.json`, and the three reports; `results/protein/` (worktree) keeps the small ones | ESM2 embeddings of each read-out gene's canonical human protein, the mapping that produced them, and how close every gene table is to them |
| protein arms | `protein_sweep/<line>/`, `protein_shuffle/<line>/`, `protein_transfer/` (store) | the ESM2 arms and their width-matched null over the same grid as every other arm, their shuffled-label draws, and both transfer directions |
| protein in-job reference | `protein_reference/<line>/`, `protein_transfer_reference/` (store) | main-sweep arms re-run inside the protein jobs; kept out of the merged tables so no scored row is overwritten |
| two-table arm | `concat_sweep/`, `concat_smoke/` (store) | scGPT's table concatenated with the raw ESM2-3B table, 6,144 columns, over both lines and both splits, with its width-matched random-vector null |
| two-table in-job reference | `concat_reference/` (store) | scGPT, ESM2-3B and co-expression re-run inside the same job; kept out of the merged tables for the same reason |
| no-op query record | `results/handoff/no_op_queries_per_gene.csv`, `results/verification/no_op_correction.json` (worktree) | which (perturbed gene, value) queries were no-ops and in how many cells, derived from run 2's per-query records, plus the five checks behind the per-query exclusion |
| BEELINE arm (dropped) | `beeline/beeline_reproduction.json` (store), `results/beeline_reproduction.json` (worktree) | the source paper's scGPT embedding baseline re-fitted on its own benchmark. Dropped from the evidence: it scored 0.6724 against 0.6247 reported, so the setting is not established and neither it nor the controls run inside it (a permuted-embedding control, an unseen-gene slice) support a claim in the report. Kept for audit only |
| discrimination versions | `results/discrimination_versions.json`, `results/discrimination/` (worktree) | the same saved predictions rescored under cell-eval 0.5.18, 0.5.21 and 0.8.2 |
| merged tables | `results/summary/` (worktree) | `runs.parquet`, `contrasts.parquet`, `grid.parquet`, `transfer_runs.parquet`, `ordering_agreement.parquet`, `control_cell_overlap.parquet`, `null_band.parquet`, `headline.json` |
| target recompute report | `results/recompute_report.json` (worktree) | agreement of each recomputed target with run 1's effect matrix, reliability weights, cells kept |
| report contract | `report_summary.json`, `claims_manifest.json` (worktree) | generated from the tables; every claim names its source file and row |
| figures | `figures/<name>/` (worktree) | one bundle per figure: interactive HTML, `data.json`, alt text |
| report | published Results page for this experiment | built by `scfm_pfm.report.build` from `results/summary` |

`results/summary/per_perturbation.parquet` and its transfer twin are regenerated by the
analysis command from the shards and are deliberately not committed.

### Reproducing the analysis and the report

```sh
# shards pulled from the store into scratch/dl/<wave>/, later waves winning on overlap
PYTHONPATH=src apy -m scfm_pfm.analysis \
  --sweep scratch/dl/leakcheck,scratch/dl/sweep,scratch/dl/sweep_common_random,\
scratch/dl/sweep_node,scratch/dl/sweep_node_common,scratch/dl/alt_target_ctrl0,\
scratch/dl/alt_target_ctrl1,scratch/dl/alt_target_ctrlall,\
scratch/dl/protein_sweep/hepg2,scratch/dl/protein_sweep/jurkat,\
scratch/dl/protein_shuffle/hepg2,scratch/dl/protein_shuffle/jurkat,scratch/dl/concat_sweep \
  --transfer-sweep scratch/dl/transfer2,scratch/dl/transfer_unseen,scratch/dl/protein_transfer \
  --sensitivity-sweep scratch/dl/sensitivity \
  --protein-reference scratch/dl/protein_reference/hepg2,scratch/dl/protein_reference/jurkat \
  --protein-transfer-reference scratch/dl/protein_transfer_reference \
  --concat-reference scratch/dl/concat_reference \
  --bundle-root scratch/dl/bundle \
  --out results/summary
PYTHONPATH=src uv run --no-sync python -m scfm_pfm.report.contract --summary results/summary
PYTHONPATH=src uv run --no-sync python -m scfm_pfm.report.build \
  --summary results/summary --out "$SILICO_APP_DIR"
```

The no-op query table is derived once on the pod from run 2's per-query records and committed,
because the aggregate count per value cannot say which queries it refers to:

```sh
# the three run 2 query_summary.parquet files are pulled with signed links first
PYTHONPATH=src apy -m scfm_pfm.ingest.no_op \
  --aggregate results/handoff/no_op_queries.csv \
  --query-summary scgpt:hepg2:individual_cells:scratch/noop/qs_scgpt_hepg2_ic.parquet \
  --query-summary scprint:hepg2:individual_cells:scratch/noop/qs_scprint_hepg2_ic.parquet \
  --query-summary transcriptformer:hepg2:individual_cells:scratch/noop/qs_transcriptformer_hepg2_ic.parquet \
  --out results/handoff/no_op_queries_per_gene.csv
# the checks behind the correction, against run 2's own arrays
PYTHONPATH=src apy -m scfm_pfm.ingest.no_op_check --block <model>:<line>:<anchor>:<vvp>:<qs> \
  --out results/verification/no_op_correction.json
```

The discrimination axis is computed with Arc Institute's `cell-eval`. The pre-registration pinned
0.5.19, which was never released (PyPI publishes 0.1.0 then 0.5.33 onward; the upstream tags jump
0.5.18 to 0.5.21), so the jobs resolved 0.8.2 and `scratch/disc_version_check.py` rescores the
saved predictions under 0.5.18, 0.5.21 and 0.8.2 to show what that choice costs.

`scratch/verify_claims.py` re-reads each entry of `claims_manifest.json` from the file and key it
names and reports any disagreement. All 389 re-readable keys resolve and match; the remaining 11
claims are differences between two rows that are themselves claimed, and their key states the
formula instead of a locator.

No external model provider is used anywhere in this run: every number comes from the four
open single-cell foundation models' cached features and from code in this worktree.

Source experiments: run 1 `exp_01m09d588jez38j6trqk30499z` (effect matrices, split-half vectors),
run 2 `exp_01m0c7ecnnf3fvxskxj5nygwcv` (response features for four models at three anchors).
Raw control cells are rebuilt from GEO GSE264667 because run 1 persisted only pooled control means.
Seed 42.

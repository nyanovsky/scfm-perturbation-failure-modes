# Run 3 — do scFM response features beat co-expression at predicting perturbation response?

One fixed translator maps a **gene-pair feature** to the **residual** (perturbation-specific)
log2 fold change of a read-out gene under CRISPRi knockdown of a perturbed gene. Only the input
feature varies across arms. The arms that matter are the four single-cell foundation models'
response features against plain statistical co-expression computed from the same control cells —
the control the UGRN paper never ran.

## Layout

| path | what |
|---|---|
| `src/scfm_pfm/config.py` | the pre-registration: scoring rule, splits, translator, arms |
| `src/scfm_pfm/metrics.py` | reliability weights, residual ceiling, weighted correlation, bootstrap, power |
| `src/scfm_pfm/fetch.py` | pulls run 1 / run 2 artifacts into a job from a signed-link manifest |
| `src/scfm_pfm/ingest/` | the one bundle every later job reads: targets, co-expression, features |
| `src/scfm_pfm/translate/` | arms, splits, the translator, the sweep runner |
| `src/scfm_pfm/report/` | figures and the report |
| `results/handoff/` | small tables that travel with the worktree into a job |

## How to reproduce

```sh
# 1. mint signed links for the run 1 / run 2 objects (pod-side, see note below),
#    then build the bundle in one CPU job
python -m scfm_pfm.ingest.build --urls results/handoff/urls/artifact_urls.json --work ./_work

# 2. the translator sweep (GPU job, mounts the ingest job's volume)
python -m scfm_pfm.translate.runner --shard <k> --n-shards <n>

# 3. scoring, figures, report
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

Durable outputs live under this experiment's artifact root:
`artifact://fundacion-instituto-leloir-553557/experiments/exp_01m0emwktvfe892x6ebqyga5ws/`.

| output | location | what it is |
|---|---|---|
| bundle | `bundle/<line>/` | targets (effect + both half-estimates), co-expression and node features, per-model response features, `reference.json` |
| results | `results/` | per-arm scores, the correlation grid, discrimination, bootstrap intervals |
| report | `report/` | the HTML report and its figure bundles |

Source experiments: run 1 `exp_01m09d588jez38j6trqk30499z` (effect matrices, split-half vectors),
run 2 `exp_01m0c7ecnnf3fvxskxj5nygwcv` (response features for four models at three anchors).
Raw control cells are rebuilt from GEO GSE264667 because run 1 persisted only pooled control means.
Seed 42.

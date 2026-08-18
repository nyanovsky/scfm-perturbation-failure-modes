# Replogle-Nadig effect matrices and the four-model response contract

This run produces two artifacts that every later run in this thread reads through a file contract: measured perturbation effects for two CRISPRi Perturb-seq cell lines, and a written, tested description of what each of four single-cell foundation models emits when asked about a cell.

_Generated 2026-08-18 05:57 UTC from the artifacts named below._

## The measured effects

Source: GEO GSE264667, the two deposited raw single-cell matrices. Both were downloaded with their md5 recorded and read directly as HDF5 row chunks rather than through anndata, because `X` is a dense float32 matrix and streaming it keeps peak memory at one chunk plus accumulators.

| | HepG2 | Jurkat |
|---|---|---|
| cells x measured genes | 145,473 x 9,624 | 262,956 x 8,882 |
| file size | 5.614 GB | 9.366 GB |
| md5 | `dfc676e4186b8c8c173ff14e1194cc8a` | `dae9fab2a8b0430099a76ce60d0c5405` |
| sequencing batches | 56 | 55 |
| cells after QC | 135,368 | 259,335 |
| non-targeting control cells | 4,916 | 12,004 |
| retained perturbations | 1,982 | 2,201 |
| expressed genes (CP10K >= 0.5) | 3,119 | 3,469 |
| median on-target knockdown | 0.8976 | 0.8866 |
| median DE genes per perturbation | 36 | 61 |
| processing runtime | 169 s | 279 s |

### What the filters remove

The deposited matrices are already partly quality-controlled, so per-cell filters remove very little: in HepG2 only the mitochondrial-fraction filter removes anything (3,137 cells), and in Jurkat only the minimum-count filter does (631 cells). Almost all of the attrition is the requirement that a perturbation have at least 25 QC-passing cells across at least 2 batches, which takes 2,393 perturbations with any passing cell down to 1,982 in HepG2 and 2,201 in Jurkat.

![Filter funnel](figures/filter_funnel/filter_funnel.html)

### The control contrast

The median (perturbation, batch) cell count in these libraries is one to two cells, so a strictly within-batch estimator would discard nearly everything. The primary estimate instead references each perturbation against a batch-matched control profile: control means per `gem_group`, weighted by that perturbation's own per-batch cell counts. The batch-pooled estimate is kept beside it as a secondary column, and the two are compared directly in the cross-check section.

Differential expression is a Welch t-test on log1p(CP10K) against pooled non-targeting controls, Benjamini-Hochberg controlled at 0.05 across expressed genes.

### On-target knockdown

Median knockdown is 0.8976 in HepG2 and 0.8866 in Jurkat. 228 HepG2 and 306 Jurkat perturbations fall below the 0.30 detection floor, and 187 and 223 target genes are not in the measured panel at all, so no knockdown can be computed for them. They are flagged in `perturbations.parquet` rather than dropped: whether to use them is a decision for the run that consumes them.

![Knockdown distribution](figures/knockdown_distribution/knockdown_distribution.html)

### Effect size and differential expression

| stratum | HepG2 n | HepG2 median DE | Jurkat n | Jurkat median DE |
|---|---|---|---|---|
| S1_lowest | 496 | 3 | 551 | 4 |
| S2 | 495 | 5 | 550 | 31 |
| S3 | 496 | 79 | 550 | 130 |
| S4_highest | 495 | 984 | 550 | 448 |

![Effect size against DE count](figures/effect_size_vs_de/effect_size_vs_de.html)

### The noise ceiling

A model asked to predict these effects cannot do better than the data predicts itself, so each perturbation is split in half and one half is scored against the other. Two split forms are reported. Splitting cells references each half against a disjoint half of the control cells, so shared control noise cannot inflate the agreement; splitting by batch alternates each perturbation's batches so the two sides use different control cells by construction.

| stratum | HepG2 cells | HepG2 batches | Jurkat cells | Jurkat batches |
|---|---|---|---|---|
| S1_lowest | 0.0961 | 0.0934 | 0.1041 | 0.0992 |
| S2 | 0.1247 | 0.1184 | 0.1906 | 0.1930 |
| S3 | 0.2980 | 0.2910 | 0.2933 | 0.2946 |
| S4_highest | 0.7031 | 0.7044 | 0.4347 | 0.4305 |

Agreement rises monotonically with effect size in both cell lines and both split forms, and the two split forms agree closely with each other, which is what a working ceiling estimate should look like: the smallest effects are barely reproducible, the largest are.

![Noise ceilings](figures/noise_ceilings/noise_ceilings.html)

## Cross-check against the published reference

The reference release `tahoebio/replogle-nadig-de-rhaister` holds per-batch fold changes computed from the same series. Pooling its batches the same way we pool ours, our per-gene log2 fold changes agree with it at Spearman **0.9251** over 8,464,986 (cell line, perturbation, gene) entries covering 2,877 shared perturbations; the gate for this run was 0.70. Median agreement within a single perturbation is 0.9449, and it rises with effect size:

| stratum | perturbations | median Spearman vs reference |
|---|---|---|
| S1_lowest | 663 | 0.9191 |
| S2 | 741 | 0.9300 |
| S3 | 713 | 0.9519 |
| S4_highest | 760 | 0.9772 |

The same module compares our two estimators against each other. Batch-matched and batch-pooled log2 fold changes agree at median Pearson 0.9963, and the deltas at 0.9971. Where they disagree, batch composition is doing the work, which is why the batch-matched estimate is the primary one.

![Reference cross-check](figures/reference_crosscheck/reference_crosscheck.html)

## The four-model response contract

The shared gene universe holds **7,444 genes**: measured in both cell lines (7,632 genes) and present in all four model vocabularies. Identifiers are resolved to Ensembl gene ids through the assay's own var table first, then HGNC approved symbols, then HGNC aliases and previous symbols, and the route taken is recorded per token. Round-tripping the 10,874 genes named in the series gives 1 Ensembl->symbol->Ensembl and 0 symbol->Ensembl->symbol failures.

| model | vocabulary | identifier | fixed input order | covers HepG2 measured | covers Jurkat measured |
|---|---|---|---|---|---|
| scgpt | 60,697 | symbol | no | 9,522 / 9,624 | 8,780 / 8,882 |
| aido_cell | 19,264 | symbol | yes | 9,296 / 9,624 | 8,548 / 8,882 |
| scprint | 44,756 | ensembl | yes | 9,297 / 9,624 | 8,546 / 8,882 |
| transcriptformer | 23,830 | ensembl | no | 9,297 / 9,624 | 8,549 / 8,882 |

![Model gene coverage](figures/model_gene_coverage/model_gene_coverage.html)

### What each model emits

| model | per-gene value | information flow | ordering matters | Jacobian |
|---|---|---|---|---|
| scgpt | predicted binned expression value for that gene in this cell, on the 51-bin training scale (not counts, not CP10K) | bidirectional full attention over gene tokens; no positional encoding over genes | no | central_finite_difference(step=0.5)_on_smooth_binning |
| aido_cell | one scalar per gene position (the model's masked-language-model regression target, i.e. predicted high-read-depth expression on the log1p CP10K scale); the last two positions are read-depth tokens and are dropped | bidirectional encoder (18 layers) over the full fixed gene order | no | central_finite_difference(step=1.97) |
| scprint | predicted share of the cell's counts for that gene: the expression decoder ends in a softmax across the supplied gene positions, conditioned on the requested read depth | bidirectional flash transformer over the supplied gene set plus cell-embedding tokens | no | forward_mode_autodiff_jvp |
| transcriptformer | predicted share of the cell's total counts assigned to that gene, conditioned on the gene's own identity and on the identity and count of every gene earlier in the ordering | autoregressive, causal mask over the gene sequence | yes | central_finite_difference(step=1)_source_first_ordering |

The per-gene values are on four different scales and are not comparable across models without a transformation the consuming run has to choose deliberately. Two of them (scPRINT, TranscriptFormer) end in a softmax across the genes supplied in the same forward pass, so their values are shares of a cell's counts and depend on which genes were supplied together.

![Model responses on one cell](figures/model_responses_on_one_cell/model_responses_on_one_cell.html)

### TranscriptFormer is not symmetric, and the measurement says so

TranscriptFormer is autoregressive with a causal mask, so a gene responds to another only when that other gene precedes it in the sequence. Reading the same target gene twice from the same cell with 254 context genes between them: raising the source gene's count from 197 to 296 changes the target's predicted share by +1.565e-03 (on a base of 1.271e-02) when the source is ahead of it, and by +0.0 exactly when the source is behind it. A run that treats a TranscriptFormer response matrix as symmetric is reading half of it from a masked-out region.

### Is the response matrix symmetric? Measured, not assumed

A run that reads one column of the response matrix and treats the matrix as symmetric is making a claim no model documents. One column cannot answer it, so each model was asked for a second column at the gene its source gene moves most, and the two cross entries were compared. A verdict is only recorded when the Jacobian estimate itself is stable across step sizes (Pearson at least 0.99 and relative L2 difference at most 0.5 between two steps); otherwise step sensitivity and genuine asymmetry are not separable and the contract says so.

| model | second column | d target / d source | d source / d target | ratio | verdict |
|---|---|---|---|---|---|
| scgpt | central_finite_difference(step=0.4831)_on_smooth_binning | +2.8408e-02 | -2.0489e-02 | -0.7212 | asymmetric |
| aido_cell | central_finite_difference(step=0.02) | -7.8509e-04 | -4.5300e-04 | +0.577 | not_determined |
| scprint | forward_mode_autodiff_jvp | +4.9628e-03 | +6.8167e-03 | +1.374 | asymmetric |
| transcriptformer | central_finite_difference(step=1)_source_first_ordering | +5.7457e-06 | -6.1287e-05 | -10.67 | not_determined |

Where the measurement is decisive it comes out asymmetric: scgpt, scprint. The response matrix is directional, so a consuming run has to say which index is the perturbed gene and which is the read-out gene, and cannot fill the other half by transposition.

For aido_cell the finite-difference Jacobian is itself step-dependent at this cell, so the contract records symmetry as not determined rather than guessing. Anything in a later run that depends on the magnitude (not just the sign or ranking) of these Jacobians needs a step-size study first.

For transcriptformer the cross-entry probe is also step-limited, but symmetry is already settled by the architecture rather than by this measurement: a causal mask makes half the matrix unreachable, as the ordering probe below shows directly.

| model | Jacobian method | cross-check | Pearson | relative L2 gap | column L2 |
|---|---|---|---|---|---|
| scgpt | central_finite_difference(step=0.5)_on_smooth_binning | second central finite difference at step=1 | 0.9993 | 0.1616 | 0.1855 |
| aido_cell | central_finite_difference(step=1.97) | second central finite difference at step=1 | 0.9991 | 1.379 | 0.0107 |
| scprint | forward_mode_autodiff_jvp | central_finite_difference(step=1.73) | 1 | 2.629e-04 | 0.7169 |
| transcriptformer | central_finite_difference(step=1)_source_first_ordering | second central finite difference at step=2 | 0.8714 | 0.9803 | 1.289e-05 |

Only scPRINT's column is an exact derivative (forward-mode autodiff, agreeing with a finite difference to a relative L2 difference of 2.629e-04). The other three are finite differences over integer counts, which is the honest description of what a discretized or masked input allows: their direction is reproducible but their magnitude depends on the step, and TranscriptFormer's per-gene sensitivities on this cell are close to numerical noise (column L2 1.289e-05).

### Substitutions

Where a published inference path could not be run as shipped, the substitution is recorded in the model's `contract.json` rather than hidden:

- **scgpt**:
  - flash-attention not installed (the published checkpoint was trained with scGPT's FlashMHA layers, whose API exists only in flash-attn 1.x, which cannot build against this torch); used scGPT's own fallback torch.nn.TransformerEncoder path (use_fast_transformer=False), which changes the attention kernel and therefore the exact numerics
  - remapped 24 fused attention tensors from the checkpoint's FlashMHA layout (self_attn.Wqkv) onto torch's packed projection (self_attn.in_proj); both hold [query; key; value] over the same head partition, so the attention computed is the same function of the same weights. Verified numerically at load time (see attention_remap_check)
  - imported scgpt.model.model by file path to avoid executing scgpt/__init__ (scvi-tools / torchtext / scib); model source unmodified
  - forward-mode AD unavailable (OutOfMemoryError: CUDA out of memory. Tried to allocate 2.70 GiB. GPU 0 has a total capacity of 79.18 GiB of which 2.31 MiB is free. Process 1 has 79.17 GiB memory in use. Of the allocated memory 76.32 GiB is allocated by PyTorch, and 2.19 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)); used central finite differences instead
- **aido_cell**:
  - replaced BertSelfFlashAttention.forward with a shape-correct SDPA implementation: the shipped non-flash fallback softmaxes across heads instead of across the sequence, and the flash-attention path is fp16/bf16 only with no forward-mode derivative. Weights and attention mathematics are unchanged; the kernel and dtype are not.
  - forward-mode AD unavailable (NotImplementedError); used central finite differences
- **scprint**:
  - simpler_flash.FlashSelfAttention.forward delegated to the package's own pure-torch SelfAttention: the fused kernel asserts fp16/bf16 and has no forward-mode derivative, so the response and its Jacobian would not be computable in float32. Same weights, same attention, different kernel (upstream forward kept at FlashSelfAttention.forward)
  - stubbed scprint.model.utils / scprint.utils / scprint.tasks.* at import time to avoid bionty+lamindb+scdataloader (a configured lamindb instance is not available and is unnecessary for a forward pass); model source unmodified
- **transcriptformer**:
  - compile_block_mask disabled (eager flex-attention block mask) so that a single-cell forward pass does not pay a torch.compile warm-up

### How the four stacks were actually run

The plan called for one pinned environment per model, on the assumption that the four dependency sets are mutually unsatisfiable. They are not quite: pinning torch first and installing the four model packages with `--no-deps` (each would otherwise drag torch back to the version it was built against) gives one stack, `envs/model_stack.sh`, that runs all four. Every model here ran on torch 2.5.1+cu121 in the same image, and each still records its own resolved package set in `environment.json`, so a later run can tell whether a result came from the same stack or a different one. Each model is still verified independently and still runs in its own job; only the installation is shared.

| model | image | device | checkpoint load | forward |
|---|---|---|---|---|
| scgpt | `docker.io/library/python:3.11-slim` | cuda | 0.85 s | 0.463 s |
| aido_cell | `docker.io/library/python:3.11-slim` | cuda | 2.02 s | 1.145 s |
| scprint | `docker.io/library/python:3.11-slim` | cuda | 3.75 s | 0.336 s |
| transcriptformer | `docker.io/library/python:3.11-slim` | cuda | 7.41 s | 0.891 s |

## What a later run reads

```
effects/<cell_line>/effects.npz              delta and log2FC matrices, both estimators
effects/<cell_line>/perturbations.parquet    per-perturbation metadata, knockdown, DE, strata, ceilings
effects/<cell_line>/genes.parquet            gene index, control level, expressed flag
effects/<cell_line>/noise_ceiling_summary.parquet
effects/<cell_line>/qc.json                  provenance, filter funnel, config, environment
crosscheck/summary.json                      agreement with the published reference
gene_universe/gene_universe.parquet          one row per shared gene, with each model's token position
gene_universe/vocabulary_<model>.parquet     every vocabulary token, its gene id, and how it resolved
gene_universe/coverage.json                  coverage and the identifier audit
models/<model>/response_vector.parquet       per-gene response on the demonstration cell
models/<model>/jacobian_column.parquet       d response / d source gene count
models/<model>/contract.json                 the declared response contract
models/<model>/environment.json              pinned packages, image, device
models/<model>/demo.json                     what was run, timings, checks
```

`uv run python -m scfm_pfm.verify --all` re-checks every claim above against these files: matrix shapes and finiteness, knockdown and DE, strata, ceilings bounded and ordered, the reference Spearman against its gate, vocabulary positions inside each model's vocabulary, and one finite response vector per model with its contract fields filled in.

## Provenance

| artifact | source | revision or checksum |
|---|---|---|
| HepG2 matrix | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_hepg2_raw_singlecell_01.h5ad | md5 `dfc676e4186b8c8c173ff14e1194cc8a` |
| Jurkat matrix | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_jurkat_raw_singlecell_01.h5ad | md5 `dae9fab2a8b0430099a76ce60d0c5405` |
| scgpt vocabulary | wanglab/scGPT-human | `a24c237737a40f3720f75abb555489e9fe753be6` |
| aido_cell vocabulary | https://files.pythonhosted.org/packages/65/3c/4f36684dad8b4cc5b3145dea2f05f25fd8a3ae29934f90e30f31f1ca377b/modelgenerator-0.1.3.post0.tar.gz | `6fb102f3bf76d75a04cf93ca57f55bf5` |
| scprint vocabulary | jkobject/scPRINT | `8e1ad1a1bc40c1fc2f4f1700fe27923eea9a2bed` |
| transcriptformer vocabulary | czi-transcriptformer | `59fe97ed61265c99e1059514fcf6ff2a` |
| scgpt checkpoint | wanglab/scGPT-human | `a24c237737a40f3720f75abb555489e9fe753be6` |
| aido_cell checkpoint | genbio-ai/GB.Cell-100M (plan alias genbio-ai/AIDO.Cell-100M, upstream rename) | `a983b388e1025860f2c413ecef60e76078253c24` |
| scprint checkpoint | jkobject/scPRINT:medium-v1.5.ckpt | `8e1ad1a1bc40c1fc2f4f1700fe27923eea9a2bed` |
| transcriptformer checkpoint | s3://czi-transcriptformer/weights/tf_sapiens.tar.gz | `tf_sapiens (public bucket object, unversioned)` |
| reference release | tahoebio/replogle-nadig-de-rhaister | pdex/all_pdex.parquet |

Seed 42 throughout. Configuration lives in `src/scfm_pfm/config.py` and is copied into every `qc.json`, so a later run can see exactly which thresholds produced these matrices.

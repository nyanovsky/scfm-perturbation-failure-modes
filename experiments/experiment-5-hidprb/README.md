# Run 5 — does the input intervention propagate at all?

**Status: both conditions are complete as of 2026-08-31.** `mean_control` covers three models
x two cell lines at full in-scope coverage; `individual_cells` (S3) covers scGPT and scPRINT
at **16 real control cells** per cell line, TranscriptFormer excluded on cost. Every block was
written by one version of the driver. 2264 checks pass, 0 fail, 26 are recorded absences
(TranscriptFormer's architectural C1/S1 gaps).

⚠️ **S3 is the number to quote, and it is not 1.000.** The split-half score reads both halves
inside the *same* anchor profile; S3 holds out whole cells, and that is what licenses the word
*generalizes*. At the knockdown dose the two disagree sharply --- scGPT/HepG2 last layer goes
from **1.000 split-half to 0.457 held-out**, scPRINT from 1.000 to 0.189. See "What held-out
cells change" below.

The result is in the next section. Also below: one architectural finding about
TranscriptFormer that changes how the whole thread's results for that model have to be read,
and the smoke-stage controls, kept because they are what established the intervention lands
at all.

## The question

Runs 2–3 found that a static gene table beats every queried-response feature, and that
scGPT's queried response does not clear the shuffled-label band at all. Neither says **why**.
One hypothesis is that the information is in the model but not decodable from the output head.

This run tests the weakest precondition of that hypothesis, which is also the cheapest thing
to measure: after an input intervention on gene *i*, do the hidden states of the **other**
genes' tokens carry any trace of *which* gene was intervened on?

Both outcomes say something. Every head in the thread is a shared per-gene-token function of
the last hidden state (`ExprDecoder.fc`, `decoders.ExprDecoder`, `CountDecoderHead 'mu'`,
`Linear(hidden,1)` — each recorded as `output_head` in its adapter's `contract()`), so the
response scalar `e_ij` is a strict lossy projection of `Δh_ij`. The hidden state cannot carry
less. If identity is not decodable there either, the failure is upstream of the readout and
the full hidden-state arm is unnecessary. If it is, the bottleneck hypothesis survives.

It also sidesteps `TODO.md` §1: run 2's per-query standardization divides out the dose axis of
`vvp`/`gdt`, so "the response adds nothing" is strictly a claim about *that encoding*. This run
reads raw `Δh` and never applies that normalization.

## ⚠️ TranscriptFormer's intervention is not the same operation as the other two

Established here on 2026-08-29 by direct probe, and it is not recorded anywhere else in the
tree. Reading `transcriptformer/model/model.py:280-330`:

```python
right_shifted_counts = torch.cat([ones_like(gene_counts[:, :1]), gene_counts[:, :-1]], dim=1)
...
right_shifted_gene_embeddings = torch.cat([aux_embeddings, right_shifted_gene_embeddings], dim=1)
...
log_counts = torch.log1p(right_shifted_counts + eps)
score_mod  = self._score_mod_factory(log_counts, emb_mode=embed)
transformer_output = self.transformer_encoder(x=..., score_mod=score_mod, block_mask=...)
```

Two things follow, both verified empirically:

1. **A gene's count never enters the residual stream.** It is turned into an attention *score
   modifier*. Editing any gene's count moves **no position at all** at the input to block 0 —
   measured, `max |Δ| = 0` at every position for every gene tried. For scGPT and scPRINT the
   value is part of the input embedding; for TranscriptFormer it is not. These are different
   interventions wearing the same name.
2. **Counts and tokens are right-shifted**, so a gene's own count is never an input at its own
   position. Run 2 already recorded the consequence as `self_response_defined = False`; this is
   the mechanism behind it.

Consequences for this run, all handled rather than papered over:

- the perturbed gene's own position is `nan` for TranscriptFormer, so **C1 is unavailable by
  construction** — a recorded ABSENCE, never a failure and never a zero;
- **the S1 propagation ratio has no denominator** for this model. Its numerator, the absolute
  off-target norm, is defined and is what its depth profile is read from
  (`propagation_off_target_norm` in the summary table);
- the auxiliary token **leads** the sequence. This is not inferred: the checkpoint's own forward
  concatenates `aux_embeddings` first and recovers the genes with
  `transformer_output[:, num_aux_tokens:, :]`. The adapter takes the model's own slice, and
  raised rather than guessing until it was pinned — guessing the wrong end offsets every gene's
  state by one and yields a perfectly well-shaped array belonging to the wrong genes.

## What has actually fired, on real model output

Smoke runs (8 perturbed genes, layers 0 / 1 / last, HepG2, `mean_control`), all three models:

| check | scGPT | scPRINT | TranscriptFormer |
|---|---|---|---|
| C2 layer-0 off-target delta exactly `0.0` | PASS (`0`) | PASS (`0`) | PASS (`0`) |
| C1 perturbed gene's own position moves | PASS (median 6.89, 0/40 zero) | PASS (median 1.05, 0/40 zero) | **ABSENT** (by construction) |
| C3 no-op queries exactly zero | **ABSENT** (0 of 40 were no-ops) | **ABSENT** (0 of 40) | **ABSENT** at 8 genes — but **not** zero at full coverage, see the count-clip note below |
| S1 propagation ratio, layer 1 *(median, dimensionless)* | 0.0640 | 0.1441 | **absent** — no denominator |
| S1 propagation ratio, last layer *(median, dimensionless)* | 1.9909 | 0.9309 | **absent** — no denominator |
| max single-position off-target ‖Δh‖, layer 1 / last *(model's own hidden units)* | — | — | 2.38 / 20.89 |

### ⚠️ TranscriptFormer's value grid has four distinct doses, not five, and it has real no-ops

`to_native` clips counts at the model's own training-time `COUNT_CLIP = 30`, so the 16 and 64
CP10K grid points both saturate to 30 and the top of the grid is not the dose the record says
it is. For any gene whose mean-control level already sits above the clip, the 64 CP10K query is
then **bit-identical to the baseline** — a genuine no-op at the *primary* anchor, not only at
`individual_cells`.

This is not a footnote: it aborted a 524-gene TranscriptFormer sweep at gene ~300, because
`is_no_op` tested `|Δ| <= 0` and the model's `nan` self position fails that test, so a query
behaving exactly as designed was reported as two bit-identical inputs disagreeing. Both halves
are fixed — the self position is excluded from the no-op test by read-out index, and the sweep
no longer lets one block's failure kill the others — and both are regression-tested.

The consequence for reading results: TranscriptFormer's dose axis is `(0, 1, 4, 30, 30)`, its
top two grid columns are duplicates, and its no-op count at `mean_control` is **not** zero. The
`value_cp10k` column records the *requested* dose; `layout.notes.count_clip` in each block
records what the model actually received.

Measured on the full HepG2 block: **20 no-ops of 2,620 queries, every one at 64.0 CP10K**, each
with a pooled signature of exactly `0.0`. C3 therefore fired for the first time in this run,
and passed. The analysis drops those rows at their own dose before any statistic sees them and
re-states the chance level as `1/n_kept` — leaving them in would not merely add noise, it would
put dead rows in the retrieval database while `1/n` still claimed they were live candidates.
That handling was missing from `analysis.py` when it was first written and is the second defect
this one architectural fact turned up.

**C3 did not pass — it did not run.** Finding zero no-ops at `mean_control` is the predicted
result and a reason that anchor is primary: every read-out gene sits above run 1's 0.5 CP10K
floor, so a zero-out query is never accidentally a no-op. But that means the exactly-zero
comparison had nothing to compare, so C3 is recorded as an **absence**, not a pass. "It ran" is
not "it is right", and neither is "it had no work to do". The control fires for the first time
at `individual_cells`, where no-ops are common (12.5% HepG2 / 17.5% Jurkat of gene×cell pairs)
and those rows must be dropped, never averaged in.

The last row of the table is **not** the same statistic as the two above it: scGPT's and
scPRINT's numbers are medians of a dimensionless ratio, TranscriptFormer's is a maximum over
single positions in that model's own 2048-dimensional units. They are in separate rows because
putting them in one would invite exactly the comparison they do not support.

**These are 8-gene smoke numbers and nothing may be read from them.** They establish that the
intervention lands and that the controls can fire, which is what a smoke run is for.

## The result

**Full grid: three models x two cell lines at `mean_control`, full in-scope coverage (524
HepG2, 705 Jurkat), each model's layer sweep, 2,150 scored rows.** Numbers below are the
zero-out dose; every one traces to a row of `results/probe/summary.parquet` through
`claims_manifest.json`.

Split-half retrieval of **which gene was perturbed**, read from the hidden states of the
**other** genes' tokens (the perturbed gene's own position is excluded throughout):

| | layer 1 | last layer | null max | chance |
|---|---|---|---|---|
| scGPT, HepG2 | 1.000 | 1.000 | 0.0057 | 0.0019 |
| scPRINT, HepG2 | 0.998 | 1.000 | 0.0057 | 0.0019 |
| TranscriptFormer, HepG2 | 1.000 | 1.000 | 0.0057 | 0.0019 |
| all three, Jurkat | 1.000 | 1.000 | ≤0.0057 | 0.0014 |

Layer 0 sits at chance with every signature exactly degenerate, so the control that would
catch a mis-attributed hook holds in all six blocks.

**Four controls. Three clean; the fourth does not generalise past scGPT — see below.**

| control | asks | result |
|---|---|---|
| C6, magnitude | is it response *size*, which tracks the perturbed gene's expression level? | norm-only reaches 0.027–0.380 against 1.000 — it is **direction**. ⚠️ that range is the **last layer at dose 0**; over the full sweep it is 0.016–0.702 |
| C4, static table | is the signature the gene table echoed down the residual stream? | CKA 0.23–0.53: shares geometry, **is not** the table |
| C7, response group | are paralogs being counted as misses? | 1.000, consistent with the strict score |
| rank structure | is it one direction broadcast to every token, making identity trivially poolable? | ⚠️ **scGPT only** — 5–34% shared. Recomputed for all three models it is 0.22–0.29 (scGPT), 0.22–0.68 (scPRINT), **0.58–0.81 (TranscriptFormer)** |

### ⚠️ Corrections to the two rows above, made 2026-08-30

Both were found by rescoring the blocks already on disk — no GPU, no re-extraction. Neither
overturns the retrieval result; both narrow how it may be worded.

**1. The rank-structure control was generalised from one model to three.** The 5–34% figure is
correct but comes from `scratch/rank_structure.py`, which ran on **scGPT / HepG2 only, 6
perturbations, dose 0, layers 1/4/12**. Recomputed at each model's last block over the full
sweep, the fraction of the delta's energy lying in the direction shared by every read-out gene
is:

| model (HepG2 / Jurkat) | VVP=0 | VVP=16 |
|---|---|---|
| scGPT | 0.22 / 0.25 | 0.26 / 0.29 |
| scPRINT | 0.22 / 0.23 | **0.68 / 0.66** |
| TranscriptFormer | **0.60 / 0.58** | **0.79 / 0.81** |

So "~80% is differentiated across read-out genes" holds for scGPT and **fails for the other
two**. This does not make the retrieval trivial — a pure broadcast shared across *perturbations*
would put wrong-pair distances near zero and they are 0.29–0.89 — but the split-half estimator
averages over read-out genes and therefore reads the broadcast component by construction. For
TranscriptFormer, "richly encoded in **every other gene's** hidden state" is stronger than this
estimator can support; "the perturbed gene is identifiable from the other genes' states"
is what was actually measured.

**2. There is no control for the generic per-gene response, and scPRINT would not pass it.**
Each read-out gene has a displacement it makes whenever anything is knocked down (the average
of its delta over all perturbations). Being identical across perturbations it carries no
identity, so it should be removable without cost. Removing it leave-one-out and rescoring
`retrieval_top1` at the last block:

| model (HepG2 / Jurkat) | VVP=0 | VVP=16 |
|---|---|---|
| scGPT | 1.000 / 1.000 | 1.000 / 0.996 |
| TranscriptFormer | 1.000 / 1.000 | 1.000 / 1.000 |
| scPRINT | **0.805 / 0.853** | 1.000 / 0.996 |

Two of three models are untouched; for TranscriptFormer the wrong-pair distances actually grow
(0.030 → 0.059), so the generic component had been compressing perturbations together rather
than separating them. **scPRINT at the knockdown dose loses about a fifth of its
perturbations.** This is a missing control, not a contradicted number.

Reproducers, both torch-free and reading only `pool.npz` / `norms.npz`:
`scratch/centred_retrieval.py` and `scratch/distance_distribution.py`. The latter also rebuilds
the retrieval distance matrix, which is not stored but is exactly reconstructible; it shows the
own-vs-wrong separation behind every 1.000 (own distance 0.0002–0.003, nearest wrong 0.03–0.25,
smallest margin anywhere 0.0007).

### What this means

Against the reading written into this file *before* any of it was computed, this is squarely
the **bottleneck** branch:

> the intervention propagates; the perturbed gene's identity is perfectly and richly encoded
> in every other gene's hidden state, at every depth, in all three models and both cell lines
> — and the decoder head exposes almost none of it.

So runs 2–3's finding is a statement about **the readout, not the weights**. "The queried
response adds nothing over a static gene table" remains true as measured, and its explanation
is not that these models failed to encode the perturbation. They encoded it; the per-gene
scalar the head emits is where it is lost.

Two limits on that sentence, both real.

⚠️ **The first has since been measured, and it bites.** The block quote above says "perfectly
and richly encoded"; that was written from the split-half estimator, which reads both halves
inside one anchor profile and so never tested transfer between cells. S3 has now run, and at
the knockdown dose the last-layer score falls from 1.000 to **0.457** (scGPT/HepG2) and
**0.189** (scPRINT/HepG2). The direction of the finding stands --- 0.457 is 240x chance and
101x its own null --- but **"perfectly" does not**, and the word should not be carried into
any text about S3. The next section has the numbers and two further qualifications.

The second limit is unchanged: decodability of identity is not the same quantity as a *correct*
response. This run shows the information is present, not that the model's implied response
matches the measured one.

## What held-out cells change (S3, run 2026-08-30/31)

`individual_cells`, 16 real non-targeting control cells per line, kept unaveraged, one probe
block each. Nearest-centroid retrieval with the split **over cells**: the centroid is built
from 8 training cells and evaluated on the 8 the centroid never saw. scGPT and scPRINT only;
TranscriptFormer would have cost ~24x its already-large mean-control time and was excluded by
the job script, not by an argument about the result.

**Strict top-1, best layer per cell line** (chance 1/524 HepG2, 1/705 Jurkat; every null_max
here is <= 0.0076):

| model / line | VVP=0 | VVP=1 | VVP=4 | VVP=16 | VVP=64 |
|---|---|---|---|---|---|
| scGPT / HepG2 | 0.561 | 0.700 | 0.931 | 0.979 | 0.995 |
| scGPT / Jurkat | 0.523 | 0.728 | 0.914 | 0.978 | 0.994 |
| scPRINT / HepG2 | 0.388 | 0.315 | 0.739 | 0.939 | 0.976 |
| scPRINT / Jurkat | 0.309 | 0.334 | 0.691 | 0.958 | 0.990 |

Every score at layer > 0 clears its own null, worst margin 0.067. Layer 0 sits at
0.0007-0.0031, so C2 holds across all 22 new depth-units.

**Three things follow, and the first two narrow this run's headline.**

**1. The perfect split-half score does not survive a cell split, at the dose that matters.**
Both halves of the split-half estimator are read inside the *same* anchor profile, so it never
tested transfer between cells. At the knockdown dose, last layer:

| | split_half | heldout_cell |
|---|---|---|
| scGPT / HepG2 | 1.000 | **0.457** |
| scGPT / Jurkat | 1.000 | **0.409** |
| scPRINT / HepG2 | 1.000 | **0.189** |
| scPRINT / Jurkat | 1.000 | **0.164** |

0.457 against a chance of 0.0019 is still an enormous effect --- 240x chance, 101x its own
null --- so the qualitative claim survives: the identity of the perturbed gene *is* carried in
the other genes' hidden states, across cells. But "perfectly encoded" is a statement about the
split-half estimator and must not be carried over to S3.

**2. The near-perfect numbers live at doses the screen is not about.** The scores only approach
1.000 at VVP=16 and 64, which is strong *overexpression*. VVP=0 --- setting the gene's input to
zero, the direction a CRISPRi knockdown actually goes --- is the weakest column for both
models. Any sentence about "the perturbation" should name its dose.

**3. scPRINT's depth profile is non-monotonic and runs the wrong way.** Its best layer is 1
(0.388 HepG2 / 0.309 Jurkat at VVP=0), collapsing to 0.098 / 0.070 at layer 4 and recovering
only partly by layer 8. scGPT's profile is nearly flat with a mild decline (0.561 at layer 1 ->
0.457 at layer 12). So in scPRINT the perturbation's identity is *most* recoverable right after
the first block and is progressively lost with depth --- the opposite of what "the information
is there but the head cannot reach it" predicts, and worth its own look before that phrase is
used about scPRINT.

### Two defects this condition exposed, both in the analysis rather than the probe

Recorded because S3 was the first thing to run either code path, and neither would have been
found by re-reading the mean-control result.

**`heldout_cell`'s C7 control had no null.** `verify`'s `every_decision_score_has_a_null`
failed on exactly the 110 new `group_collapsed_top1` rows: the held-out path computed
shuffled-label bands for `retrieval_top1/top10/mrr` and hardcoded `nan` for C7, while the
split-half path had always given C7 a band. The check's own contract is that exemptions are "by
name rather than by omission", so this was an omission. Fixed.

**The first attempt at that fix was a regression, and is the more instructive of the two.**
Permuting the database looks like it should be accompanied by permuting the group labels ---
column `j` of `cosine_distance(q, centroid[perm])` holds perturbation `perm[j]`, so surely its
group is `groups[perm][j]`. Doing that makes C7's null exactly **1.000**, i.e. vacuous:
permuting the database breaks the *strict* score only because `rank_of_own` reads the diagonal
as truth, whereas C7's truth is the response group of whatever came back nearest, and the
permutation leaves the geometry --- hence what comes back nearest --- untouched. Relabel
consistently and you retrieve the query's own group by construction.

Leaving the labels unpermuted, which is what the shipped code does, makes the retrieved column
index an effectively uniform perturbation --- which *is* C7's null. Measured at
scgpt/hepg2/layer 12/dose 0: 0.1088 via the database permutation, 0.1088 via a direct
group-label permutation, against an analytic group-collision rate `sum(p^2)` of 0.0963. So the
shipped code was correct and only its docstring was wrong. Reverted to it; both paths now
agree (C7 null median 0.059 split-half, 0.067 held-out) and no published number moved.
`tests/test_c7_null.py` pins the equivalence, including a test that asserts the
permute-both combination returns exactly 1.000 --- the test that would have caught this. The
regression passed `verify` cleanly, because every row had *a* null; the null just measured
nothing.

### `analysis.py` now runs its depths in parallel

56 depth-units on 22 workers (`config.N_ANALYSIS_WORKERS`, `--jobs` to override), 21.9 min
against ~57 min serial for the smaller mean-control-only table. A depth is the right unit: it
is the level at which arrays are read off disk and no statistic crosses depths.

This is an **execution knob and not part of the pre-registration** --- every seed in the module
is derived from the metric cell's own name (`derive_seed`), so the worker count cannot reach a
number. Asserted, not assumed: `--jobs 1` and `--jobs 8` produce a bit-identical frame, and all
2,150 mean-control rows are bit-identical to the pre-refactor serial run (max abs diff 0.0 on
`score`, `lo`, `hi`, `null_max`).

## Design

- **Anchor.** `mean_control` primary — the anchor every scored arm in runs 2–3 is read at
  (`PRIMARY_ANCHOR`), so a null here speaks to the published result. It supplies exactly one
  `Δh` per perturbation, so the primary statistic is a **split-half retrieval over read-out
  positions**, not a trained classifier. `individual_cells` secondary, with 24 real control
  cells kept **unaveraged** — run 2's shipped anchor uses four and averages their responses,
  which is right for a response feature and destroys the axis a probe trains on.
- **Layers.** Layer `l` = output of block `l`; `l = 0` = post-embedding, run as control C2 and
  not as a sweep point. Every third block plus the last: scGPT {1,4,7,10,12}, scPRINT
  {1,4,7,8}, TranscriptFormer {1,4,7,10,12}. A last-layer read alone cannot separate the two
  hypotheses — information that rises through the stack and collapses at the head is
  *discarded*; flat-and-low at every depth is *absent*.
- **The perturbed gene's own position is excluded from every headline statistic.** Decoding
  *i* from `h_i` is trivial and says nothing about propagation; it is kept only as C1.
- **Cost.** One shared baseline row plus one row per (gene, value) — about 1/15 of run 2's
  per-anchor work, which is what makes this affordable on one RTX 3090.

Controls C1–C7, the null band and the group bootstrap are specified in the plan and in
`config.py`; every constant is echoed verbatim into each block's `manifest.json`.

## One divergence from the plan, recorded rather than hidden

The plan said a degenerate (all-zero) `Δh` should come back **undefined**. The implementation
instead returns a score **at chance** plus explicit `n_degenerate` / `degenerate` flags, with
ties broken at a seeded random rather than by array index. That choice is load-bearing: on a
dead layer every cosine distance is exactly 1.0, and breaking ties by index would report
**top-1 = 1.0 on a model that does nothing**. Measured over 400 tie-break seeds the estimator
sits at chance (0.05275 vs 0.05000 at n=20; 0.00516 vs 0.00500 at n=200), so the tie-break is
unbiased rather than lucky. It satisfies "counted, never a spurious 1.0" while keeping the
chance reference a `nan` would lose.

## Layout

| what | where |
|---|---|
| hidden-state accessors, added by subclassing run 2 | `src/scfm_probe/models/hidden.py` |
| the pure index maps that put a state back in read-out order | `src/scfm_probe/models/reassembly.py` |
| half-split, pooling, retrieval, propagation | `src/scfm_probe/probe/{pool,retrieval}.py` |
| the driver | `src/scfm_probe/probe/run.py` |
| environments, and the scPRINT import trap | `envs/README.md` |
| the three job scripts, in the order they must run | `jobs/` |

Run 2 is imported and never edited: its shipped blocks, manifests and reports are cited by the
paper, and the adapter producing run 5's hidden states has to be the same object that produced
run 2's responses — down to the path it reads the weights from.

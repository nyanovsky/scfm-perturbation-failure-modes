# Run 6 — the hidden-state response arm

**Status: scored, 2026-09-02.** Stage 1 is complete for four of five models on both cell
lines — 38 units, 0 failures — and the findings are in §0 below and in `manifest.yaml:results`.
TranscriptFormer is deferred and the `k = 16` rung is not yet run; `PROGRESS.md` §5b is the
build-state handoff and lists everything still open. This file is the design record and is
authoritative for *intent*; the numbers live in `results/summary/` and the manifest. Read
`docs/INDEX.md` §4 (run 2's blocks), §5 (run 3's arms) and §6b (run 5) first; this file
assumes them.

---

## 0. What was found

Every number here is `residual__weighted__disattenuated` — run 3's **pre-registered** cell —
at `k = 64` (arm dimension 640), grouped split, `mlp`, unless said otherwise. Arms are only
*ordered* by `centered__weighted__disattenuated`, both are in `results/summary/stage1_depth.csv`,
and **neither confirms the other**. ⚠️ The `k = 16` rung is **not yet run**, so the verdict
currently stands on the re-registered rung alone (§2, `jobs/stage1_k16.sh`).

### The arm carries pair-specific signal, everywhere

All **38 of 38** (cell line, model, layer) cells clear their own machinery null and their own
width null, in their own cell. Across all 38, `hidperm` spans **−0.0055 … 0.0094** and
`hidrand` **−0.0053 … 0.0026**, against arms of 0.046 … 0.170. At each model's primary layer
the arm additionally clears its **own** five shuffled-label draws — 8 of 8. So run 5's
identity-level result does extend to pair resolution: this is not a null run.

### But it does not reach the static gene table

| | HepG2 | Jurkat |
|---|---|---|
| `null_randpair` | −0.0003 | −0.0022 |
| `coexp` | 0.0601 | 0.0543 |
| `coexp_node` | 0.1361 | 0.0613 |
| **best hidden arm** | **0.1440** (scGPT l7) | **0.1067** (scGPT l10) |
| `scgpt_emb` (static table) | 0.2695 | 0.1657 |
| `scprint_emb` | 0.2463 | 0.2226 |
| `transcriptformer_emb` | 0.2019 | 0.2428 |

The hidden arm beats plain co-expression in both lines and reaches ~53% (HepG2) to ~64%
(Jurkat) of the static scGPT table. It beats the static table **nowhere**. ⚠️ The two lines
disagree on the ranking against `coexp_node` — 0.1361 in HepG2 against 0.0613 in Jurkat — so
neither line's ordering may be quoted alone.

**So H2's answer is not "the head is the bottleneck".** The information the head discards is
real and measurable, and it is still worth less than a gene table that is not queried at all.
That is a *weaker* readout-bottleneck claim than run 6 was built to test, and a
correspondingly stronger version of runs 2–3's negative.

### The depth profile rises and then flattens — it does not collapse

Every model's maximum is mid-stack (l4–l10) and every scGPT-family model is lowest at the last
block. ⚠️ **This is not "rising then collapsing".** No interval was computed for these rows,
and adjacent-layer differences of 0.02–0.03 are inside what 226 (HepG2) / 452 (Jurkat)
response groups resolve. §8's H4 is answered as *rises through the early stack, then flat
within resolution*.

### ✅ The fitted basis is doing no work — §2's concern is answered empirically

`hidrp` is real `Δh` through a **frozen random orthonormal** basis. Against `hid` over all 38
cells: mean difference **−0.0005**, range −0.0305 … +0.0400, and `hidrp ≥ hid` in **20 of 38**.
A random projection scores the same as the fitted PCA one.

That is the measurement §2 asked for. The uneven captured energy (0.8590 … 0.9998 at `k = 64`,
20 of 38 cells at or above the declared 0.95) therefore **cannot** be what drives any
arm-to-arm difference, including `scgpt_ft` against `scgpt_ft_frozen`. It does **not** retire
the declared target, and the wording of §2 stands unchanged.

### ⚠️ Fine-tuning put nothing into the weights that the hidden states expose — as a bound

The five-fold rows for the two fine-tuned keys are training numbers for three folds in five
and **may not be quoted**. Restricted to folds (0, 1) through run 4's own `restrict_scores`
(fold vector: **524 of 524** and **705 of 705** agree with run 4):

| | HepG2, 5-fold | HepG2, folds (0,1) | Jurkat, 5-fold | Jurkat, folds (0,1) |
|---|---|---|---|---|
| scGPT (frozen) | 0.0887–0.1440 | 0.0610–0.0946 | 0.0602–0.1067 | 0.0675–0.1427 |
| scGPT-ft trainable | 0.0979–0.1701 | **−0.0180–0.0278** | 0.0837–0.1034 | 0.0514–0.1051 |
| scGPT-ft frozen table | 0.0735–0.1309 | −0.0038–0.0713 | 0.0462–0.0907 | 0.0566–0.1058 |

In HepG2 the fine-tuned arm's apparent advantage vanishes entirely and it does not separate
from its own shuffled draws (−0.0357 … −0.0039). In both lines frozen scGPT is **higher** than
the fine-tuned model at every depth on the folds it is entitled to.

**This is a bound at the achieved resolution, not a demonstrated null** — run 4's limitation
carried over verbatim. The restriction leaves 243 of 524 perturbations in **54 of 226**
response groups (HepG2) and 282 of 705 in **137 of 452** (Jurkat), and the group-bootstrap
intervals overlap throughout: HepG2's `scgpt_ft_frozen` at l10 is 0.0176 **[−0.0077, 0.3082]**.
On folds 0 and 1 co-expression itself is not separated from the band. And the contrast is
native-regime (flag on) against only-regime, never a weights contrast (§6).

### Read both translators

`mlp` ≫ `ridge` throughout — scGPT/HepG2/l12 is 0.1213 against 0.0584. At 640 dimensions the
translator contrast is doing real work, which is exactly what the `k = 64` re-registration
(§2) said would happen and why `mlp` alone would under-report.

### Where the numbers are

| what | where |
|---|---|
| Every scored row, both metric cells, tracked | `results/summary/stage1_depth.csv` |
| The null band each arm was read against | `results/summary/stage1_null_bands.csv` |
| The in-job reference arms | `results/summary/stage1_references.csv` |
| The shuffled-label draws | `results/summary/stage1_shuffle_draws.csv` |
| Counts and the pre-registered cell | `results/summary/stage1_headline.json` |
| The depth figure, both cells | `figures/depth_profile_{residual,centered}/` |
| Restricted rows + group-bootstrap CIs (heavy) | `$SCFM_PFM6_DATA_ROOT/sweep/stage1/restricted/` |

**Still open**, in `PROGRESS.md` §5b: the `k = 16` rung; TranscriptFormer (~9 h extraction
**and** the `nulls.hidperm` fix, §7); the `random` split; the `k = 1`/`4` rungs; the centred
basis; `claims_manifest.json`; a `verify --section stage1`; and H2 as a written-out post-hoc
comparison against runs 2–4 as published.

## The question

Runs 2–3 read the **decoder head's per-gene scalar** and found the queried response adds
nothing over a static gene table. Run 4 fine-tuned scGPT with its own recipe and the response
features still did not leave the band — as a bound. Run 5 looked inside and found the perturbed
gene's identity **is** recoverable from the *other* genes' hidden states: split-half top-1 =
1.000 at every block layer, though that estimator reads both halves inside one anchor profile,
and the held-out-cell condition gives 0.457 (scGPT/HepG2, last layer, knockdown dose) and 0.189
(scPRINT).

So identifiability passed as a **prerequisite only**. Run 5 never tested per-gene resolution,
and identity being decodable is not the same quantity as a correct response.

Run 6 closes that gap. It rebuilds run 2's `vvp` and `gdt` feature blocks from **hidden-state
deltas** instead of the head's scalar, and scores them through run 3's translator, folds, split
and metric **unchanged**. Both outcomes are informative:

- the hidden arm clears its nulls and beats runs 2–3's published head scores → their result is a **readout
  bottleneck**, and the information is there but the head discards it;
- it does not → the information run 5 found is **identity-level only** and does not resolve per
  gene pair, which is a stronger and more interesting negative than run 3's.

It also answers `TODO.md` §1 by construction: it reads raw `Δh` and never applies run 2's
per-query standardisation, so the dose axis survives into the translator.

---

## Start here (a fresh session's first moves)

0. ⚠️ **The code is now largely built. Read `PROGRESS.md` first** — it records what
   exists, what has been measured (fine-tuned checkpoints verified, fold vector 524/524
   and 705/705, basis pool 365/512 with 0 leakage, the 3090's ~87 s per MLP fit), the four
   test files that are still missing, and the launch sequence. This file stays the design
   record; `PROGRESS.md` is the build state and is deleted when run 6 closes.
1. **Nothing is blocking — every input is on local disk as of 2026-08-31** (§10).
2. **Stage 0 first.** It needs no GPU and no fetch (§4), and it returns a real result while
   exercising install → stage → score → analyse end to end.
3. **Run 6 emits hidden states only — never a decoder-head reading** (§5b). The head results are
   settled; that comparison happens afterwards against runs 2–4 as published.
4. Scaffold to `docs/INDEX.md` §2. Write `config.py` and `paths.py` first, then the tests in
   §9, **then** the driver.
5. Do not edit anything under `experiments/experiment-{2,3,4,5}-*`. Run 6 writes bundle files
   and calls run 3's runner; that is the whole integration (§3).

---

## 1. What the arm is

One `Δh` is `h(perturbed) − h(baseline)` at one transformer block, for one read-out gene's
token, with the baseline computed **in the same batch under the same declared `perturbed_col`**
(run 5's `probe/run.py:19-27` discipline; its `_baseline_pairing_probe` quantifies why reusing
one baseline across genes is a real bug).

Run 2's `resp` block is 11 scalars per pair (`experiment-2-nygwcv/.../features/extract.py:229-241`):

| run 2 slot | count | run 6 hidden analogue |
|---|---|---|
| `zero_out` | 1 | **dropped** — it is `vvp[:, zero_index]` re-normalised. With no normalisation applied the hidden version would be **bit-identical** to the `vvp` dose-0 slot. Recorded, not silently omitted. |
| `vvp` | 5 | `Δh` at each of `VALUE_GRID_CP10K = (0, 1, 4, 16, 64)` → 5 slots × d |
| `gdt` | 5 | central finite difference of the hidden state in dose at each base value → 5 slots × d |

So the raw arm is **10 slots × d**: 5,120 (scGPT), 2,560 (scPRINT), 20,480 (TranscriptFormer).
This is why extraction needs run 2's **16** forward rows per gene (1 baseline + 5 doses + 10 for
the five central differences), not run 5's 6.

### The reduction: PCA on the model dimension only, slots kept separate

`d → k` by PCA with the **10 slots kept separate**: feature = `concat_s (Wᵀ Δh_s)`, dim `10k`.

Collapsing the slots into one vector would re-flatten the dose axis — the axis `TODO.md` §1 says
run 2 divided out and the axis run 6 exists to recover. So the slots stay.

**One basis per (model, cell line, layer), shared by all 10 slots.** A per-slot basis would put
each dose in its own coordinate frame and "how the response changes with dose" would stop being
a comparison of like coordinates. Consequences, recorded rather than smoothed:

- fit on **raw** `Δh` pooled over slots, so high-dose slots dominate the leading PCs. Deliberate:
  RMS-normalising per slot before the fit would remove exactly the dose-magnitude axis this run
  is built to preserve;
- **no whitening** — run 3's `Standardizer` z-scores every column in-fold anyway, so whitening
  would be absorbed and would only obscure the eigenvalue ordering;
- **uncentred** (second-moment) PCA is primary. A centred variant is a sensitivity arm and is
  exactly run 5's missing generic-per-gene-response control (`docs/METRICS.md` §12.7).

`K_STORE = 64` on disk, `K_PRIMARY = 64` scored (re-registered from 16 on 2026-08-31 --- see §2). PCs are eigenvalue-ordered, so the ladder
`k ∈ {1, 4, 16, 64}` is a **column prefix** — free. `k = 1` gives dim 10, within one of run 3's
`resp` dim 11, so the ladder contains a genuinely width-matched comparison.

⚠️ **"Prefix" is exact for a reader and only float32-exact for a re-projection.** Measured while
writing `tests/test_extract.py` (2026-08-31). Slicing the *stored* block — which is what every
rung below 64 actually does, and the only path any reported number takes — is bit-exact, and the
basis prefix `W[:, :k]` is bit-identical to the `k`-wide basis (`max abs diff 0.0`). But
*re-projecting* `Δh` through `W[:, :16]` and slicing a 64-wide projection to 16 differ by 1–3
float32 ulps (measured 3.6e-7 absolute, 2.7e-7 relative; exactly `0.0` at `k = 64`, ~1e-16 in
float64), because float32 BLAS reassociates a 64-column gemm differently from a narrow one. So the
§9 check `max abs diff 0.0` must be read on the **stored block and the basis**, not on a
recomputed projection — and nothing may re-project to obtain a lower rung.

**Rejected: an extra MLP layer.** Run 3's premise is one fixed translator with only the input
changing. A learned reduction puts the arm outside the ~2,300 runs already scored.

---

## 2. Where the PCA basis is fitted

Run 3 scores only `PERTURBED_STRATA = ("S3", "S4_highest")` perturbations whose target is itself
a read-out gene. The screen holds far more. Counted from run 1's effect matrices and run 2's
scope tables:

| pool | HepG2 | Jurkat |
|---|---|---|
| in-scope, scorable | 524 | 705 |
| **screened, target is a read-out gene, excluded by stratum** | **365** | **512** |
| read-out genes never screened at all | 1,893 | 2,099 |

`N_BASIS_GENES = 64`, seeded draw from the **middle row** — real CRISPRi knockdowns with
measured knockdown efficiency, in the same experimental regime as the scored set, but excluded
from every fold on either split because the stratum filter removes them. Stratified on **both**
`pooled_control_cp10k` decile and measured effect size. Gene ids echoed into every block
manifest. The never-screened pool is the fallback if 64 stratified draws cannot be filled from
365, but it is second choice — those genes are mostly not in the library, so nothing matches
their perturbation regime.

Why this discipline, not a relaxation of one:

- **Strictly stronger than in-fold.** In-fold PCA still lets a perturbation in fold 2 shape the
  basis used to score fold 1. Here no scored perturbation shapes any basis at all.
- **It is what makes the run affordable.** One fold-independent basis means extraction projects
  on the GPU and **raw `Δh` never touches disk**.
- **It keeps run 3 unmodified.** `ArmFeatures.batch(rows, cols)` has no fold argument
  (`experiment-3-yga5ws/.../translate/arms.py:74`), so a per-fold feature array cannot reach run
  3's runner without forking `arms.py`/`runner.py` or running the sweep five times and
  re-aggregating.
- A fixed linear map applied identically to every pair is **already inside run 3's regime** —
  `randgene` and `randpair` are exactly that.

### ⚠️ The risk this carries, measured rather than argued

The exclusion is **not random with respect to biology**: the scored set is the strong-effect
strata and is enriched for core-essential genes, while the basis pool is by construction
weaker-effect. If a model's `Δh` for an essential knockdown visits directions its `Δh` for a
weak knockdown does not, the top-`k` PCs **under-span** the subspace the scored perturbations
occupy.

**The error runs one way only.** PCA never sees the targets and the basis perturbations are
disjoint from the scored ones, so no channel exists by which the basis can encode which
perturbation is which. Under-spanning *loses* signal; it cannot manufacture it. The failure mode
is a **false null, never a false positive** — the safe side for a run whose headline may be
negative, but a limitation to quantify rather than assert. Two things quantify it, both free:

1. **Captured-energy diagnostic.** Accumulate a second `d × d` second-moment matrix over the
   **scored** perturbations in the same streaming pass (33 MB at `d = 2048`), eigendecompose to
   `W_all`, and report — **diagnostic only, never used to score** — the fraction of the scored
   perturbations' `Δh` energy captured by `W_basis`'s top-`k` against `W_all`'s top-`k`, plus the
   principal angles, per (model, line, layer, k). ≥95% closes the concern; below that the arm is
   known to be downward-biased and by how much, and it goes in the limitations *before* any score
   is read.
2. **`hidrp` brackets it.** A random orthonormal basis is unbiased with respect to essentiality
   by construction. `hid ≈ hidrp` means the basis choice is not load-bearing; `hid ≫ hidrp` means
   it is, and (1) is the number to read.

#### ⚠️ MEASURED, 2026-08-31 — half the cells miss the target, and not at random

Read off the block manifests of the eight extracted sets, `k = 16`, before any arm was scored.
The number is the fraction of the **scored** perturbations' `Δh` energy the fitted basis captures
against the best possible `k = 16` basis.

| model | line | l1 | l4 | l7 | l8 | l10 | l12 |
|---|---|---|---|---|---|---|---|
| scgpt | HepG2 | 0.9576 | 0.8921 | 0.9079 | — | 0.9051 | 0.8764 |
| scgpt | Jurkat | 0.9191 | 0.9111 | 0.8931 | — | 0.9273 | 0.9152 |
| scgpt_ft | HepG2 | 0.9873 | 0.9870 | 0.9765 | — | 0.9820 | 0.9706 |
| scgpt_ft | Jurkat | 0.8941 | 0.9268 | 0.9777 | — | 0.9924 | 0.9715 |
| scgpt_ft_frozen | HepG2 | 0.9073 | 0.9173 | 0.9688 | — | 0.9942 | 0.9864 |
| **scgpt_ft_frozen** | **Jurkat** | **0.7848** | **0.8284** | **0.8531** | — | **0.8358** | **0.7665** |
| scprint | HepG2 | 0.9995 | 0.9833 | 0.9843 | 0.9558 | — | — |
| scprint | Jurkat | 0.9989 | 0.9668 | 0.9579 | 0.9086 | — | — |

**19 of 38 cells reach `BASIS_COVERAGE_TARGET = 0.95`; range 0.7665 – 0.9995.** So the concern
this diagnostic exists to close is **not closed**, and the limitation is on the record before a
single score exists — which is the point of measuring it here rather than after.

Two readings, and the second is the one that matters:

- **On its own it is the safe direction.** Under-spanning loses signal and cannot manufacture it,
  so every below-target cell is biased *downward*. A null in one of those cells is weaker evidence
  of absence than its number suggests; a positive result there is not weakened at all.
- ⚠️ **But it is not uniform across arms, and it is worst exactly where run 6's own comparison
  lives.** `scgpt_ft` in HepG2 spans 0.97–0.99 while its control `scgpt_ft_frozen` in Jurkat spans
  0.77–0.85 — a ~15-point gap between two arms whose *difference* is supposed to isolate what
  fine-tuning put in the weights. A difference between them is therefore **partly a difference in
  how well each arm's basis spans its own `Δh`**, not only a difference in what the model encodes.
  The same caution applies to reading depth profiles across a row, where the spread is up to 11
  points (scgpt_ft/Jurkat, 0.8941 at l1 vs 0.9924 at l10).

Consequences to carry into the analysis, not to resolve now:

- **Quote the captured-energy value beside every score**, per cell, the way the null band is
  quoted. A depth profile drawn without it is a profile of two mixed quantities.
- **`hidrp` is now load-bearing rather than a nicety.** It is the arm that says whether the PCA
  basis is doing the work, and it is the natural check on whether the coverage gap is reaching the
  scores. It was extracted for all eight sets and costs no extra forward pass.
- **`k = 64` is free and already on disk.** If the gap tracks `k`, reading the ladder at 64 bounds
  how much of it is a `k = 16` artefact. That costs no GPU, only translator fits.
- Do **not** re-fit the basis on the scored perturbations to close the gap. That is the leakage
  the whole design exists to prevent, and §2's "Rejected: strict per-fold in-fold PCA" already
  argues the alternative.

**The `k = 64` rung, measured 2026-08-31.** Reading the free upper rung helps but does not close it:

| | k = 16 | k = 64 |
|---|---|---|
| cells at or above 0.95 | 19 of 38 | **20 of 38** |
| worst ratio | 0.7665 | 0.8590 |
| spread across cells | 0.2330 | 0.1408 |

So `k = 64` buys **one** additional cell over the target. It halves the spread and lifts the worst
case, and the arm-to-arm asymmetry that matters survives it: `scgpt_ft`/HepG2 0.981–0.990 against
its own control `scgpt_ft_frozen`/**Jurkat** 0.859–0.923, a ~10-point gap between two arms whose
difference *is* the measurement. The gap is specifically a Jurkat-vs-HepG2 asymmetry in the
control arm.

✅ **`K_PRIMARY` was re-registered from 16 to 64 on 2026-08-31 — deliberately, by the user, and
written down here rather than drifted into.** The sequence matters and must be stated wherever the
verdict is: the change was made **after** this captured-energy diagnostic was read and **before any
arm was scored**. So it cannot have been chosen for its answer — no answer existed — but it was
chosen knowing a diagnostic, which is weaker than the original pre-registration and must not be
presented as equivalent to it.

It costs nothing in compute: `k = 64` **is** `K_STORE`, already on disk for all eight extracted
sets, and every rung of `K_LADDER` is reported whichever one is primary. The cost is interpretive,
and the write-up carries it:

- **the `k = 16` rung stays reported and must be quoted beside the `k = 64` verdict**, so a reader
  can see that the primary rung moved and what the other rung said;
- arm dim at the primary rung is now `10 × 64 = 640`, against 160 at `k = 16`. That changes what
  the width-matched comparison is: `k = 1` (dim 10) remains the rung matched against
  `null_randpair`'s dim 11, and a 640-dim arm is **not** width-matched to any run 3 reference, so
  `hidrand` (which controls for width only) becomes the relevant width control at this rung;
- more translator capacity at 640 dims than at 160 also means the ridge/MLP contrast is doing more
  work — read both translators, not just `mlp`.

**Judgement taken, 2026-08-31 (user's call, recorded rather than silently applied):** these levels
are **accepted as livable** for an out-of-sample basis, on the view that 0.95 is a stringent bar —
it is a round number written down before any measurement, not a figure derived from anything. Two
things follow, and `BASIS_COVERAGE_TARGET` is deliberately **left at 0.95** so the record still
reads against the number that was pre-registered rather than one chosen afterwards:

- the wording is *"the declared target was 0.95; 19 of 38 cells reach it at k = 16 and 20 of 38 at
  k = 64; in hindsight 0.95 is stringent for a basis fitted out of sample"* — **not** "the
  coverage concern is closed";
- the captured-energy value is still quoted per cell beside every score, and `hidrp` is still the
  arm that says whether the basis choice is load-bearing.

Nothing here can inflate a positive result: every below-target cell is biased downward only.

⚠️ **`verify` reports these 38 checks as ABSENT, wrongly.** `extract_line` writes
`captured_energy` into the **block** manifest, but `check_basis` looks for it in the **basis**
provenance json, so it emits "the diagnostic is written by the main extraction pass, which has not
run" for 38 cells where it did run and the number exists. The verdict above was read directly off
the block manifests. Fix `check_basis` to read the block manifest — until then the ABSENT is a
false negative, and it is the one check that would otherwise have flagged the table above
automatically.

### Rejected: strict per-fold in-fold PCA

Five grouped + five random `d × d` accumulators is free, but projecting then needs a **second**
GPU pass (or ~460 GB of raw `Δh`), 10 blocks per (model, layer), and either forking run 3 or
running the sweep once per fold and re-aggregating with run 4's `restrict_scores`. That would put
run 6's scoring outside run 3's unmodified rule, which is the one thing this run exists to
preserve. Recorded so the choice is auditable, not taken.

---

## 3. How it reaches run 3 without touching run 3

`LineBundle` reads response blocks from `<bundle>/<line>/features/<model>__<anchor>.npz`
(`translate/data.py:104-129`) with keys `pert_index`, `values`, `value_keep`, optional
`zero_out`, `vvp`, `gdt`. **It does not validate that the value axis has length 5** — `_dim`
reads `e[form].shape[1]` and `batch` gathers `e[form][m, :, cols]`. Therefore:

> Run 6's features install as a **synthetic run-2-shaped block of a pseudo-model**, and run 3's
> unmodified `resp` arm reads them.

- `vvp` ← `(n_pert, 5k, n_readout)`, axis 1 flattened C-order as `(dose, pc)`
- `gdt` ← `(n_pert, 5k, n_readout)`, same order
- `values` ← the dose repeated `k` times; `value_keep` ← all True
- `zero_out` ← **omitted**, so that branch never fires and the dim is exactly `10k`
- `pert_index` ← every scope row, so `row_map` has no `-1`

`resolve_arms` emits `{m}_resp` for every model key in `models`, so the arm is selected with
`--arms <key>_resp`. `{m}_emb`/`_full`/`_hybrid` would `KeyError` on `bundle.emb[m]` and are
never requested. `_keep_predictions` keeps predictions for names ending `_resp`, so saved
predictions come free.

**The nulls install the same way**, as further pseudo-model `resp` blocks, so `_resp_dim` gives
them the identical width by construction. This is the reason not to touch run 3: `randpair` is
**hardcoded to 11** (`arms.py:67`, `arms.py:105`), so a `10k`-wide arm has no matched null in
run 3's code, and widening it would mean editing a file run 4 imports.

### ⚠️ Traps, every one of them load-bearing

| trap | handling |
|---|---|
| A model key containing `__` makes the block unreadable and the arm **vanish with no error** (cp-13 hit this for real) | keys use hyphens: `scgpt-hid-l12`, `scgptft-hid-l12`, …; encode/decode via run 6's own `config.model_stem`/`run_of_stem`, copied from `scfm_ft/config.py` |
| A `features/*.npz` filename with **no** `__` raises `ValueError` at bundle construction and kills every arm for that line | nothing but blocks goes in `features/`; bases live under `basis/` |
| Installing a block redefines `scope_rows("common")` for **every** arm in the sweep | **`--scopes full` only**; `common` exists for AIDO.Cell comparability and run 6 has no AIDO arm |
| `run_cell` **silently skips** a model whose `row_map[rows]` has a `-1` | asserted in `verify`, printed as a number |
| Sharding is a running counter, not a hash, so adding an arm reshuffles every task | own output directory + explicit `--arms`; merge only run 6's own dirs |
| Run 3's `analysis.py` hardcodes arm names in seven places, so a run-6 arm is silently omitted | run 6 writes its own `analysis.py`; imports only run 3's `metrics.py` and `groups.py` |
| `RANK_ONLY_GDT_MODELS = ("transcriptformer",)` does not apply to a run-6 key, so TF's `gdt` keeps its magnitudes | recorded divergence — run 2 measured *output-head* gradient instability, never established for hidden states, and rank-percentiling a PC coordinate is a different operation. TF is also reported **`vvp`-only** (a free column subset) so its verdict does not rest on `gdt` |
| `fit_ridge`'s standardiser subsample is not seeded, unlike `fit_mlp` | pre-existing in run 3, inherited; note it, do not fix it inside run 3 |
| `figures/entity_colors.json` holds nine entities and `EDITORIAL_8` is exhausted | reuse each model's colour, separate run 6's arms by marker/dash |

---

## 4. Stage 0 — a per-pair arm needing **no GPU and no fetch**

Run 5's `norms.npz` already holds `norm` at shape **`(n_pert, n_values, n_readout)`** —
`‖Δh[p, v, g]‖` per pair, per dose, per layer, for 3 models × 2 lines × every sweep layer, on
disk at
`/biodata/nyanovsky/projects/scfm-pert-fail-modes/experiment-5-hidprb/probe/<model>/<line>/anchor=mean_control/layer=<l>/`.
That is **exactly** the on-disk `vvp` layout run 3's reader expects.

So a per-pair hidden-**magnitude** arm of dim 5 is installable immediately: `vvp ← norm`,
`values ← VALUE_GRID_CP10K`, no `gdt`, no `zero_out`.

Why first:

- it is the **per-pair analogue of run 5's C6 control** (`norm_only_top1`, 0.016–0.702 against
  1.000). C6 said identity is carried by *direction* at the pooled level; Stage 0 asks the same
  question at pair resolution, in run 3's metric;
- it de-risks ~9 GPU hours: if magnitude alone clears the band that is the finding, and the
  vector arm becomes a refinement rather than the whole bet;
- it exercises install → stage → score → analyse on real data before any extraction code exists.

Caveats: `norm` includes the self position (TranscriptFormer's is `nan`, so the `(p, p)` pair
follows §10's policy), it covers the 5 `vvp` doses only, and its matched null is a 5-dim
pseudo-model block plus `hidperm` on the same array. Dim 5 against `resp`'s 11 is **not**
width-matched, so quote the dim-5 null, never `null_randpair`.

---

## 5. Extraction

Reuse run 5; the producer needs no change, only a new consumer.

| reuse | from |
|---|---|
| per-layer hidden states by forward hook, three adapters subclassing run 2 | `experiment-5-hidprb/src/scfm_probe/models/hidden.py` — `hidden_batch(x_native, layers, perturbed_col) -> {layer: (B, n_readout, d)}` |
| the torch-free index maps that put a state back in read-out order | `.../models/reassembly.py` |
| anchor construction, scope selection, per-column baseline pairing | `.../probe/run.py:342 anchor_profiles`, `:599-668` |
| the **16-row** query batch (baseline + lo/mid/hi at each dose) | run 2's `features/extract.py:137 query_batch` — **not** run 5's 6-row version |
| GPU-less test fixtures with a duck-typed `hidden_batch` | `experiment-5-hidprb/tests/toy.py` |

The new sink replaces run 5's `probe/run.py:212-229`: where run 5 computes `delta` and
immediately pools, run 6 forms the 10 slots and writes `delta_slot @ W_layer`, so raw `Δh` never
exists beyond one `(n_readout, d)` array.

⚠️ **The write loop is where a completed pass can still be lost, and the culprit is a
diagnostic.** Recorded 2026-08-31 from `tests/test_extract.py`. `extract_line` computes the
captured-energy diagnostic *inside* the per-layer write loop, and `basis.fit.eigendecompose`
raises on a second-moment matrix of zero total energy. A layer whose off-target `Δh` were exactly
zero would therefore abort the job **after every forward pass had run**, and after earlier layers
had already been written — a block directory with some layers in it and no error visible in the
directory itself. On a real checkpoint the scored second moment is never exactly zero, so this
cannot fire in production; it is recorded because the failure mode is *partial output from a
successful computation*, which is the shape of failure this run's checks exist to catch. The
mitigations already in place: the driver writes `failure.txt` into the output directory on any
exception, `jobs/_common.sh` deliberately does not `set -e` so one combination's abort does not
take the others, and `verify --section blocks` must count layers per (model, line) rather than
assert a directory exists.

### 5b. Run 6 emits **hidden states only** — never a decoder-head reading

The head results are **settled**: runs 2–4 measured them, they are published, and run 6 does not
recompute, re-normalise or re-score them. Run 6's output is the hidden-state arm and its nulls,
full stop.

Consequences, all deliberate:

- **`<model>_resp` is not an in-job reference arm** and run 2's shipped response blocks are **not
  fetched** (§10a). Nothing in run 6 reads them.
- **H2 is a post-hoc comparison against the published numbers**, made after run 6 closes, not a
  paired in-job difference. It therefore has no in-job bootstrap and no paired interval — say so
  rather than implying one, and compare against run 3's and run 4's tables as published.
- ⚠️ **That comparison crosses a normalisation boundary.** Run 2's shipped `resp` block is the
  `standardized` form, z-scored across read-out genes within each query — the per-row division
  `TODO.md` §1 says removes the dose axis, and one run 3's per-column `Standardizer` cannot undo.
  Run 6's hidden features carry no such division. So "hidden beats head" read across the two is
  confounded with raw-vs-standardised, and the wording must carry that.
- The in-job references that remain are the ones that need no head reading: `coexp`,
  `coexp_node`, `<model>_emb`, `null_randpair`.

**Layers.** Hooks capture every layer in one forward pass, so the depth sweep is free in GPU time
and costs only disk. Run 5's sets: scGPT `{1,4,7,10,12}`, scPRINT `{1,4,7,8}`, TranscriptFormer
`{1,4,7,10,12}`, plus layer 0 as control **C2**.

**Anchor.** `mean_control` only — the anchor every scored arm in the paper was read at, so a
result here speaks to the published numbers. `individual_cells` would be ~16× the cost.

**Models.** scGPT, scPRINT, TranscriptFormer, plus run 4's fine-tuned scGPT (§6). AIDO.Cell is
not local, scored at zero, and costs ~40× scGPT per forward pass.

### Model-specific facts to carry, not rediscover

- **TranscriptFormer's intervention is not the same operation.** Its count is an attention score
  modifier, never part of the residual stream, and its self position is written `nan`
  (`hidden.py:818-820`). The `(p, p)` pair is undefined for it — match what run 3 already
  receives, per §10.
- **TranscriptFormer's dose axis is `(0, 1, 4, 30, 30)`**: `to_native` clips at
  `COUNT_CLIP = 30`, so slots 4 and 5 duplicate, and there are genuine no-ops at the primary
  anchor (20 of 2,620 in HepG2, all at 64 CP10K). Structural zeros, counted in `qc.json`, never
  averaged away. Runs 2–3's shipped TF blocks carry the identical duplication, so this is
  inherited, not new. `value_cp10k` records the requested dose; `layout.notes.count_clip` records
  what the model received.
- **`gdt` is a central finite difference for every model**, where run 2 used `exact_jvp` for
  scPRINT. Recorded divergence: the hidden-state JVP is available in principle for scPRINT only,
  and using it there alone would make one model's `gdt` a different estimator from the others'.

### The two policies those facts force — as implemented

Both were decided in code before any block was written (`extract/slots.py`, `extract/run.py`,
`metrics_extra.py`, `stage0.py`) and are recorded here because a policy that lives only in a
docstring is a policy a reviewer has to reconstruct.

**1. TranscriptFormer's self pair is `nan`, and it stays `nan`.** The `(p, p)` hidden delta is
undefined for that architecture — the count never enters the residual stream — so writing a `0.0`
there would put a fabricated number into a feature matrix and nothing downstream could tell it
from a measured one. **A measured absence is not a zero.** What makes that safe is that every
scoring cell excludes the column by construction rather than by tolerance, in four independent
places:

- run 3's `pair_index` drops the self pair from the training set, so no fit ever sees it;
- run 3's `weights[..., self_cols] = 0` gives it zero weight in every weighted cell — and run 6's
  `verify --section ft` asserts `self_col == -1` for **0 of 524** and **0 of 705** perturbations,
  because a `-1` there would silently zero the *last* column instead of the intended one;
- the basis fit deletes the self position before accumulating (`fit_basis`, and the same
  `np.delete` in `extract_line`'s scored-moment accumulation), so a basis is never fitted partly
  on `nan`;
- run 6's discrimination score excludes it from **both** members of every pair
  (`config.PDS_EXCLUDE_SELF_COLUMNS`), and sets non-finite entries to `0.0` *before* the pairwise
  pass so the rank-1 self-column corrections remove them exactly. Sanitising **without** the
  exclusion raises rather than scoring a fabricated value — `config.PDS_SANITISED_NOTE`.

`assert_no_op_zero` also excludes the self position from its comparison, because `nan <= 0` is
`False` and the un-excluded form aborted a 524-gene sweep at gene ~300 in run 5.

**2. A no-op is a structural zero, detected exactly and named.** Where the anchor's own value at
the perturbed gene already equals the dose being requested, the two input rows are **bit-identical**
and the `vvp` delta must be exactly `0.0` through a frozen model. So the detection is made on the
**float32 values the query batch actually carries** (`slots.mid_row_values`), not on the float64
CP10K grid — bit-identity is the definition, and a comparison on the grid would miss a pair the
model's `to_native` collapsed. Each one is then

- **asserted** exactly zero (`slots.assert_no_op_zero`, `!= 0.0`, not a tolerance) — if it moved,
  either the batch is not what the estimator thinks it is or the forward pass is not
  deterministic, and both are defects rather than data;
- **counted** and **named by (gene, dose)** in the block manifest
  (`diagnostics.n_no_op_queries`, `diagnostics.no_op_queries`), so a reader can see which rows
  are structural rather than measured;
- **never averaged in.** They are not small responses that happen to round to zero.

Only `vvp` can be a no-op. A `gdt` slot at the same base value is a *derivative* rather than a
substitution and is unaffected, which is the distinction run 3's `NO_OP_AFFECTED_FORMS` already
records for the head features.

### Storage

Projected only, `K_STORE = 64`, 10 slots, float32 → `n_pairs × 640 × 4` bytes per layer.

| | HepG2 (524 × 2,721) | Jurkat (705 × 3,247) |
|---|---|---|
| one layer, one model | 3.65 GB | 5.86 GB |
| all sweep layers, 4 models | ~73 GB | ~117 GB |
| **total** | **~190 GB**, on `/biodata` (9.2 T free) | |

Raw `Δh`, for contrast and as the reason it is not stored: ~15 GB (scGPT/HepG2) to ~94 GB
(TF/Jurkat) per layer at 5 doses, ~210 GB over 3 models × 2 lines, and 2.2× that again for 10
slots.

**RAM/VRAM.** `LineBundle` puts every array on the scoring device, and this machine has 70 GB
RAM / 24 GB VRAM. **Stage one bundle per (model, layer, k)** holding exactly one response block:
0.91 GB (HepG2) / 1.46 GB (Jurkat) at `k = 16`; 3.65 / 5.86 GB at `k = 64`. Never all blocks in
one bundle.

⚠️ **The extraction pass's own RAM was never budgeted here, and it does not fit.** Found
2026-08-31 by extrapolating the 8-gene smoke before launching the wave. `extract_line` allocates
its whole `(n_pert, n_values, n_readout, k)` accumulator up front — 2 variants × 2 forms × sweep
layers — which is **34.0 GB for HepG2 and 54.6 GB for Jurkat** against 70 GB total. In RAM the
larger line would have died *after every forward pass had already run*, the most expensive
possible moment to fail.

The accumulator is therefore **memory-mapped on `/biodata`** rather than allocated
(`config.STORE_MEMMAPPED`, `paths.store_scratch_dir`). Why not something else:

- **streaming a layer out early is not available** — the gene loop is the outer loop, because one
  forward pass captures every layer at once, so no layer's block is complete until the last gene;
- **splitting the sweep into layer groups** costs one extra full forward pass per group, which for
  TranscriptFormer/Jurkat is **+3.4 h of GPU** each;
- **storing fewer components** is not available: `K_STORE` is pre-registered.

`/biodata` sustains ~480 MB/s on mmap, so the 34–55 GB of writes cost ~1–2 min against a 0.3–3.4 h
pass. **Verified bit-neutral on real scGPT weights**: the same 8-gene smoke re-run through the
mapped store reproduced the pre-change blocks over 42 arrays / 217.5 MB at `max abs diff 0.0`. The
scratch files are removed on success and deliberately left behind on failure.

⚠️ **`--smoke` writes its basis to the production path, and a smoke basis is the same shape as a
production one.** `paths.basis_file(model, line, layer)` carries no smoke suffix, so an 8-gene
`--smoke` basis sits exactly where the 64-gene one belongs and passes the `(d, K_STORE)` shape
check — after which a production main pass would project all 524 scored perturbations through a
basis fitted on eight genes. `verify --section basis` now compares the recorded draw size against
`config.N_BASIS_GENES` (`layer{l}_draw_size`), which is the only check that separates them; the
provenance json already recorded `draw.n_requested` and `config.N_BASIS_GENES` side by side. The
8-gene smoke tree was deleted before the production wave. Blocks were already safe —
`check_blocks` compares `perturbed_gene_id` to `targets.npz` row for row, which an 8-gene block
fails. Minor: the smoke path sets `n_basis=SMOKE_N_BASIS_GENES` but leaves `draw.smoke = False`,
so that field cannot be used to tell them apart.

### Cost

Run 5's scGPT/HepG2 `mean_control` block took 440.8 s on the 3090 and its full six-block sweep
~3 h, at 6 rows per gene. Run 6 is 16 rows, so the main pass is ~8 h plus ~1 h for the 64-gene
basis pass. **Time a one-model pilot before committing** — TranscriptFormer dominates (run 2
measured 4,621 / 6,232 s per model-line on an H100 against scGPT's 372 / 643).

---

## 6. The fourth model: run 4's fine-tuned scGPT (`own_trainable`)

Run 4 found the *decoder head* of a fine-tuned scGPT still inside run 3's band. Reading the same
checkpoints' **hidden states** asks whether fine-tuning put a response into the weights that the
head throws away — the one arm that separates "fine-tuning installed nothing" from "fine-tuning
installed something the readout cannot expose". scGPT is the cheapest model, so two more blocks
is ~**+25–30 min GPU**.

Arm: `own_trainable` — scGPT's own gene table, trainable, the standard published recipe
(`experiment-4-74xnwm/src/scfm_ft/config.py:117`).

### ⚠️ It cannot be scored on run 3's five folds

`MODEL_HELDOUT_FOLDS = (0, 1)`. The fine-tuned model **trained on folds 2, 3 and 4**, so run 3's
unmodified five-fold sweep would make three fifths of its perturbations a training number. This
is the one place run 6 cannot reuse run 3's rule end to end.

**The fix is run 4's, imported not reinvented.** `scfm_ft/score.py` already does it: run 3's
runner scores per perturbation, then `restrict_scores` keeps the held-out folds and
`restricted_ceilings` recomputes the ceiling on that subset — the same correlations, the same
weights, read on the subset the model is entitled to. Run 6 imports run 4 as a fourth sibling.

**Frozen scGPT must be re-read on the same two folds**, or a 2-fold score is being compared
against 5-fold scores. Free: `_keep_predictions` saves per-perturbation predictions for `*_resp`
arms and `restrict_scores` re-reads them. That is run 4's `compare_run3.py` pattern.

**And run 4's limitation carries over verbatim.** On folds 0 and 1 co-expression itself is not
separated from the band (`docs/INDEX.md` §9), so whatever this arm shows is a **bound at the
achieved resolution**. Word it that way from the start.

### The perturbation flag is on, in both passes

The flag is how this model is *told* a perturbation is happening; it was fine-tuned with the flag
set, so the no-flag reading is the off-distribution one. **The flag is present throughout.**
Neither of run 4's existing families is this estimator:

| | flag, baseline pass | flag, perturbed pass | dose axis |
|---|---|---|---|
| run 4 value substitution (`forward_batch`, `query.py:83-89`) | off | off | 5 doses |
| run 4 flag family (`pert_flag_response`, `query.py:154-166`) | **off** | on | **none** — one vector per gene |
| **run 6** | **on** | **on** | 5 doses |

```
Δh[p, g] at dose v  =  h( flags[p]=1, x[p]=v      )[g]
                     − h( flags[p]=1, x[p]=anchor )[g]
```

Two things this buys:

- **the estimator stays structurally identical to the frozen models'** — exactly one thing
  changes between passes, the value; the flag is fixed input context like the anchor profile. So
  `vvp`/`gdt` mean the same operation here as everywhere else: the dose sweep and the flag are
  orthogonal;
- **the controls survive** — because the flag is identical in both passes, layer-0 off-target
  `Δh` is still **exactly `0.0`** and C2 holds. A flag-off baseline would have broken it, since
  flag onset does reach other tokens, and would also have conflated flag onset with dose.

Implementation is contained: `_forward` already accepts `flags` (`query.py:83`) and only
`forward_batch` hardcodes `_zero_flags`. Run 6's `_hidden_pass` sets the flag column from
`perturbed_col` and threads it through both passes. ⚠️ `configure`'s inherited layout note reads
`"perturbation_flags": "all zero for this query family: only the input value moves"` — run 6's
block manifest must record the **opposite**, verbatim, or the block misdescribes itself.

⚠️ **The flag claim is a claim about `vvp` alone, and `gdt` cannot test it.** Found while writing
`tests/test_flag.py` (2026-08-31), not previously recorded here. `gdt` differences the two
*perturbed* rows `(v-h, v+h)`, and both of them are flagged under the correct implementation **and
under a flag-off-baseline one** — the baseline row never enters a `gdt` slot. Measured on the
deliberately-broken adapter the test file builds: a flag-only toy gives a non-zero `vvp` at every
layer and a `gdt` of exactly `0.0` at every layer. So:

- the flag-constancy assertion, and with it C2's survival, rests on `vvp`;
- `gdt` is *insensitive* to the defect rather than protected from it, which is a weaker statement
  and must not be quoted as a second independent check;
- a future change to the batch layout that moved the baseline row into the derivative would break
  that, and only the `vvp` assertion in `tests/test_flag.py` would catch it.

### What the flag changes about how the arm is read

1. **It gets its own nulls; run 4's calibration does not transfer.** Run 4's flag family was
   flagged *not a clean null* — 2 of 10 readings above the band in Jurkat. That was the
   **flag-off-baseline, no-dose** estimator, not this one. Run 6's flag-constant version has
   never been calibrated by anyone, so it takes fresh shuffled-label draws in **its own cell**
   plus its own `hidperm`/`hidrand`/`hidrp`, and may borrow **neither** run 4's flag-family band
   nor the frozen models'.
2. **The dose regime is asymmetric, in this arm's favour at the bottom.** Trained on (control
   profile, flag on `p`) → measured post-perturbation profile, it never saw "flag on `p` and
   `p`'s input at 64 CP10K". Dose 0 is arguably *more* on-distribution than the anchor value,
   since CRISPRi does zero the gene; the top of the grid is off-distribution.
3. **Fine-tuned vs frozen scGPT is not a pure weights contrast.** It is "the fine-tuned model in
   its native regime" against "the frozen model in the only regime it has". The right comparison
   and the only one available, but never worded as isolating the weights.
4. ⚠️ **The head comparison for this model carries a flag confound, and it is not fixable inside
   run 6.** Run 4's published `ft_own_trainable_resp` is the *flag-off* head reading, so setting
   run 6's flag-on hidden score beside it compares hidden-vs-head **and** flag-vs-no-flag at
   once. Run 6 does not emit decoder-head output (§5b), so this cannot be resolved in-job. State
   the confound wherever the two are put side by side; do not present the difference as
   hidden-vs-head alone.

### The adapter

`ScGPTPertQuery` already subclasses run 2's `QueryModel` (`scgpt_pert/query.py:33`), so the
hidden variant is the same mixin-first MRO run 5 uses three times:

```python
class ScGPTPertHiddenQuery(HiddenQuery, run4("scgpt_pert.query").ScGPTPertQuery):
    _stack_candidates = ("transformer_encoder.layers",)   # resolve, never guess
```

Five things it must supply: `name` (already `"scgpt_ft"`), an expected layer count for
`_resolve_stack`, a `_final_capture` (run 5's scGPT reads the last layer at
`transformer_encoder` itself, not `layers[-1]`, so the captured tensor is what the decoder
consumes), a constructor bridge (`ScGPTPertQuery.__init__(model, n_readout, device, amp)` takes
an already-built model where run 5's adapters take `(cache, universe, vocab, device)`), and the
flag tensor threaded into both passes.

⚠️ **Resolve the stack path against the *generation* model**, not plain scGPT — run 4 builds
`import_generation_model` with a substitutable `TableGeneEncoder`. `_resolve_stack` refuses
ambiguity and returns a reason; pin the path from what it reports.

**Keep `amp=False`.** The adapter's own docstring records why: a response is a difference between
two passes agreeing to ~1e-4, which in half precision is below the representable gap and the
feature becomes rounding noise. That applies with more force to `Δh` than to a scalar.

### Two run-4 specifics that are easy to get wrong

1. **Two executions.** `SCFM_FT_RUN_SUFFIX=_pearson` is the reported one (highest validation
   Pearson, as scGPT's tutorial selects); unsuffixed is the earlier masked-MSE execution. Read
   `_pearson`, record the suffix — unset silently reads the wrong tree.
2. **The folds must be run 4's, asserted not assumed.** `ingest.py:38 fold_assignment` recomputes
   run 3's grouped folds and `make_folds` is deterministic given the same effect matrix and scope
   rows, so run 6's fold vector *should* equal run 4's. A named check asserts it element-wise; if
   it differs, `MODEL_HELDOUT_FOLDS` does not mean what run 6 thinks and the arm is unscoreable.

**One thing that turns out clean:** run 4 trained on **in-scope perturbations only**
(`ingest.py:7`), so §2's weaker-strata basis genes were never in its training set. The basis is
out-of-distribution for the fine-tuned model exactly as for the three frozen ones.

### Environment

Run 4's pinned stack is torch 2.5.1+cu121 / numpy 1.26.4 / `scgpt 0.2.4 --no-deps`; run 5's
`scgpt` env is torch 2.9.1+cu128. **Verify the checkpoint loads and forwards in run 5's env
before assuming one env covers both**, and add a run-4-matched env under `envs/` if it does not.

### Not added, but one line away

`own_frozen` (scGPT's own table, **not** trainable) is the same fetch and another ~25 min, and is
the control that splits any gain between the readout and the table. Worth adding if the
`own_trainable` hidden arm moves at all.

---

## 7. Nulls

Four layers. The first is required by the new-arm protocol; the second and third are the ones
that bite.

1. **`<model>-hidrand-l<l>` — width-matched frozen random.** iid Gaussian in `k` per slot,
   per-slot scale matched, drawn once and stored.
   ⚠️ **A null cannot meaningfully "carry the same projection" through an orthonormal basis:** if
   `z ~ N(0, σ²I_d)` then `Wᵀz ~ N(0, σ²WᵀW) = N(0, σ²I_k)`, independent of `W`. Drawing in `d`
   and projecting would burn ~15–94 GB of RNG per block to reach a distribution available in `k`
   dimensions directly. So this null controls for **width only**; the equivalence goes in the
   docstring and the substantive machinery controls are (2) and (3).
2. **`<model>-hidperm-l<l>` — the machinery null.** The **real** projected features with the
   perturbed-gene index permuted independently per read-out gene (seeded). Preserves the marginal
   distribution, the basis, the covariance structure, the dose axis and the per-slot scales;
   destroys only the pair-specific association.
3. **`<model>-hidrp-l<l>` — frozen random basis.** Real `Δh` through a random orthonormal
   `d → k` basis. Isolates "the PCA basis is doing the work" from "`Δh` carries pair-specific
   signal in any `k`-dim slice". One extra matmul, no extra forward passes.
4. **Run 3's own shuffled-label draws**, free: `SHUFFLE_NULL_ARMS`, `N_SHUFFLE_DRAWS = 5`,
   permuted **within the training perturbations only**. An arm counts only if it clears the
   **largest** draw in **its own cell**, against fresh draws — never a number copied from run 3.

`null_randpair` (11 dims) stays in the arm list as the reference the `k = 1` rung is
width-matched against.

### ⚠️ The permutation null relocates TranscriptFormer's undefined self entry — **unscoreable as written**

Found 2026-08-31 by the stage-0 pilot, which **failed on it**:

```
[hepg2/full/grouped] transcriptformer-normperm-l12_resp/mlp  d=5  resid_w_dis=nan (90.3s)
torch._C._LinAlgError: torch.linalg.solve: the input matrix is singular
```

The real TF arm scored fine in the same cell (`transcriptformer-norm-l12_resp/mlp` = 0.0263). The
difference is **where the `nan` sits, not whether there is one.**

TF never reads a gene's count at its own position, so entry `(p, self_col(p))` is `nan` — §5's
policy, and deliberately kept as `nan` because a measured absence is not a zero. That is safe in
the real arm for one reason only: every cell sets `weights[..., self_cols] = 0`, and the `nan` is
exactly on the masked entry. But `nulls.hidperm` / `normperm` is a pure gather along axis 0 *within
each read-out column*:

```python
dst[:, :, g] = src[perms[:, g], :, g]
```

so the `nan` stays in column `self_col(p)` and moves to a **different row** — a row whose own self
mask covers a different column. Nothing masks it any more. The feature matrix carries a live `nan`,
the MLP returns `nan`, and ridge's Gram matrix goes singular.

**This is TranscriptFormer-only.** The smoke measured `self_position_defined=True` and 40 of 40
self positions finite for scGPT, scPRINT and both fine-tuned keys; TF alone is 0 of 40. So the
non-TF wave is unaffected, and `normrand`/`hidrand` are unaffected (a fresh Gaussian has no `nan`).

**It will hit `hidperm` for the vector arm too**, and for the same reason: `extract_line` stores the
full `(n_readout, k)` slot including the self row, so TF's self row projects to `nan` and `hidperm`
relocates it identically. **TF's `hidperm` must be fixed before TF is extracted or scored.**

The fix must not fabricate a value. The option that preserves what the null is for — destroy the
pair association, change nothing else — is to permute within each column **over the defined rows
only**, leaving the single undefined entry on the self column where the real arm has it, so the
existing mask protects the null exactly as it protects the arm. `nan`-to-zero is not available: it
is the substitution §5 exists to forbid, and it would let the discrimination score ride on a
fabricated column. Sanitising without the exclusion already raises elsewhere by design
(`config.PDS_SANITISED_NOTE`).

Until then, **no TF `*perm` row may be quoted**, and a TF row read against `hidrand`/`hidrp` alone
is read against width and basis controls only — not against the machinery control.

---

## 8. The sweep and the pre-registration

Every constant below lives in `src/scfm_hidarm/config.py`, written before any arm is scored and
echoed verbatim into `qc.json`.

**Pre-registered primary cell:** `residual__weighted__disattenuated` at
`(layer = last block, k = 64 — re-registered from 16 on 2026-08-31, §2 — scope = full, split = grouped, translator = mlp,
anchor = mean_control)`. Arms are **ordered** by the post-hoc secondary
`centered__weighted__disattenuated`. Both reported for every arm; neither confirms the other
(`docs/INDEX.md` §5, `docs/METRICS.md` §3.6 — this reversed at cp-23).

**Splits `grouped,random`, translators `mlp,ridge`, scope `full` only.** `make_folds` groups from
the measured effect matrix of the scope rows, independent of the feature, so a run-6 arm inherits
run 3's exact folds provided the scope rows are unchanged.

| stage | what | gate |
|---|---|---|
| **0** | the norm arm (§4), all sweep layers, 3 models × 2 lines, + dim-5 null + `hidperm` + references | none — start here |
| **1** | primary layer, 4 models × 4 keys (`hid`, `hidrp`, `hidperm`, `hidrand`) + references ≈ 22 arms → ~88 runs | a timing pilot |
| **2** | the four remaining sweep layers for `hid`/`hidrp`, the `k ∈ {1,4,64}` rungs, the centred variant | stage 1's timing |

`MLPConfig` is `hidden=(128,64), lr=1e-3, batch_size=4096, epochs=30` and nothing overrides it,
so a `10k = 160`-dim arm trains on the same schedule as the 1024-dim `emb` arms;
`_rows_per_chunk` already budgets prediction by width.

**In-job reference arms, per run 3 §7:** `coexp`, `coexp_node`, `<model>_emb`, `null_randpair`.
**No `<model>_resp`** — run 6 emits no decoder-head reading (§5b). Reference rows stay out of the
merged run table; a separate check compares them to run 3's published rows, so no scored row is
overwritten.

**The contrasts.** Each a paired difference bootstrapped over response groups
(`groups.py:GroupTable`), each read against `metrics.min_resolvable_difference`:

| | asks |
|---|---|
| H1 | `<model>-hid` vs its own shuffled-label band + `hidperm` + `hidrand` — any pair-specific signal at all? |
| H2 | `<model>-hid` against runs 2–4's **published** head numbers — is the hidden state better than the head? **The headline**, but a **post-hoc** comparison, not an in-job paired difference: no bootstrap, no paired interval, and it crosses the raw-vs-standardised boundary (§5b). For the fine-tuned model it also crosses flag-on vs flag-off (§6). |
| H3 | `<model>-hid` vs `coexp` and vs `<model>_emb` — does it beat co-expression and the static table? |
| H4 | depth profile — rising-then-collapsing is *discarded* information, flat-and-low is *absent*. Run 5 found scPRINT non-monotonic with its best layer at 1, so read per model. |
| H5 | fine-tuned (flag on) vs frozen scGPT, **both restricted to folds 0 and 1** (§6) — a bound at the achieved resolution, and native-regime vs only-regime rather than a weights contrast. |

**Claims discipline, before any number is written down:** quote the reference arm and the null
band, never a bare score; read every score against the replicate-split noise ceiling; say which
metric cell; report scGPT separately from the ESM-derived arms; and where a difference is smaller
than what the comparison resolves, say **underpowered**, not "no effect".

---

## 9. Verification

Both kinds are required and neither substitutes for the other.

**Software tests** (`tests/`, pytest, torch-free where the logic allows):

| file | covers |
|---|---|
| `test_slots.py` | the 10 slots in the declared `(dose, pc)` order; a permuted input permutes exactly the expected columns; the dropped `zero_out` slot really is bit-identical to the dose-0 `vvp` slot |
| `test_basis.py` | `eigh` on a hand-built covariance returns the planted directions; `k = 16` is the first 16 columns of `k = 64`; a rank-1 input gives one non-zero eigenvalue; the basis genes contain **no** in-scope perturbation |
| `test_install.py` | the synthetic block round-trips through run 3's **real** `LineBundle`: `_resp_dim() == 10k`, `batch(rows, cols)` returns the exact planted numbers, `row_map` has no `-1`, the arm name resolves out of `resolve_arms` |
| `test_nulls.py` | `hidperm` preserves per-column mean/sd exactly and destroys the pairing; `hidrp`'s basis is orthonormal; `hidrand` is reproducible at its seed |
| `test_extract.py` | the driver on run 6's **own** `tests/toy.py` (not run 5's — run 5's toys expose 6 rows and cannot tell a central difference from a substitution): a toy whose hidden response is a known function of dose is recovered by arithmetic; a no-op gives **exactly** zero; layer 0's off-target delta is exactly `0.0` through the driver; a model that cannot supply hidden states is **recorded**, not zero-filled |
| `test_flag.py` | the flag tensor is identical in both passes and set at exactly one column; a toy responding **only** to the flag gives `Δh` of exactly zero across the whole sweep; a toy responding only to the value is recovered in full; a flag-off baseline is **caught**, not silently accepted |
| `tests/toy.py` | run 6's own toys: `LinearDoseToy`, `QuadraticDoseToy` (the only one that can falsify `gdt_denominator`), `IsolatedToy` (C2), `MuteToy`, and a `TransformerGenerator`-shaped `make_flag_toy`. Coefficients are dyadic and the toys return float64, so derivative recovery is asserted at `max abs diff == 0.0` rather than at a tolerance |
| `scratch/smoke_extract.py` | 8 genes, synthetic data in the real on-disk layout, basis stage → main stage → install → stage → run 3's **real** `LineBundle`, 50 named checks each printing its number. **Not a measurement of any model** |

**State, 2026-08-31: `pytest tests -q` → 136 passed; `scratch/smoke_extract.py` → 50 of 50 checks
passed.**

⚠️ **Two gaps the tests exposed, neither fixed yet.** Recorded here because a decision absent from
the record is as much a finding as one that contradicts it.

1. **There is no CLI path from a run-6 extraction to an installed block.** `extract_line` writes
   `vvp`/`gdt` **packed** as `(n_pert, n_dose·k, n_readout)` (`slots.pack_block`), but
   `install.source_slots` / `install.main` accept only the unpacked `(n_pert, n_dose, k,
   n_readout)` or run 5's 3-D norms. `install.write_block` *does* take the packed layout — which
   is why `test_install.py`, `stage0.py` and the smoke all work — so the gap is in the CLI alone.
   It **fails loudly** (`"asked for k=4 but only 1 components are stored"`, or `"slot form 'gdt'
   supplies dose indices [0..19]"`), never silently, and `jobs/` has no install step yet, so
   nothing in production is mis-wired. **This must be closed before §12 item 9.**
2. **`extract_line` records C2 but never asserts it.** `diagnostics.c2_layer0_max_off_target`
   travels in every block manifest, and enforcement lives only in `verify.py` — so a
   mis-attributed hook would let a whole extraction wave complete and write blocks, and would be
   caught only afterwards. Nothing wrong is *reported*, because no number is read before
   `verify` runs; but the run's own discipline is to fail loudly on a broken scientific
   invariant, and this one does not. Smoke measurements so far are exactly `0.0` for scGPT and
   scPRINT, so a hard raise looks safe — it is deliberately **not** being added mid-wave.

Minor, recorded rather than fixed: `slots.mid_row_values(values_native, step)` ignores `step` and
`no_op_mask` passes `0.0` for it — a dead parameter, harmless. And `paths.run2_handoff` has no
environment override while `run3_bundle_root` has two, so a smoke or test that needs synthetic
handoff tables has to redirect that one lookup by other means.

Plus `scratch/smoke_extract.py` — 8 genes, synthetic data in the real on-disk layout, end to end.

**Named checks** — `Checks.add(section, name, ok, detail)` where the detail **prints the number
checked**; a missing file **fails**, never skips; output captured verbatim into
`results/verification/verify_all.txt`:

| check | asserts |
|---|---|
| C2 layer 0 | off-target `Δh` **exactly** `0.0`, per (model, line) |
| C1 self position | the perturbed gene's own `Δh` moves; **absent by construction** for TranscriptFormer, never a failure and never a zero |
| C3 no-ops | every no-op query's `Δh` is exactly zero, counted and named; TF's 20/2,620 at 64 CP10K expected |
| basis leakage | `N_BASIS_GENES` ∩ in-scope perturbations = ∅, printed as `0 of 64` |
| basis coverage | captured-energy `W_basis` vs `W_all` on the scored perturbations, plus principal angles, per (model, line, layer, k) — §2's diagnostic, never used to score |
| coverage | `row_map` has no `-1` on the `full` scope |
| block identity | run 6's `pert_index` / `readout_gene_id` match `targets.npz` row for row |
| ladder prefix | `max abs diff 0.0` between the `k = 16` block and the first 16 columns of `k = 64` — on the **stored** block and on `W[:, :16]`, never on a recomputed projection (§1) |
| reference agreement | in-job `coexp` / `coexp_node` / `<model>_emb` against run 3's published rows, difference printed |
| identity | a perfect prediction scores 1.0; a permuted input scores at chance |
| ft fold vector | run 6's grouped folds equal run 4's element-wise, printed as `524 of 524 agree` |
| ft restriction | every reported fine-tuned row carries `folds = (0, 1)` with a ceiling recomputed on that subset; a five-fold row **fails** |
| ft frozen counterpart | frozen scGPT is reported at the same two folds; a fine-tuned score without one **fails** |
| ft flag constant | the flag is set on `perturbed_col` in **both** passes — asserted element-wise, and by C2 still reading exactly `0.0` |
| ft flag recorded | the manifest records `perturbation_flags: set on the perturbed gene in both passes`, contradicting `configure`'s inherited note, and names the estimator as neither of run 4's families |
| ft own null band | the fine-tuned arm's shuffled draws are its **own**; a row scored against run 4's flag-family band or the frozen models' band **fails** |
| no head output | run 6 writes **no** decoder-head block and no `*_resp` arm of its own; a head-derived array in the output tree **fails** (§5b) |
| ft execution | `SCFM_FT_RUN_SUFFIX` recorded verbatim; unset **fails** |

⚠️ **A constant prediction scores well above zero on the pre-registered cell** — 0.2 on one metric
cell and undefined on another (`run3/tests/test_score_run.py`). Every arm is read against fresh
shuffled draws in **its own cell**, never against zero.

Plus a `claims_manifest.json` on run 3's pattern, re-checkable by `scratch/verify_claims.py`.

---

## 10. Dependencies — **nothing is blocking**

Everything run 6 needs is on local disk as of 2026-08-31. Both artifact-store fetches are
resolved or dropped.

**Fetch hygiene, if anything more is ever needed (learned the hard way, §10b):** `curl -fL` —
`-f` so an HTTP error body is never written to the output path, `-L` so a redirect is followed —
into a staging path, then verify *content* (not just size) and only then move into place. Write
under `/biodata`, never into the repo.

### 10a. Run 2's response blocks — ❌ **NOT fetched, and not needed**

Run 6 emits no decoder-head reading (§5b), so it has no `<model>_resp` arm and nothing in it
reads run 2's shipped response blocks. The head results are settled; the comparison is made
afterwards against runs 2–4 as published.

The local bundle at `/biodata/nyanovsky/projects/scfm-pert-fail-modes/run4/bundle` is a complete
run-3-format bundle (`targets.npz`, `coexpr.npz` incl. `node`, `emb__*.npz`, `reference.json`,
`MANIFEST.json`, 276 MB) and has **no `features/` directory** — which is now fine, because run 6
installs its own blocks there and needs no others.

Two questions I had expected these files to settle are answerable **from local code instead**:

- **TranscriptFormer's `(p, p)` self pair.** Read run 2's TF adapter
  (`experiment-2-nygwcv/.../models/queries.py`) and run 5's `hidden.py:818-820`, which writes the
  self position as `nan`. Run 6 sets its own policy and records it; it no longer has to match a
  block it never reads.
- **The no-op correction.** `experiment-3-yga5ws/.../ingest/no_op.py` is local and readable, and
  run 6 **detects its own no-ops directly** as run 5 already does — it does not inherit run 2's
  per-(gene, value) correction factors.

If `TODO.md` §1's block-standardised `resp` arm is ever wanted, that is a **separate** piece of
work and needs its own fetch (`vvp.npz`'s `raw` key plus `query_summary.parquet`). Not run 6.

### 10b. Run 4's fine-tuned checkpoints — ✅ **DONE, all local**

**Not** under the project directory — they live at
`/biodata/nyanovsky/models/checkpoints_pearson/<line>__<arm>/`. See that directory's
`PROVENANCE.md` for sizes, md5s and the verification. `own_trainable` was fetched 2026-08-25 and
`own_frozen` plus a repaired Jurkat `own_trainable` on 2026-08-31; all four `model_best.pt` load
with `weights_only=True` and yield 176 tensors, all four md5s are distinct, and HepG2's two arms
differ in 172 of 176 tensors, so they are not the same object twice.

`esm2_trainable`, `esm2_frozen` and `rand_frozen` are **not** local. `own_trainable` also carries
`gene_table_final.npz` and `train_config.json`; `own_frozen` carries neither — worth fetching if
the frozen arms' `build_report` is wanted.

⚠️ **The 2026-08-25 fetch failed silently and stood for six days**: Jurkat's `model_best.pt` was
134 bytes of HTML `301 Moved Permanently` saved under a `.pt` name, because the download neither
followed the redirect nor failed on a non-200. `torch.load` raises something unhelpful on it, and
an existence check passes it. **Run 6's named checks must assert every checkpoint is a readable
archive with a plausible tensor count**, not that the file exists and not that it exceeds a size
threshold — a larger error body would defeat both.

### `own_frozen` is now available, so take the control

§6 left `own_frozen` out as "one line away". It is now on disk, so include it: scGPT's own table,
**not** trainable, which splits any gain in the hidden arm between **the readout** and **the
table**. Without it, a positive `own_trainable` result cannot say which of the two moved. Cost is
another ~25–30 min of extraction and it reuses every line of the `own_trainable` path.

### What `train_config.json` gives for free

It removes one planned dependency and pins one thing that was going to be probed blind:

- `model_heldout_folds: [0, 1]` **and** `preregistered_heldout_folds: [0, 1]` — so §6's
  fold-vector check has a local reference and needs no fetch.
- `arm_spec: {table: scgpt, trainable_table: true, projected: false}`, `run_name: own_trainable`
  — the arm identity without loading the checkpoint.
- `build_report` carries `n_layers`, `d_model`, `ntoken`, `table_dim`, `table_projected`,
  `load_param_prefixes`, `checkpoint_keys`, `fused_qkv_tensors_remapped`, `substitutions`,
  `scgpt_version`. **Pin the transformer stack path from this** rather than letting
  `_resolve_stack` probe the generation model blind.
- `recipe_source: "scGPT tutorials/Tutorial_Perturbation.ipynb; model class
  scgpt.model.generation_model.TransformerGenerator"` — the exact class to import.

⚠️ **`encoder.table` is `(2722, 512)` / `(3248, 512)` — the substituted table in read-out gene
order, not scGPT's 60,697-row vocabulary order.** Never compare it row-wise against
`scGPT_human/best_model.pt:encoder.embedding.weight`; that compares unrelated genes. Join through
`gene_table_final.npz:gene_id`.

⚠️ **And it records a deviation from the published recipe that bears on run 6:**
`force_perturbed_gene_in_sequence: true`, with `deviation: "the batch's own perturbed genes are
always inside the sampled 1536-gene sequence; upstream samples uniformly and can drop them"`. The
perturbed gene being in the sequence is a **precondition for the flag-on query to mean
anything** — a flag set on a gene the sequence dropped is a flag on nothing. Verify run 6's query
path honours the same guarantee, and assert it rather than assuming the adapter inherits it.

---

## 11. Layout

Package **`scfm_hidarm`** — it imports two `scfm_pfm` packages plus `scfm_probe` and `scfm_ft`,
so it cannot be `scfm_pfm`. Skeleton per `docs/INDEX.md` §2.

| module | what |
|---|---|
| `config.py` | THE pre-registration: `VALUE_GRID_CP10K`, per-model GDT step, `SWEEP_LAYERS`, `K_STORE=64`, `K_PRIMARY=16`, `K_LADDER`, `N_BASIS_GENES=64`, `BASIS_SEED`, `NULL_SEEDS`, `PRIMARY_METRIC`, `model_stem`/`run_of_stem`, `ARM_KEYS` |
| `paths.py` | ladder `SCFM_PFM6_DATA_ROOT` → platform var → local fallback. Write root `/biodata/nyanovsky/projects/scfm-pert-fail-modes/experiment-6-hidarm`. **Not** run 5's root — run 5's `analysis.discover()` globs `anchor=*` under `probe/<model>/<line>` and would ingest run 6's tree |
| `sibling.py` | run 5's `sibling.py` + `run5()` (`scfm_probe`) + `run4()` (`scfm_ft`); runs 2 and 3 keep the `scfm_pfm_r2` / `_r3` aliases |
| `basis/fit.py` | streaming second-moment accumulator, `eigh`, `W` and its provenance, plus the `W_all` captured-energy diagnostic (§2) |
| `models/hidden_ft.py` | `ScGPTPertHiddenQuery`, flag threaded into both passes, `amp=False` (§6) |
| `extract/run.py` | the driver: 16-row `query_batch`, `hidden_batch`, project, write. **Hidden states only — it never reads or writes a decoder-head value** (§5b) |
| `install.py` | the synthetic run-2-shaped block writer (pattern: run 3 `protein/install_arm.py`) |
| `stage.py` | one bundle per (model, layer, k): run 3's bundle + one `features/` block |
| `nulls.py` | `hidrand` / `hidperm` / `hidrp` block writers |
| `restrict.py` | run 4's `restrict_scores` / `restricted_ceilings`, imported, for the fine-tuned arm **and** its frozen counterpart on folds (0, 1) |
| `analysis.py` | merge, contrasts, `min_resolvable_difference`, both metric cells |
| `verify.py` | the `Checks` harness (§9) |
| `report/{figures,build,contract}.py` | figures, the page, the claims contract |

`jobs/`: `smoke.sh`, `stage0.sh`, `fit_basis.sh`, `extract.sh`, `install.sh`, `score.sh`,
`finalise.sh`. `envs/`: per-model conda envs, pinned — copy `experiment-5-hidprb/envs/README.md`,
including the scPRINT import trap.

**Heavy bytes** go to `/biodata/nyanovsky/projects/scfm-pert-fail-modes/experiment-6-hidarm/` via
the `paths.py` env override — never a hardcoded path, never in the repo.

---

## 12. Order of work

1. **Stage 0** (§4) — install the norm arm from run 5's existing `norms.npz`, stage, score,
   analyse. No GPU, no fetch. Exercises install → stage → score → analyse on real data and
   returns a real result before any extraction code exists.
2. Confirm the fine-tuned checkpoints **build and forward** (not just `torch.load`) in run 5's
   `scgpt` env; add a run-4-matched env only if they do not (§6).
3. Set run 6's own TF self-pair and no-op policies and record them — no fetch needed (§10a).
   ✅ done: both are implemented in `extract/slots.py`, `extract/run.py`, `metrics_extra.py` and
   `stage0.py`, and written into §5 above under *The two policies those facts force*.
4. `config.py` and `paths.py`, then the §9 tests, **then** the driver.
5. `jobs/smoke.sh` — 8 genes, layers 0/1/last, HepG2, all five model keys. C1–C3 on real output,
   and the §6 fold-vector check before either fine-tuned model is extracted at all.
6. Timing pilot: one model, one line, full coverage, wall clock recorded.
7. `jobs/fit_basis.sh` → `W` per (model, line, layer), provenance, leakage check, captured-energy
   diagnostic.
8. `jobs/extract.sh` → the **ten** projected blocks (5 model keys × 2 lines: scGPT, scPRINT,
   TranscriptFormer, scGPT-ft-trainable, scGPT-ft-frozen), `K_STORE = 64`, all sweep layers.
9. `jobs/install.sh` + `nulls.py` + `stage.py` → one bundle per (model, layer, k).
10. Stage 1 sweep (§8), then the §6 fold restriction over **both** fine-tuned arms and frozen
    scGPT, `verify --all`, then stage 2 gated on stage 1's timing.
11. `analysis.py` → `report/` → `claims_manifest.json` → this file's findings + `manifest.yaml`,
    and one entry each in `docs/INDEX.md` §6c/§9, `docs/METHODS.md` §6c, `docs/METRICS.md` §10c.

## Done condition

`PYTHONPATH=src pytest tests -q` green; `python -m scfm_hidarm.verify --all` exits 0 with every
check printing its number into `results/verification/verify_all.txt`;
`scratch/verify_claims.py` resolves every key in `claims_manifest.json`; and the headline
readable off `manifest.yaml:results` without opening a notebook.

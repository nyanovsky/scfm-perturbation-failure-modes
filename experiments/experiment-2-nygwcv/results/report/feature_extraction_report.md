# Response features for four single-cell foundation models over three query anchors

Run 2 of the feature-swap thread. For each of scGPT, AIDO.Cell, scPRINT and TranscriptFormer it extracts the response of every read-out gene to a change in every perturbed gene, at three query anchors and in four feature forms, together with the evidence that those numbers are numerically trustworthy. It answers no scientific question and compares no models: the comparison run swaps the feature and keeps everything else fixed.

## What a feature block is

A model's **response** at read-out gene j is the single scalar it emits for that gene given a whole-cell input vector. A **response difference** is that scalar from two forward passes that are identical except for the input value of the perturbed gene i, subtracted at gene j. The sign convention is perturbed minus baseline throughout, so a positive entry means raising gene i raises the model's prediction for gene j. Zero-out is the special case where the perturbed value is 0, reported with that same convention rather than the source paper's reversed one.

- **zero-out** — one value per gene pair, the naive in-silico perturbation.
- **virtual value perturbation** — the response difference at each of 5 virtual values (0, 1, 4, 16, 64 CP10K), concatenated.
- **gradient trajectory** — the derivative of the read-out gene's output with respect to the perturbed gene's input, at each of the same values, concatenated.
- **gene embeddings** — each model's own vocabulary embedding for the read-out genes, the pair feature the comparison run uses as its no-response baseline.

## What was produced

| model | cell line | read-out genes | perturbed genes | supplied set | supplied genes | passes per query | GPU seconds, 3 anchors |
|---|---|---|---|---|---|---|---|
| scGPT | HepG2 | 2,721 | 250 | tiled | 2,721 | 3 | 89.1 |
| scGPT | Jurkat | 3,247 | 250 | tiled | 3,247 | 4 | 114 |
| scPRINT | HepG2 | 2,721 | 250 | readout | 2,721 | 1 | 167 |
| scPRINT | Jurkat | 3,247 | 250 | readout | 3,247 | 1 | 176 |
| TranscriptFormer | HepG2 | 2,721 | 250 | readout_in_passes | 2,721 | 2 | 1.08e+03 |
| TranscriptFormer | Jurkat | 3,247 | 250 | readout_in_passes | 3,247 | 2 | 1.09e+03 |

The last column is the wall clock spent producing the three anchors' zero-out, virtual-value and gradient features for that block, on one H100. It excludes the masked block, the step sweep and the layout side checks, and it is the number to size run 3 against: the four models differ in cost by two orders of magnitude for the same query.

## The read-out set, and what the expression floor does to the ceiling

### Procedure

Run 1 called a gene expressed at a pooled control level of 0.5 CP10K and measured every replicate-split ceiling over that set. Ranking genes by absolute log fold change preferentially selects small denominators, so a low floor can seat the ceiling on genes the assay barely measured. The ceilings were recomputed at five floors (0.5, 1, 2, 5, 10 CP10K) from run 1's persisted split-half vectors, for both cell lines, both split forms and all four effect strata. The stated rule was to take the lowest floor at which the top-stratum ceiling stops rising, within a tolerance of 0.01.

Three further measurements ask whether a higher floor buys its ceiling by narrowing the read-out to the response every perturbation shares: a leave-one-out mean-effect baseline (predict each perturbation by the average of the others in its stratum, built from one replicate half and scored against the other), the median correlation between different perturbations across halves, and the share of the effect matrix's sum of squares carried by its first singular direction.

### Findings

**Figure 1: Replicate ceiling and read-out coverage against the expression floor** — Top row: median per-perturbation Pearson correlation between the two replicate halves' log2 fold-change vectors, over genes above each floor, for the two in-scope effect strata. Bottom row: how many genes survive that floor. Split form split_half_cells; the batch split agrees to within 0.002. The ceiling never stops rising, so no floor satisfies a plateau rule, and each step up costs most of the read-out set. (`figures/floor_ceiling/`)

The ceiling never stops rising inside the grid. In HepG2 the top-stratum ceiling goes from 0.7031 at the 0.5 CP10K floor to 0.9163 at 10 CP10K, and in Jurkat from 0.4347 to 0.8557, while the read-out set falls from 3,119 to 142 genes in HepG2. Part of run 1's low ceiling was therefore the floor rather than the biology, but no floor on the grid satisfies the plateau rule, and every step up costs most of the read-out set.

Extraction therefore uses the inclusive floor: 2,721 read-out genes in HepG2 and 3,247 in Jurkat. Every block stores the read-out genes in a fixed order with each gene's pooled control level, so any floor on this curve is a column mask applied at scoring time rather than a re-extraction.

**Figure 2: How much of the ceiling a stratum-average predictor already reaches** — Ceiling as in the previous figure; mean-effect baseline predicts each perturbation by the average of the other perturbations in its stratum, built from one replicate half and scored against the other; cross-perturbation is the median Pearson between different perturbations across the two halves. The baseline is nearly flat across the floor grid, so the extra correlation a higher floor buys is mostly perturbation-specific. At the inclusive floor the baseline already reaches 86% of the HepG2 S4 ceiling. (`figures/floor_shared_response/`)

The mean-effect baseline barely moves with the floor while the ceiling climbs, so the extra correlation a higher floor buys is mostly perturbation-specific. The warning is at the inclusive floor instead: a predictor that emits the stratum's average response already reaches 85.8% of the HepG2 top-stratum ceiling and 91.6% of the S3 ceiling, leaving 0.0999 and 0.0249 of correlation for everything perturbation-specific. A score at any floor should be reported as headroom over that baseline as a fraction of the ceiling's own headroom, not as a raw correlation.

## The three query anchors

### Procedure

Every feature is computed at all three anchors and the three are kept separate. Each anchor is a CP10K profile over the read-out genes; each model converts CP10K into its own input scale through one recorded monotone map, so a virtual value means the same biological level for all four.

- **uniform virtual** — every supplied gene pinned at the uniform share of a 10,000-count virtual cell, 3.6751 CP10K in HepG2 and 3.0798 CP10K in Jurkat; the paper's construction, and far off the distribution the models were trained on
- **mean control** — the pooled non-targeting control mean expression profile of this cell line, restricted to the read-out genes; the on-distribution counterpart
- **individual cells** — 4 real non-targeting control cells, each converted to CP10K so every anchor sits at one depth; responses are averaged over the cells and their spread is recorded

- AIDO.Cell, scPRINT: the input is a raw count and the query is a virtual cell of 10,000 counts, so a CP10K value is the count.
- TranscriptFormer: the same, then clipped at 30, which is the model's own training-time clip; a query above 30 CP10K saturates and that is recorded.
- scGPT: the input is a quantile bin index, and the bin edges are built once per cell line from that line's mean control profile and then held fixed, so two queries are comparable.

### Findings

**Figure 3: Size of the response at each virtual value** — Median across queries of the standard deviation of the response difference over read-out genes, in each model's own output units, which are not comparable between models (log axis). The four scales differ by orders of magnitude, which is why every block also ships a per-query standardized form. (`figures/response_by_value/`)

Two forward passes that differ only in the perturbed gene's input value are identical when that value equals the gene's own level in the anchor, so the response has to be smallest at the grid point nearest that level. It is, which is the cheapest available check that a query means what it says. At the uniform-virtual anchor, where every gene sits at the same constant (3.0798 and 3.6751 CP10K), the smallest response is at the 4 CP10K query in all 6 blocks, between 1.66 and 21.6 times smaller than the median over the other four values. At the mean-control anchor the minimum moves to the 1 CP10K query in all 6 blocks, and the perturbed genes' median pooled control level is 1.333 CP10K in HepG2 and 1.258 CP10K in Jurkat. At the individual-cells anchor it sits at 0 CP10K for scGPT, scPRINT; 1 CP10K for TranscriptFormer. Of the perturbed gene and cell pairs, 12.5% in HepG2 and 17.5% in Jurkat are at exactly zero counts, so the 0 CP10K query is already a no-op for a sizeable share of them.

A related quantity run 3 needs is how many queries the model could not see at all. When the value asked for is the value the input already carried, the two passes are the same call and the response is identically zero rather than small. It happens almost only at the 0 CP10K query, at the individual-cells anchor, where a gene that already reads zero in that cell is unchanged by being set to zero:

| model | cell line | anchor | value (CP10K) | no-op queries | fraction |
|---|---|---|---|---|---|
| scGPT | HepG2 | individual cells | 0 | 37 of 240 | 0.154 |
| scGPT | Jurkat | individual cells | 0 | 46 of 240 | 0.192 |
| scPRINT | HepG2 | individual cells | 0 | 16 of 240 | 0.067 |
| scPRINT | Jurkat | individual cells | 0 | 16 of 240 | 0.067 |
| TranscriptFormer | HepG2 | individual cells | 0 | 5 of 240 | 0.021 |
| TranscriptFormer | HepG2 | individual cells | 64 | 1 of 240 | 0.004 |
| TranscriptFormer | Jurkat | individual cells | 0 | 10 of 240 | 0.042 |

The count is model-specific because it is set by how each model encodes the input, not by the cell. Of the perturbed gene and cell pairs, 37 of 240 in HepG2 and 46 of 240 in Jurkat sit below the 0.5 CP10K read-out floor and are therefore delivered as zero. scGPT's no-op count is exactly that number in both lines, so every below-floor pair collapses into its lowest bin. scPRINT and TranscriptFormer record fewer, so for them some below-floor pairs still differ from an explicit zero. The single TranscriptFormer no-op at the 64 CP10K query is recorded and not explained: it is not that model's 30 CP10K count clip, which 9 HepG2 pairs sit at or above.

Nothing in the stored features exposes this. No stored response vector is identically zero, because the stored individual-cells block averages over the four cells and a no-op in one cell is diluted by the other three. A consumer that averages over these pairs is averaging a structural zero in with real effects, so run 3 should read the per-query record and drop them rather than treat a zero as a small response.

## The masking regime, separated from the anchor

### Procedure

Three of the four models are bidirectional and see the read-out gene's own input value while predicting that gene's output, so an unmasked response measures only what survives after the model has been told the answer. The uniform-virtual anchor removes that information as a side effect, because every gene sits at the same constant. The real-background anchors restore it, so the two anchors differ along two axes at once. To separate them, the mean-control anchor is also computed with the read-out genes masked, at each model's own convention read from its checkpoint.

| model | mask kind | mask value | rate | coverage | source |
|---|---|---|---|---|---|
| scGPT | input value | -1 | 0.25 | 3 cycles of 4 disjoint blocks | scGPT args.json: mask_value and mask_ratio; the mildest pretraining rate is used |
| scPRINT | boolean mask argument | n/a | 0.15 | 3 cycles of 7 disjoint blocks | scPrint.forward(mask=...); the value encoder zeroes the expression embedding at masked positions. The checkpoint records no pretraining mask rate, so 0.15 is used and recorded as a fallback |
| TranscriptFormer | none | n/a | n/a | not applicable | the model is autoregressive: a gene never sees its own value, so it is natively in the masked regime and a masked variant would be the same query |

Masking a random subset independently per pass would leave the unluckiest gene unmasked for about fifty passes at a 15% rate, so the mask is a seeded partition into disjoint blocks: every read-out gene is masked exactly once per cycle, at the model's own pretraining rate, and the cycle is repeated so each gene is read masked 3 times. The uniform-virtual anchor needs no masked variant because the read-out gene's own input already carries no information about its true level. TranscriptFormer needs none at all: it is autoregressive, so a gene never sees its own value and is natively in the masked regime.

### Findings

**Figure 4: Masking the read-out gene, at the same anchor** — Median absolute zero-out response at the mean control anchor, with the read-out gene's own input value visible and with it replaced by each model's own mask convention (log axis, each model in its own output units). TranscriptFormer has no masked variant: it is autoregressive, so a gene never sees its own value. Masking shrinks the response for both bidirectional models, by about a factor of two for scGPT and by an order of magnitude for scPRINT. Removing the gene's own value made the model less sensitive to the perturbed gene, not more. (`figures/masked_regime/`)

Masking the read-out gene's own input value shrinks the response at the same anchor. Median absolute zero-out response, unmasked against masked: scGPT: 0.00389 to 0.00228 in HepG2 (a factor of 0.586); 0.0036 to 0.002 in Jurkat (a factor of 0.556). scPRINT: 0.00223 to 0.000217 in HepG2 (a factor of 0.097); 0.00111 to 0.000151 in Jurkat (a factor of 0.136). The two variants are therefore materially different features at one anchor, which is what the masked block was extracted to establish, and the size of the gap differs by model rather than being a fixed offset.

The direction is worth stating plainly because it is not the naive one: taking away the gene's own value made the model less sensitive to the perturbed gene, not more. Run 2 records the effect and does not explain it. Two accounts are consistent with it, a masked prediction falling back on a population prior that no single other gene moves much, and the mask token itself being off the distribution the model was trained to read, and nothing here separates them.

## Gradient trajectories and the finite-difference step

### Procedure

A gradient-trajectory feature is a magnitude, so a step-dependent derivative would make it meaningless. Forward-mode automatic differentiation is used wherever the model's layout is one differentiable function of the input; elsewhere the derivative is a central difference whose step is selected by sweeping 6 steps on 50 perturbed genes at a 4 CP10K base value and taking the smallest step that agrees with the next larger step at median Pearson at or above 0.999 and median relative L2 difference at or below 0.1. Where no step qualifies, the block is flagged: its gradient features are usable for direction and ranking, not magnitude.

### Findings

**Figure 5: Finite-difference step stability** — Each point compares the gradient column computed at that step with the column at the next larger step, as the median Pearson correlation over 50 perturbed genes at a 4 CP10K base value. A step is selected only if it also agrees in magnitude (median relative L2 difference at or below 0.10). Where the curve sits on the threshold line the gradient is a property of the model rather than of the step; where it falls away, the block's gradient features are usable for direction and ranking but not magnitude. (`figures/step_stability/`)

12 of the 18 block and anchor pairs converge and 6 do not, and the split is by model rather than by anchor: every failure is TranscriptFormer. Its adjacent-step agreement climbs steadily without ever reaching the threshold, from 0.2441 at step 0.05 to 0.9742 at step 1, so the rule falls back on the step with the best direction agreement (1) and flags the block. The reason is visible in the size of the numbers: across its swept steps the gradient columns being differenced have a median absolute value of 1.39e-08, close enough to float noise that the difference between two nearby steps is dominated by it. Those gradient features are usable for direction and ranking and not for magnitude, and a consumer reads that verdict out of `step_selection.json` rather than inferring it.

| model | cell line | anchor | selected step | verdict | gradient source |
|---|---|---|---|---|---|
| scGPT | HepG2 | uniform virtual | 0.05 | stable in direction and magnitude | central_finite_difference(step=0.05) |
| scGPT | HepG2 | mean control | 0.02 | stable in direction and magnitude | central_finite_difference(step=0.02) |
| scGPT | HepG2 | individual cells | 0.02 | stable in direction and magnitude | central_finite_difference(step=0.02) |
| scGPT | Jurkat | uniform virtual | 0.02 | stable in direction and magnitude | central_finite_difference(step=0.02) |
| scGPT | Jurkat | mean control | 0.02 | stable in direction and magnitude | central_finite_difference(step=0.02) |
| scGPT | Jurkat | individual cells | 0.02 | stable in direction and magnitude | central_finite_difference(step=0.02) |
| scPRINT | HepG2 | uniform virtual | 0.02 | stable in direction and magnitude | forward_mode_autodiff_jvp |
| scPRINT | HepG2 | mean control | 0.02 | stable in direction and magnitude | forward_mode_autodiff_jvp |
| scPRINT | HepG2 | individual cells | 0.02 | stable in direction and magnitude | forward_mode_autodiff_jvp |
| scPRINT | Jurkat | uniform virtual | 0.02 | stable in direction and magnitude | forward_mode_autodiff_jvp |
| scPRINT | Jurkat | mean control | 0.02 | stable in direction and magnitude | forward_mode_autodiff_jvp |
| scPRINT | Jurkat | individual cells | 0.02 | stable in direction and magnitude | forward_mode_autodiff_jvp |
| TranscriptFormer | HepG2 | uniform virtual | 1 | magnitude unstable | central_finite_difference(step=1) |
| TranscriptFormer | HepG2 | mean control | 1 | magnitude unstable | central_finite_difference(step=1) |
| TranscriptFormer | HepG2 | individual cells | 1 | magnitude unstable | central_finite_difference(step=1) |
| TranscriptFormer | Jurkat | uniform virtual | 1 | magnitude unstable | central_finite_difference(step=1) |
| TranscriptFormer | Jurkat | mean control | 1 | magnitude unstable | central_finite_difference(step=1) |
| TranscriptFormer | Jurkat | individual cells | 1 | magnitude unstable | central_finite_difference(step=1) |

Where an exact derivative exists it is the feature, and the central difference at the selected step is its cross-check. scPRINT admits forward-mode autodiff in the layout it is actually run in, so the cross-check runs on the production features. scGPT does not, because its read-out set is delivered in several tiled passes; the exact derivative exists only in the untiled layout, so that comparison is a side check on a smaller gene sample rather than a statement about the production features. The medians below are taken over the base values whose central difference stays inside the input range: at a base of 0 CP10K the difference would have to cross the floor, so that value is one-sided and excluded.

| model | cell line | anchor | layout | median Pearson | median relative L2 | step |
|---|---|---|---|---|---|---|
| scGPT | HepG2 | uniform virtual | side check, 20 genes | 0.999672 | 0.0256 | 0.02 |
| scGPT | Jurkat | uniform virtual | side check, 20 genes | 0.999787 | 0.01941 | 0.02 |
| scPRINT | HepG2 | uniform virtual | production | 0.999767 | 0.02108 | 0.02 |
| scPRINT | HepG2 | mean control | production | 0.999829 | 0.01807 | 0.02 |
| scPRINT | HepG2 | individual cells | production | 0.999968 | 0.007959 | 0.02 |
| scPRINT | Jurkat | uniform virtual | production | 0.999728 | 0.02261 | 0.02 |
| scPRINT | Jurkat | mean control | production | 0.999848 | 0.01672 | 0.02 |
| scPRINT | Jurkat | individual cells | production | 0.999954 | 0.009871 | 0.02 |

## Normalization, which is not cosmetic here

### Procedure

The four models emit values on four incommensurable scales, and two of them end in a softmax across the genes supplied in the same forward pass. Three consequences are handled explicitly: the supplied gene set is fixed per model and cell line and recorded; the cross-gene mean response is measured per query and a recentred variant is emitted beside the raw one for the softmax models; and every block also carries a per-query standardized form, z-scored across read-out genes, so the comparison run can use features that are invariant to each model's output scale.

A predicted log fold change is emitted only where a defensible bridge to a linear expression scale exists: AIDO.Cell's head predicts log1p CP10K, so the bridge is expm1; scPRINT and TranscriptFormer predict a share of the cell's counts, which is proportional to CP10K. scGPT's output is a per-cell quantile bin index, which is not an expression level and has no cell-independent inverse, so the bridge is declined rather than invented.

### Findings

**Figure 6: How much of each response is redistribution** — For a model whose head ends in a softmax across the supplied genes, raising one gene's share must lower the others, so the response differences sum to about zero by construction. The bars show the median cross-gene mean divided by the cross-gene spread (log axis). scPRINT sits below the tolerance at every query, which is the signature of exact redistribution. TranscriptFormer sits below it at the low values and rises above it at the top of the grid: its read-out set is delivered in two separately normalized passes, so the shares sum to one inside a pass and not across the union. scGPT, whose head is not a softmax over the supplied genes, sits well above. (`figures/softmax_coupling/`)

The redistribution identity holds where the head is a softmax over the supplied genes and not otherwise. Ratio of the absolute cross-gene mean response to its cross-gene spread, over every query in both cell lines: scGPT median 0.161, worst 1.26; scPRINT median 0.0226, worst 0.0725; TranscriptFormer median 0.0444, worst 0.243.

TranscriptFormer misses the 0.05 tolerance on 4 of its 6 anchor and cell-line pairs, at 0.09089, 0.063, 0.06497, 0.05795. The cause is the layout rather than the head. Its read-out set does not fit in one pass, so it is delivered in two, each normalized by that pass's own total; the shares sum to one inside a pass and not across the union of the two, which leaves exactly this kind of small residual offset. Every block ships a recentred variant beside the raw one, and the recentred variant removes it. A consumer that cares about the identity should read `recentred`.

**Figure 7: The four output scales, per feature form** — Median and 99th percentile of the absolute raw feature value at the mean control anchor, in each model's own output units (log axis). Circles are medians, diamonds 99th percentiles; both cell lines are shown. Raw features are not comparable across models, which is what the standardized and, where defensible, predicted log fold-change forms are for. (`figures/feature_scale/`)

The four raw scales are not comparable, and the gap is not small. Median absolute zero-out response at the mean-control anchor: scGPT 0.0036 to 0.00389; scPRINT 0.00111 to 0.00223; TranscriptFormer 2.09e-06 to 3.14e-06, about 1,858-fold between the extremes, which reflects each head's units rather than anything biological. The comparison run should read the standardized form by default and treat a raw cross-model magnitude comparison as meaningless.

## Two architectural constraints that shape the query

### scGPT: how wide the supplied set is

scGPT was trained at a maximum sequence length of 1,200 but carries no positional encoding over genes, so a longer input is architecturally admissible; run 1 supplied the whole 7,444-gene universe and recorded it as an extrapolation. Run 2 measured what the width changes, on 40 common perturbed genes at the mean control anchor, comparing the expressed read-out width (2,721 genes), the full universe (7,444) and tiles of at most 1,200 sharing a fixed 256-gene prefix.

| comparison | median Pearson | median relative L2 difference |
|---|---|---|
| readout vs universe | 0.9686 | 0.81 |
| readout vs tiled | 0.7374 | 0.9382 |
| universe vs tiled | 0.7844 | 1.122 |

**Figure 8: scGPT: what the supplied-set width changes** — Zero-out response differences for the same perturbed genes at the mean control anchor, computed with three supplied widths: the expressed read-out set, run 1's full 7,444-gene universe, and tiles of at most 1,200 genes sharing a fixed prefix. A relative L2 difference of 0 means identical vectors, 2 means opposite. Directions largely agree and magnitudes do not, so the width is a measurement choice rather than a detail; the production sweep uses the tiled width, which is the one inside scGPT's training length. (`figures/scgpt_context/`)

The three widths do not agree closely, so the production sweep uses the tiled width, the one inside scGPT's training length. The cost is that a tiled query is several forward passes rather than one differentiable function of the input, so scGPT's production gradients are central differences at the selected step; the exact forward-mode derivative is available in the untiled layout and was validated against a central difference there.

### TranscriptFormer: ordering

TranscriptFormer is autoregressive with a causal mask, so gene j responds to gene i only when i precedes it. The perturbed gene is placed at position 0 of every pass, which makes every read-out gene measurable; the cost is that the context differs between queries by that one gene. Under a single fixed ordering 50% of ordered pairs would be structurally unmeasurable, which is a property of the model rather than of this run. Each pass opens with the same 256-gene prefix and its shares are normalized by that pass's own total, so values from different passes are comparable.

## Read-out order alignment

### Procedure

The failure that would silently invalidate everything is a gene-index misalignment between a model's vocabulary order and the read-out gene order. The pre-registered guard reads the response between the lowest positive value on the grid and the highest and asks that the perturbed gene's own entry be the largest response in the vector, whatever its sign; the plan's sign version asks instead that the gene's own predicted output move the expected way. Each is repeated with the perturbed gene's own read-out column relabelled at random, which gives the statistic its null.

The pre-registered guard turns out to have no power for an autoregressive model. With the perturbed gene placed first, TranscriptFormer never conditions on its own value, so its own entry is never the largest response and the permuted control is zero as well: test and null sit together at the floor and the comparison says nothing. Two statistics that keep their power there were added: the median rank of the own entry within the response vector, whose null is the middle of the vector, and the sign rate, whose null is a coin flip. A misalignment would drive all three statistics to their nulls at once, so one wide separation is enough to rule it out.

### Findings

**Figure 9: Read-out order alignment, each statistic against its permuted null** — At the mean control anchor, over the perturbed genes sampled in each block. Each statistic is shown beside the same statistic computed after relabelling the perturbed gene's own read-out column at random (the pale bar). 'Own entry largest' is the pre-registered guard; 'own entry high in the vector' is one minus the median rank of the own entry as a fraction of the vector; 'own output moves as expected' is the sign reading. scPRINT saturates the pre-registered guard on both lines. scGPT misses its 0.95 threshold while its own entry still ranks first for the median gene. TranscriptFormer scores exactly zero on that guard, and so does its null: with the perturbed gene placed first, an autoregressive model never conditions on its own value, so the guard cannot have power there. The other two statistics do, and TranscriptFormer separates from its null on at least one of them at every anchor. (`figures/wiring_checks/`)

| model | cell line | anchor | perturbed genes sampled | own entry largest (null) | median self-rank (null) | own output moves as expected (null) | separated on |
|---|---|---|---|---|---|---|---|
| scGPT | HepG2 | uniform virtual | 250 | 0.832 (0.000) | 0 of 2,721 (1,401.5) | 0.884 (0.960) | self_rank |
| scGPT | HepG2 | mean control | 250 | 0.900 (0.000) | 0 of 2,721 (1,451.5) | 0.944 (0.608) | self_rank, sign |
| scGPT | HepG2 | individual cells | 60 | 0.850 (0.000) | 0 of 2,721 (1,354.5) | 0.967 (0.683) | self_rank |
| scGPT | Jurkat | uniform virtual | 250 | 0.920 (0.000) | 0 of 3,247 (1,560.5) | 0.992 (0.924) | self_rank |
| scGPT | Jurkat | mean control | 250 | 0.932 (0.000) | 0 of 3,247 (1,539.5) | 0.816 (0.652) | self_rank |
| scGPT | Jurkat | individual cells | 60 | 0.933 (0.000) | 0 of 3,247 (1,669.5) | 0.967 (0.633) | self_rank, sign |
| scPRINT | HepG2 | uniform virtual | 250 | 1.000 (0.000) | 0 of 2,721 (1,316) | 1.000 (0.468) | own_largest, self_rank, sign |
| scPRINT | HepG2 | mean control | 250 | 1.000 (0.000) | 0 of 2,721 (1,312) | 1.000 (0.636) | own_largest, self_rank, sign |
| scPRINT | HepG2 | individual cells | 60 | 1.000 (0.000) | 0 of 2,721 (1,610.5) | 1.000 (0.567) | own_largest, self_rank, sign |
| scPRINT | Jurkat | uniform virtual | 250 | 1.000 (0.000) | 0 of 3,247 (1,626) | 1.000 (0.508) | own_largest, self_rank, sign |
| scPRINT | Jurkat | mean control | 250 | 1.000 (0.000) | 0 of 3,247 (1,609) | 1.000 (0.728) | own_largest, self_rank |
| scPRINT | Jurkat | individual cells | 60 | 1.000 (0.000) | 0 of 3,247 (1,478.5) | 1.000 (0.783) | own_largest, self_rank |
| TranscriptFormer | HepG2 | uniform virtual | 250 | 0.000 (0.000) | 909 of 2,721 (1,557) | 0.936 (0.636) | sign |
| TranscriptFormer | HepG2 | mean control | 250 | 0.000 (0.004) | 66.5 of 2,721 (1,474.5) | 0.904 (0.736) | self_rank |
| TranscriptFormer | HepG2 | individual cells | 60 | 0.000 (0.000) | 56.5 of 2,721 (1,507) | 0.850 (0.750) | self_rank |
| TranscriptFormer | Jurkat | uniform virtual | 250 | 0.000 (0.000) | 614.5 of 3,247 (1,768.5) | 0.976 (0.600) | sign |
| TranscriptFormer | Jurkat | mean control | 250 | 0.000 (0.000) | 80.5 of 3,247 (1,616) | 0.976 (0.840) | self_rank |
| TranscriptFormer | Jurkat | individual cells | 60 | 0.000 (0.000) | 75 of 3,247 (1,616.5) | 0.967 (0.783) | self_rank |

Each cell gives the observed statistic with its permuted null in parentheses. The last column names the statistics that separate widely enough to establish alignment for that block. All 18 block and anchor pairs separate from their null on at least one statistic.

The pre-registered guard is reported as it stands and is not retro-fitted to the result. It is met by scPRINT at every anchor in both cell lines and it is not met by scGPT or TranscriptFormer, so for those two models the alignment conclusion rests on the permutation-controlled statistics rather than on the threshold written before the data existed. What the guard would have caught the replacements also catch: a misalignment relabels the own entry at random, which is exactly the permutation, and it would drive all three statistics to their nulls at once.

## What run 3 reads

A feature block is addressable by (model, anchor, form, cell line) without reading any extraction code:

```
features/<model>/<cell_line>/
  manifest.json                     anchors, grids, steps, mask spec, substitutions,
                                    the model's response contract
  readout_genes.parquet             read-out order, model positions, control level
  perturbed_genes.parquet           perturbed order, stratum, subset flags
  gene_embeddings.npz               (read-out genes, d) vocabulary embedding
  step_sweep.parquet                adjacent-step agreement at every swept step
  step_selection.json               the selected step and the rule that chose it
  anchor=<anchor>/zero_out.npz      (perturbed, read-out)
  anchor=<anchor>/vvp.npz           (perturbed, values, read-out)
  anchor=<anchor>/gdt.npz           (perturbed, base values, read-out)
  anchor=<anchor>/query_summary.parquet   per-query diagnostics
  anchor=mean_control_masked/       the masked counterpart, where it applies

```

Each npz carries `raw`, `standardized`, `recentred` where the model's head ends in a softmax, and `log2fc` where a scale bridge exists, plus the perturbed and read-out gene identifiers and the value grid, so a block is self-describing.

## Verification

Verification runs at two levels. Every block is checked inside a job that mounts the volume the extraction wrote, against that block's own npz files rather than against a summary: 272 checks over 6 blocks, 256 passed. The roll-up then reads the 6 per-block records together and adds coverage and alignment checks: 30 checks, 25 passed. Both outputs ship with the report, and the failures are listed here rather than summarised away.

| failing check | blocks | what it means |
|---|---|---|
| softmax_coupling | 4 anchor and block pairs in transcriptformer/hepg2, transcriptformer/jurkat | the redistribution identity, missed by a small margin because the read-out set is delivered in two separately normalized passes. The recentred variant that ships in every block removes the offset. |
| wiring_magnitude | 12 anchor and block pairs in scgpt/hepg2, scgpt/jurkat, transcriptformer/hepg2, transcriptformer/jurkat | the pre-registered 95% own-entry-largest guard. Reported as it stands; the permutation-controlled statistics in the wiring section establish alignment for these blocks instead. |

The roll-up's own failures, which are the per-block failures restated plus anything missing from the tree:

| check | detail |
|---|---|
| coverage/model_x_cell_line | 6/8 blocks present; missing [('aido_cell', 'hepg2'), ('aido_cell', 'jurkat')] |
| block_checks/scgpt/hepg2 | 38 block-level checks, 3 failed: ['uniform_virtual/wiring_magnitude', 'mean_control/wiring_magnitude', 'individual_cells/wiring_magnitude'] |
| block_checks/scgpt/jurkat | 38 block-level checks, 3 failed: ['uniform_virtual/wiring_magnitude', 'mean_control/wiring_magnitude', 'individual_cells/wiring_magnitude'] |
| block_checks/transcriptformer/hepg2 | 45 block-level checks, 4 failed: ['uniform_virtual/softmax_coupling', 'uniform_virtual/wiring_magnitude', 'mean_control/wiring_magnitude', 'individual_cells/wiring_magnitude'] |
| block_checks/transcriptformer/jurkat | 45 block-level checks, 6 failed: ['individual_cells/softmax_coupling', 'mean_control/softmax_coupling', 'uniform_virtual/softmax_coupling', 'uniform_virtual/wiring_magnitude', 'mean_control/wiring_magnitude', 'individual_cells/wiring_magnitude'] |

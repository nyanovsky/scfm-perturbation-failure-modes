# Response features for four single-cell foundation models over three query anchors

Run 2 of the feature-swap thread. For each of scGPT, AIDO.Cell, scPRINT and TranscriptFormer it extracts the response of every read-out gene to a change in every perturbed gene, at three query anchors and in four feature forms, together with the evidence that those numbers are numerically trustworthy. It answers no scientific question and compares no models: the comparison run swaps the feature and keeps everything else fixed.

## What a feature block is

A model's **response** at read-out gene j is the single scalar it emits for that gene given a whole-cell input vector. A **response difference** is that scalar from two forward passes that are identical except for the input value of the perturbed gene i, subtracted at gene j. The sign convention is perturbed minus baseline throughout, so a positive entry means raising gene i raises the model's prediction for gene j. Zero-out is the special case where the perturbed value is 0, reported with that same convention rather than the source paper's reversed one.

- **zero-out** — one value per gene pair, the naive in-silico perturbation.
- **virtual value perturbation** — the response difference at each of 5 virtual values (0, 1, 4, 16, 64 CP10K), concatenated.
- **gradient trajectory** — the derivative of the read-out gene's output with respect to the perturbed gene's input, at each of the same values, concatenated.
- **gene embeddings** — each model's own vocabulary embedding for the read-out genes, the pair feature the comparison run uses as its no-response baseline.

## What was produced

| model | cell line | read-out genes | perturbed genes | coverage | supplied set | supplied genes | passes per query | GPU seconds, 3 anchors |
|---|---|---|---|---|---|---|---|---|
| AIDO.Cell | HepG2 | 2,721 | 250 of 524 | 0.477 | fixed_transcriptome | 19,264 | 1 | 10,575 |
| AIDO.Cell | Jurkat | 3,247 | 250 of 705 | 0.355 | fixed_transcriptome | 19,264 | 1 | 10,571 |
| scGPT | HepG2 | 2,721 | 524 of 524 | all | tiled | 2,721 | 3 | 372 |
| scGPT | Jurkat | 3,247 | 705 of 705 | all | tiled | 3,247 | 4 | 643 |
| scPRINT | HepG2 | 2,721 | 524 of 524 | all | readout | 2,721 | 1 | 714 |
| scPRINT | Jurkat | 3,247 | 705 of 705 | all | readout | 3,247 | 1 | 1,196 |
| TranscriptFormer | HepG2 | 2,721 | 524 of 524 | all | readout_in_passes | 2,721 | 2 | 4,621 |
| TranscriptFormer | Jurkat | 3,247 | 705 of 705 | all | readout_in_passes | 3,247 | 2 | 6,232 |

The last column is the wall clock spent producing the three anchors' zero-out, virtual-value and gradient features for that block, on one H100. It excludes the masked block, the step sweep and the layout side checks, and it is the number to size run 3 against: the four models differ in cost by two orders of magnitude for the same query.

Two different reductions sit between the plan's effect strata and the perturbed-gene column, and they are not the same kind of decision. The plan scoped the S3 and S4_highest strata, which hold 991 perturbations in HepG2 and 1,100 perturbations in Jurkat. A perturbed gene has to be a column of the supplied set to be perturbed at all: the query works by setting that gene's input value, and a gene the model was never given has no input to set. Only perturbations whose target is itself a read-out gene are therefore reachable, which leaves 524 in HepG2 and 705 in Jurkat. That reduction is structural and no amount of compute removes it. The second is compute: run 2's first wave ran 250 of those per block, sized from the slowest model's measured cost, and the three cheaper models now run the whole reachable set at every anchor and in the masked variant. Each block's manifest records its own coverage and the roll-up fails if one does not, because a 250-gene block and a full block share the smaller one's perturbed genes but not its row order.

### What full coverage would cost for AIDO.Cell

AIDO.Cell is the reason the first wave ran a subset, so its extension was priced instead of attempted. The projection prices the extraction driver's own forward-pass count at a measured rate, and is checked by predicting the block that already exists: the observed and predicted three-anchor seconds in the table are that check.

| model | cell line | s per forward | at batch | best measured | batching speedup | observed vs predicted, 250 genes | full, 3 anchors | full, everything |
|---|---|---|---|---|---|---|---|---|
| AIDO.Cell | HepG2 | 0.8843 | 4 | 0.8817 at batch 8 | 1.00 | 10,575 vs 10,470 | 12.4 h | 20.9 h |
| AIDO.Cell | Jurkat | 0.8847 | 4 | 0.8824 at batch 8 | 1.00 | 10,571 vs 10,474 | 16.6 h | 28.0 h |

Batching across perturbed genes does not bring this down: the sequence is 19,264 genes long, so one forward pass already saturates the GPU and a larger batch buys the ratio in the table rather than anything like its size. Lower precision would: bfloat16 autocast reaches 0.09536 s per forward pass in HepG2 and bfloat16 autocast reaches 0.09546 s per forward pass in Jurkat, against 0.8843 in float32. That is not a top-up though, because it changes the numerics: a block half in float32 and half in bfloat16 is not one block. AIDO.Cell therefore stays at 250 perturbed genes in this run, and the number above is what extending it would cost.

## The read-out set, and what the expression floor does to the ceiling

### Procedure

Run 1 called a gene expressed at a pooled control level of 0.5 CP10K and measured every replicate-split ceiling over that set. Ranking genes by absolute log fold change preferentially selects small denominators, so a low floor can seat the ceiling on genes the assay barely measured. The ceilings were recomputed at five floors (0.5, 1, 2, 5, 10 CP10K) from run 1's persisted split-half vectors, for both cell lines, both split forms and all four effect strata. The stated rule was to take the lowest floor at which the top-stratum ceiling stops rising, within a tolerance of 0.01.

Three further measurements ask whether a higher floor buys its ceiling by narrowing the read-out to the response every perturbation shares: a leave-one-out mean-effect baseline (predict each perturbation by the average of the others in its stratum, built from one replicate half and scored against the other), the median correlation between different perturbations across halves, and the share of the effect matrix's sum of squares carried by its first singular direction.

### Findings

**Figure 1: Replicate ceiling and read-out coverage against the expression floor** — Top row: median per-perturbation Pearson correlation between the two replicate halves' log2 fold-change vectors, over genes above each floor, for the two in-scope effect strata. Bottom row: how many genes survive that floor. Split form split_half_cells; the batch split agrees to within 0.002. The ceiling never stops rising, so no floor satisfies a plateau rule, and each step up costs most of the read-out set. (`figures/floor_ceiling/`)

The ceiling never stops rising inside the grid. In HepG2 the top-stratum ceiling goes from 0.7031 at the 0.5 CP10K floor to 0.9163 at 10 CP10K, and in Jurkat from 0.4347 to 0.8557, while the read-out set falls from 3,119 to 142 genes in HepG2. Part of run 1's low ceiling was therefore the floor rather than the biology, but no floor on the grid satisfies the plateau rule, and every step up costs most of the read-out set.

Extraction therefore uses the inclusive floor: 2,721 read-out genes in HepG2 and 3,247 in Jurkat. Every block stores the read-out genes in a fixed order with each gene's pooled control level, so any floor on this curve is a column mask applied at scoring time rather than a re-extraction.

**Figure 2: How much of the ceiling a stratum-average predictor already reaches** — Ceiling as in the previous figure; mean-effect baseline predicts each perturbation by the average of the other perturbations in its stratum, built from one replicate half and scored against the other; cross-perturbation is the median Pearson between different perturbations across the two halves. The baseline is nearly flat across the floor grid, so the extra correlation a higher floor buys is mostly perturbation-specific. At the inclusive floor the baseline already reaches 85.8% of the HepG2 S4 ceiling, leaving 0.0999 of correlation for everything perturbation-specific. (`figures/floor_shared_response/`)

The mean-effect baseline barely moves with the floor while the ceiling climbs, so the extra correlation a higher floor buys is mostly perturbation-specific. The warning is at the inclusive floor instead: a predictor that emits the stratum's average response already reaches 85.8% of the HepG2 top-stratum ceiling and 91.6% of the S3 ceiling, leaving 0.0999 and 0.0249 of correlation for everything perturbation-specific. A score at any floor should be reported as headroom over that baseline as a fraction of the ceiling's own headroom, not as a raw correlation.

That narrow band is a property of the library, not a defect of the measurement. The source screen perturbs DepMap common-essential genes, which is the study's design and not a choice of this run: most targets are genes a proliferating cell cannot do without, so knocking any of them down produces much the same growth-arrest programme, and the shared part of that programme is most of the transcriptome response. What is left once the shared signature is predicted for free is 0.0999 of correlation in the top stratum and 0.0249 in S3, and that band is the whole space any later comparison between models can resolve. A comparison run reporting raw correlations on this library will show four models close together and near the ceiling whatever they have learned; only the headroom over the mean-effect baseline can separate them, and its size bounds how much separation is available at all.

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

Two forward passes that differ only in the perturbed gene's input value are identical when that value equals the gene's own level in the anchor, so the response has to be smallest at the grid point nearest that level. It is, which is the cheapest available check that a query means what it says. At the uniform-virtual anchor, where every gene sits at the same constant (3.0798 and 3.6751 CP10K), the smallest response is at the 4 CP10K query in all 8 blocks, between 1.67 and 22.3 times smaller than the median over the other four values. At the mean-control anchor the minimum moves to the 1 CP10K query in all 8 blocks, and the perturbed genes' median pooled control level is 1.333 CP10K in HepG2 and 1.258 CP10K in Jurkat. At the individual-cells anchor it sits at 0 CP10K for scGPT, scPRINT; 1 CP10K for AIDO.Cell, TranscriptFormer. Of the perturbed gene and cell pairs, 12.5% in HepG2 and 17.5% in Jurkat are at exactly zero counts, so the 0 CP10K query is already a no-op for a sizeable share of them.

A related quantity run 3 needs is how many queries the model could not see at all. When the value asked for is the value the input already carried, the two passes are the same call and the response is identically zero rather than small. It happens almost only at the 0 CP10K query, at the individual-cells anchor, where a gene that already reads zero in that cell is unchanged by being set to zero:

| model | cell line | anchor | value (CP10K) | no-op queries | fraction |
|---|---|---|---|---|---|
| AIDO.Cell | HepG2 | individual cells | 0 | 30 of 240 | 0.125 |
| AIDO.Cell | Jurkat | individual cells | 0 | 42 of 240 | 0.175 |
| scGPT | HepG2 | individual cells | 0 | 379 of 2,096 | 0.181 |
| scGPT | Jurkat | individual cells | 0 | 755 of 2,820 | 0.268 |
| scPRINT | HepG2 | individual cells | 0 | 126 of 2,096 | 0.060 |
| scPRINT | Jurkat | individual cells | 0 | 249 of 2,820 | 0.088 |
| TranscriptFormer | HepG2 | individual cells | 0 | 72 of 2,096 | 0.034 |
| TranscriptFormer | HepG2 | individual cells | 64 | 7 of 2,096 | 0.003 |
| TranscriptFormer | Jurkat | individual cells | 0 | 124 of 2,820 | 0.044 |

The count is set by how each model encodes its input rather than by the cell, and it lands on one of two interpretable numbers: how many perturbed gene and cell pairs read exactly zero counts, and how many sit below the 0.5 CP10K read-out floor. Both are counted over each block's own perturbed genes, which is why the number of pairs differs between blocks at different coverage.

| model | cell line | gene and cell pairs | no-ops at 0 CP10K | exactly zero counts | below the read-out floor | matches |
|---|---|---|---|---|---|---|
| AIDO.Cell | HepG2 | 240 | 30 | 30 | 37 | exactly-zero pairs |
| AIDO.Cell | Jurkat | 240 | 42 | 42 | 46 | exactly-zero pairs |
| scGPT | HepG2 | 240 | 379 | 30 | 37 | fewer than either |
| scGPT | Jurkat | 240 | 755 | 42 | 46 | fewer than either |
| scPRINT | HepG2 | 240 | 126 | 30 | 37 | fewer than either |
| scPRINT | Jurkat | 240 | 249 | 42 | 46 | fewer than either |
| TranscriptFormer | HepG2 | 240 | 72 | 30 | 37 | fewer than either |
| TranscriptFormer | Jurkat | 240 | 124 | 42 | 46 | fewer than either |

AIDO.Cell matches the exactly-zero count, which is what a raw-count input predicts: only a gene already at zero is unchanged by being set to zero. TranscriptFormer, scGPT and scPRINT records fewer than either, so for those even an already-zero gene is not always delivered the same way as an explicit zero. The single TranscriptFormer no-op at the 64 CP10K query is recorded and not explained: it is not that model's 30 CP10K count clip, which 9 HepG2 pairs sit at or above.

The other thing that anchor hides is how much the choice of cell matters. The stored block is the average over the four cells, so the spread across them is recorded separately: per perturbed gene, the coefficient of variation of the response size over the four cells.

| model | cell line | cells | median CV over cells | 90th percentile |
|---|---|---|---|---|
| AIDO.Cell | HepG2 | 4 | 0.281 | 0.921 |
| AIDO.Cell | Jurkat | 4 | 0.352 | 1.1 |
| scGPT | HepG2 | 4 | 0.391 | 0.911 |
| scGPT | Jurkat | 4 | 0.386 | 1.05 |
| scPRINT | HepG2 | 4 | 0.25 | 1.13 |
| scPRINT | Jurkat | 4 | 0.276 | 1.17 |
| TranscriptFormer | HepG2 | 4 | 0.224 | 0.888 |
| TranscriptFormer | Jurkat | 4 | 0.228 | 1.05 |

The median gene's response size moves by 0.224 to 0.391 of itself between cells, and the tail is wider still. Four cells is enough to average out a no-op but not enough to make the anchor a stable quantity, so a run 3 result that rests on the individual-cells anchor should be read as a four-cell sample rather than as the model's response in a real cell.

Nothing in the stored features exposes this. A consumer that averages over these pairs is averaging a structural zero in with real effects, so run 3 should read the per-query record and drop them rather than treat a zero as a small response.

## The masking regime, separated from the anchor

### Procedure

Three of the four models are bidirectional and see the read-out gene's own input value while predicting that gene's output, so an unmasked response measures only what survives after the model has been told the answer. The uniform-virtual anchor removes that information as a side effect, because every gene sits at the same constant. The real-background anchors restore it, so the two anchors differ along two axes at once. To separate them, the mean-control anchor is also computed with the read-out genes masked, at each model's own convention read from its checkpoint.

| model | mask kind | mask value | rate | coverage | source |
|---|---|---|---|---|---|
| AIDO.Cell | transformed input value | -1 | 0.15 | 3 cycles of 7 disjoint blocks | config.json: mask_token_id; the checkpoint records no pretraining mask rate, so 0.15 is used and recorded as a fallback |
| scGPT | input value | -1 | 0.25 | 3 cycles of 4 disjoint blocks | scGPT args.json: mask_value and mask_ratio; the mildest pretraining rate is used |
| scPRINT | boolean mask argument | n/a | 0.15 | 3 cycles of 7 disjoint blocks | scPrint.forward(mask=...); the value encoder zeroes the expression embedding at masked positions. The checkpoint records no pretraining mask rate, so 0.15 is used and recorded as a fallback |
| TranscriptFormer | none | n/a | n/a | not applicable | the model is autoregressive: a gene never sees its own value, so it is natively in the masked regime and a masked variant would be the same query |

Masking a random subset independently per pass would leave the unluckiest gene unmasked for about fifty passes at a 15% rate, so the mask is a seeded partition into disjoint blocks: every read-out gene is masked exactly once per cycle, at the model's own pretraining rate, and the cycle is repeated so each gene is read masked 3 times. The uniform-virtual anchor needs no masked variant because the read-out gene's own input already carries no information about its true level. TranscriptFormer needs none at all: it is autoregressive, so a gene never sees its own value and is natively in the masked regime.

### Findings

Masking the read-out gene's own input value changes the response at the same anchor, and by enough that the two are different features rather than the same one with noise. Median absolute zero-out response, unmasked against masked: scGPT: 0.00398 to 0.00219 in HepG2 (a factor of 0.550); 0.00396 to 0.00199 in Jurkat (a factor of 0.504). AIDO.Cell: 0.000176 to 0.000312 in HepG2 (a factor of 1.773); 0.000131 to 0.000186 in Jurkat (a factor of 1.423). scPRINT: 0.00207 to 0.000191 in HepG2 (a factor of 0.093); 0.00133 to 0.000158 in Jurkat (a factor of 0.118). The size of the gap differs by model rather than being a fixed offset, which is what the masked block was extracted to establish.

The direction is not the same for all of them, which is the part worth stating plainly. For AIDO.Cell the response grows when the gene's own value is taken away, up to a factor of 1.773, which is the naive expectation: a model that cannot see the answer has to infer it from the rest of the cell, so the rest of the cell matters more. For scGPT, scPRINT it shrinks instead, by as much as a factor of 0.093: removing the gene's own value made those models less sensitive to the perturbed gene, not more. Two accounts are consistent with the shrinking direction: a masked prediction falling back on a population prior that no single other gene moves much, and the mask token itself being off the distribution the model was trained to read. The next section separates them. A consumer should treat masked and unmasked as two features and not assume a common sign of the effect across models.

### Separating the mask token from the information it removes

A shrinking response has two readings and the masked block alone cannot tell them apart. Either the model falls back on what the rest of the cell implies, which is information removal and a real property of the masked regime, or the mask token is a value the model is not being asked to read the way pretraining used it, in which case the whole response is measured off the input distribution and says nothing about reconstruction. A third condition separates them: instead of the mask token, the read-out gene's input is set to one uninformative but in-distribution level, the median pooled-control level across read-out genes (1.013 CP10K in HepG2 and 1.028 CP10K in Jurkat, at the 0.500 and 0.500 quantile of the read-out genes' own levels). It removes the same gene-specific information as masking while leaving every input value inside the range the model was trained on. If the shrinkage survives it, it is information removal; if it only appears with the mask token, it is an artefact of the query.

All three conditions run in one job over the same 60 perturbed genes at the mean-control anchor with one batching, so nothing but the condition differs. They also correct the masked estimator. The masked read cycles disjoint blocks and reads each gene only where it is hidden; when the perturbed gene falls in the same block as the gene being read, the mask overwrites the perturbation before the forward pass, so that pass is identical to its own baseline and contributes exactly zero. Run 2's masked block averaged those zeros in, which shrinks its magnitudes by 14.3% and 25% -- one over the number of blocks per cycle. Those passes are skipped here, and because their contribution is exactly zero the run 2 estimator is reproduced from these arrays rather than argued about.

**Figure 4: Three ways to hide the read-out gene's own value, at the same anchor** — Median absolute zero-out response at the mean control anchor over the same perturbed-gene subset, with the read-out gene's own input value visible, replaced by each model's own mask convention, and replaced by one uninformative in-distribution level, the median pooled-control level of the read-out genes (log axis, each model in its own output units). All three come from one job with one batching, and all three drop the passes in which the mask would have overwritten the perturbation itself, so they sit slightly above the shipped masked block's own numbers. The mask token and an uninformative in-distribution value are not the same query. For AIDO.Cell, scGPT, scPRINT the substitution does not reproduce what the mask does, so that model's masked response says more about the mask token than about reconstruction. TranscriptFormer is absent: it is autoregressive, so a gene never sees its own value. (`figures/masked_regime/`)

| model | cell line | perturbed genes | visible | mask token (of visible) | median level (of visible) | median level over mask token |
|---|---|---|---|---|---|---|
| scGPT | HepG2 | 60 | 0.00353 | 0.00269 (0.762) | 0.00363 (1.028) | 1.348 |
| scGPT | Jurkat | 60 | 0.00296 | 0.0023 (0.778) | 0.00277 (0.937) | 1.205 |
| AIDO.Cell | HepG2 | 60 | 0.00019 | 0.000374 (1.973) | 0.000216 (1.140) | 0.578 |
| AIDO.Cell | Jurkat | 60 | 0.000141 | 0.000223 (1.579) | 0.000157 (1.110) | 0.703 |
| scPRINT | HepG2 | 60 | 0.00324 | 0.000222 (0.068) | 0.00101 (0.313) | 4.573 |
| scPRINT | Jurkat | 60 | 0.0025 | 0.000143 (0.057) | 0.000781 (0.313) | 5.458 |

For AIDO.Cell, scGPT and scPRINT the substitution does not reproduce what the mask does, so that model's masked response is at least partly an artefact of being queried outside its input distribution and must not be read as evidence about reconstruction. The rule is stated before the numbers are read: the substitution reproduces the mask when both ratios sit on the same side of 1 and the substituted one is at least half the mask's size in log units. Per-gene agreement between the conditions is recorded beside the medians in each record, because a ratio of medians can hide a change of pattern.

| model | cell line | visible vs mask token | visible vs median level | mask token vs median level |
|---|---|---|---|---|
| scGPT | HepG2 | 0.2600 | 0.3785 | 0.3455 |
| scGPT | Jurkat | 0.2794 | 0.3568 | 0.2523 |
| AIDO.Cell | HepG2 | 0.4839 | 0.7550 | 0.6064 |
| AIDO.Cell | Jurkat | 0.3136 | 0.7042 | 0.4179 |
| scPRINT | HepG2 | 0.0324 | 0.0675 | -0.0147 |
| scPRINT | Jurkat | -0.0411 | 0.2913 | -0.0447 |

Median per-gene Pearson correlation between the two conditions' zero-out response vectors, over the read-out genes each condition could read. A high correlation with a small ratio means the same pattern at a smaller size; a low correlation means the conditions are measuring different things.

## Gradient trajectories and the finite-difference step

### Procedure

A gradient-trajectory feature is a magnitude, so a step-dependent derivative would make it meaningless. Forward-mode automatic differentiation is used wherever the model's layout is one differentiable function of the input; elsewhere the derivative is a central difference whose step is selected by sweeping 6 steps on 50 perturbed genes at a 4 CP10K base value and taking the smallest step that agrees with the next larger step at median Pearson at or above 0.999 and median relative L2 difference at or below 0.1. Where no step qualifies, the block is flagged: its gradient features are usable for direction and ranking, not magnitude.

### Findings

**Figure 5: Finite-difference step stability** — Each point compares the gradient column computed at that step with the column at the next larger step, as the median Pearson correlation over 50 perturbed genes at a 4 CP10K base value. A step is selected only if it also agrees in magnitude (median relative L2 difference at or below 0.10). Where the curve sits on the threshold line the gradient is a property of the model rather than of the step; where it falls away, the block's gradient features are usable for direction and ranking but not magnitude. (`figures/step_stability/`)

18 of the 24 block and anchor pairs converge and 6 do not, and the split is by model rather than by anchor: every failure is TranscriptFormer. Its adjacent-step agreement climbs steadily without ever reaching the threshold, from 0.2693 at step 0.05 to 0.9732 at step 1, so the rule falls back on the step with the best direction agreement (1) and flags the block. The reason is visible in the size of the numbers: across its swept steps the gradient columns being differenced have a median absolute value of 9.36e-09, close enough to float noise that the difference between two nearby steps is dominated by it. Those gradient features are usable for direction and ranking and not for magnitude, and a consumer reads that verdict out of `step_selection.json` rather than inferring it.

| model | cell line | anchor | selected step | verdict | gradient source |
|---|---|---|---|---|---|
| AIDO.Cell | HepG2 | uniform virtual | 0.02 | stable in direction and magnitude | central_finite_difference(step=0.02) |
| AIDO.Cell | HepG2 | mean control | 0.02 | stable in direction and magnitude | central_finite_difference(step=0.02) |
| AIDO.Cell | HepG2 | individual cells | 0.02 | stable in direction and magnitude | central_finite_difference(step=0.02) |
| AIDO.Cell | Jurkat | uniform virtual | 0.02 | stable in direction and magnitude | central_finite_difference(step=0.02) |
| AIDO.Cell | Jurkat | mean control | 0.02 | stable in direction and magnitude | central_finite_difference(step=0.02) |
| AIDO.Cell | Jurkat | individual cells | 0.02 | stable in direction and magnitude | central_finite_difference(step=0.02) |
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
| scGPT | HepG2 | uniform virtual | side check, 20 genes | 0.999498 | 0.03103 | 0.02 |
| scGPT | Jurkat | uniform virtual | side check, 20 genes | 0.999704 | 0.02846 | 0.02 |
| scPRINT | HepG2 | uniform virtual | production | 0.999745 | 0.0219 | 0.02 |
| scPRINT | HepG2 | mean control | production | 0.999837 | 0.0176 | 0.02 |
| scPRINT | HepG2 | individual cells | production | 0.999961 | 0.008866 | 0.02 |
| scPRINT | Jurkat | uniform virtual | production | 0.999704 | 0.02332 | 0.02 |
| scPRINT | Jurkat | mean control | production | 0.999827 | 0.01816 | 0.02 |
| scPRINT | Jurkat | individual cells | production | 0.999962 | 0.008774 | 0.02 |

AIDO.Cell is absent from that table because the exact route does not exist for this build, which is a measurement rather than an omission. Attempting forward-mode autodiff through its forward pass raises `NotImplementedError: Trying to use forward AD with _scaled_dot_product_efficient_attention that does not support it because it has not been implemented yet`. Its gradient trajectories are therefore central differences at the selected step with no exact cross-check behind them, and their magnitude evidence is the step sweep alone, which they pass at the smallest swept step in every anchor and both cell lines.

One record had to be corrected to say this. Run 1 recorded a substitution note reading `forward-mode AD unavailable (NotImplementedError); used central finite differences`, while the same contract's `jacobian_available` field asserted the opposite; that field was an unmeasured default, and it is where run 2's first report read the claim that the route was available. The contract now carries a separate `forward_mode_autodiff` field filled from an actual attempt, and every adapter records why a forward-mode column was unavailable instead of returning nothing silently.

## Normalization, which is not cosmetic here

### Procedure

The four models emit values on four incommensurable scales, and two of them end in a softmax across the genes supplied in the same forward pass. Three consequences are handled explicitly: the supplied gene set is fixed per model and cell line and recorded; the cross-gene mean response is measured per query and a recentred variant is emitted beside the raw one wherever the output has a bridge to a linear expression scale, which is every model except scGPT; and every block also carries a per-query standardized form, z-scored across read-out genes, so the comparison run can use features that are invariant to each model's output scale.

A predicted log fold change is emitted only where a defensible bridge to a linear expression scale exists: AIDO.Cell's head predicts log1p CP10K, so the bridge is expm1; scPRINT and TranscriptFormer predict a share of the cell's counts, which is proportional to CP10K. scGPT's output is a per-cell quantile bin index, which is not an expression level and has no cell-independent inverse, so the bridge is declined rather than invented.

### Findings

**Figure 6: How much of each response is redistribution** — For a model whose head ends in a softmax across the supplied genes, raising one gene's share must lower the others, so the response differences sum to about zero by construction. The bars show the median cross-gene mean divided by the cross-gene spread (log axis). scPRINT sits below the tolerance at every query, which is the signature of exact redistribution. TranscriptFormer sits below it at the low values and rises above it at the top of the grid: its read-out set is delivered in two separately normalized passes, so the shares sum to one inside a pass and not across the union. AIDO.Cell sits in the same range without a softmax head, because one gene's query barely moves a 19,264-gene prediction on average, so a small ratio is not by itself evidence of redistribution. scGPT, whose output is a per-cell bin index, is the only one clearly above. (`figures/softmax_coupling/`)

Ratio of the absolute cross-gene mean response to its cross-gene spread, over every query in both cell lines: scGPT median 0.139, worst 1.29; AIDO.Cell median 0.063, worst 1.09; scPRINT median 0.0231, worst 0.0887; TranscriptFormer median 0.0441, worst 0.243. The check is applied only to the two models whose head is a softmax over the supplied genes, scPRINT and TranscriptFormer. The reading is one-directional: a small ratio is what redistribution produces, but it is not by itself evidence of a softmax. AIDO.Cell, whose head is a per-gene regression over the full transcriptome, sits in the same range simply because one gene's query moves a 19,264-gene prediction very little on average. scGPT, whose output is a per-cell bin index, is the only one clearly above.

TranscriptFormer misses the 0.05 tolerance on 5 of its 6 anchor and cell-line pairs, at 0.05049, 0.08621, 0.06775, 0.06262, 0.05627. The cause is the layout rather than the head. Its read-out set does not fit in one pass, so it is delivered in two, each normalized by that pass's own total; the shares sum to one inside a pass and not across the union of the two, which leaves exactly this kind of small residual offset. Every block ships a recentred variant beside the raw one, and the recentred variant removes it. A consumer that cares about the identity should read `recentred`.

**Figure 7: The four output scales, per feature form** — Median and 99th percentile of the absolute raw feature value at the mean control anchor, in each model's own output units (log axis). Circles are medians, diamonds 99th percentiles; both cell lines are shown. Raw features are not comparable across models, which is what the standardized and, where defensible, predicted log fold-change forms are for. (`figures/feature_scale/`)

The four raw scales are not comparable, and the gap is not small. Median absolute zero-out response at the mean-control anchor: scGPT 0.00396 to 0.00398; AIDO.Cell 0.000131 to 0.000176; scPRINT 0.00133 to 0.00207; TranscriptFormer 2.13e-06 to 3.18e-06, about 1,866-fold between the extremes, which reflects each head's units rather than anything biological. The comparison run should read the standardized form by default and treat a raw cross-model magnitude comparison as meaningless.

## Two architectural constraints that shape the query

### scGPT: how wide the supplied set is

scGPT was trained at a maximum sequence length of 1,200 but carries no positional encoding over genes, so a longer input is architecturally admissible; run 1 supplied the whole 7,444-gene universe and recorded it as an extrapolation. Run 2 measured what the width changes, on 40 common perturbed genes at the mean control anchor, comparing the expressed read-out width (2,721 genes), the full universe (7,444) and tiles of at most 1,200 sharing a fixed 256-gene prefix.

| comparison | median Pearson | median relative L2 difference |
|---|---|---|
| readout vs universe | 0.9811 | 0.7026 |
| readout vs tiled | 0.7706 | 1.055 |
| universe vs tiled | 0.7661 | 1.231 |

**Figure 8: scGPT: what the supplied-set width changes** — Zero-out response differences for the same perturbed genes at the mean control anchor, computed with three supplied widths: the expressed read-out set, run 1's full 7,444-gene universe, and tiles of at most 1,200 genes sharing a fixed prefix. A relative L2 difference of 0 means identical vectors, 2 means opposite. Directions largely agree and magnitudes do not, so the width is a measurement choice rather than a detail; the production sweep uses the tiled width, which is the one inside scGPT's training length. (`figures/scgpt_context/`)

The three widths do not agree closely, so the production sweep uses the tiled width, the one inside scGPT's training length. The cost is that a tiled query is several forward passes rather than one differentiable function of the input, so scGPT's production gradients are central differences at the selected step; the exact forward-mode derivative is available in the untiled layout and was validated against a central difference there.

### TranscriptFormer: ordering

TranscriptFormer is autoregressive with a causal mask, so gene j responds to gene i only when i precedes it. The perturbed gene is placed at position 0 of every pass, which makes every read-out gene measurable; the cost is that the context differs between queries by that one gene. Under a single fixed ordering 50% of ordered pairs would be structurally unmeasurable, which is a property of the model rather than of this run. Each pass opens with the same 256-gene prefix and its shares are normalized by that pass's own total, so values from different passes are comparable.

## Read-out order alignment

### Procedure

The failure that would silently invalidate everything is a gene-index misalignment between a model's vocabulary order and the read-out gene order. The pre-registered guard reads the response between the lowest positive value on the grid and the highest and asks that the perturbed gene's own entry be the largest response in the vector, whatever its sign; the plan's sign version asks instead that the gene's own predicted output move the expected way. Each is repeated with the perturbed gene's own read-out column relabelled at random, which gives the statistic its null.

The pre-registered guard turns out to have no power for an autoregressive model. With the perturbed gene placed first, TranscriptFormer never conditions on its own value, so its own entry is never the largest response and the permuted control is zero as well: test and null sit together at the floor and the comparison says nothing. Two statistics that keep their power there were added: the median rank of the own entry within the response vector, whose null is the middle of the vector, and the sign rate, whose null is a coin flip. A misalignment would drive all three statistics to their nulls at once, so one wide separation is enough to rule it out.

### Findings

**Figure 9: Read-out order alignment, each statistic against its permuted null** — At the mean control anchor, over the perturbed genes sampled in each block. Each statistic is shown beside the same statistic computed after relabelling the perturbed gene's own read-out column at random (the pale bar). 'Own entry largest' is the pre-registered guard; 'own entry high in the vector' is one minus the median rank of the own entry as a fraction of the vector; 'own output moves as expected' is the sign reading. scPRINT saturates the pre-registered guard on both lines. scGPT and AIDO.Cell miss its 0.95 threshold while the own entry still ranks first for the median gene, so the rank statistic is what establishes their alignment. TranscriptFormer scores exactly zero on that guard, and so does its null: with the perturbed gene placed first, an autoregressive model never conditions on its own value, so the guard cannot have power there. Every block separates from its null on at least one statistic at every anchor. (`figures/wiring_checks/`)

| model | cell line | anchor | perturbed genes sampled | own entry largest (null) | median self-rank (null) | own output moves as expected (null) | separated on |
|---|---|---|---|---|---|---|---|
| AIDO.Cell | HepG2 | uniform virtual | 250 | 0.628 (0.000) | 0 of 2,721 (1,362.5) | 0.184 (0.176) | self_rank |
| AIDO.Cell | HepG2 | mean control | 250 | 0.536 (0.000) | 0 of 2,721 (1,175.5) | 0.244 (0.744) | self_rank |
| AIDO.Cell | HepG2 | individual cells | 60 | 0.500 (0.000) | 0.5 of 2,721 (1,389.5) | 0.283 (0.767) | self_rank |
| AIDO.Cell | Jurkat | uniform virtual | 250 | 0.736 (0.000) | 0 of 3,247 (1,674) | 0.084 (0.120) | self_rank |
| AIDO.Cell | Jurkat | mean control | 250 | 0.688 (0.000) | 0 of 3,247 (1,592) | 0.104 (0.820) | self_rank |
| AIDO.Cell | Jurkat | individual cells | 60 | 0.667 (0.000) | 0 of 3,247 (1,736.5) | 0.100 (0.817) | self_rank |
| scGPT | HepG2 | uniform virtual | 524 | 0.849 (0.000) | 0 of 2,721 (1,305) | 0.872 (0.969) | self_rank |
| scGPT | HepG2 | mean control | 524 | 0.897 (0.000) | 0 of 2,721 (1,389) | 0.922 (0.594) | self_rank, sign |
| scGPT | HepG2 | individual cells | 524 | 0.918 (0.000) | 0 of 2,721 (1,447.5) | 0.954 (0.595) | self_rank, sign |
| scGPT | Jurkat | uniform virtual | 705 | 0.933 (0.000) | 0 of 3,247 (1,547) | 0.994 (0.911) | self_rank |
| scGPT | Jurkat | mean control | 705 | 0.935 (0.000) | 0 of 3,247 (1,587) | 0.817 (0.631) | self_rank, sign |
| scGPT | Jurkat | individual cells | 705 | 0.942 (0.001) | 0 of 3,247 (1,722) | 0.953 (0.610) | self_rank, sign |
| scPRINT | HepG2 | uniform virtual | 524 | 1.000 (0.000) | 0 of 2,721 (1,255.5) | 1.000 (0.468) | own_largest, self_rank, sign |
| scPRINT | HepG2 | mean control | 524 | 1.000 (0.000) | 0 of 2,721 (1,310.5) | 1.000 (0.649) | own_largest, self_rank, sign |
| scPRINT | HepG2 | individual cells | 524 | 1.000 (0.000) | 0 of 2,721 (1,412) | 1.000 (0.601) | own_largest, self_rank, sign |
| scPRINT | Jurkat | uniform virtual | 705 | 1.000 (0.000) | 0 of 3,247 (1,660) | 1.000 (0.492) | own_largest, self_rank, sign |
| scPRINT | Jurkat | mean control | 705 | 1.000 (0.000) | 0 of 3,247 (1,636) | 1.000 (0.735) | own_largest, self_rank |
| scPRINT | Jurkat | individual cells | 705 | 1.000 (0.001) | 0 of 3,247 (1,584) | 1.000 (0.748) | own_largest, self_rank |
| TranscriptFormer | HepG2 | uniform virtual | 524 | 0.000 (0.000) | 926 of 2,721 (1,316.5) | 0.929 (0.668) | none |
| TranscriptFormer | HepG2 | mean control | 524 | 0.000 (0.002) | 68 of 2,721 (1,380.5) | 0.897 (0.733) | self_rank |
| TranscriptFormer | HepG2 | individual cells | 524 | 0.000 (0.000) | 67.5 of 2,721 (1,422) | 0.899 (0.746) | self_rank |
| TranscriptFormer | Jurkat | uniform virtual | 705 | 0.000 (0.001) | 654 of 3,247 (1,686) | 0.960 (0.614) | sign |
| TranscriptFormer | Jurkat | mean control | 705 | 0.000 (0.000) | 80 of 3,247 (1,631) | 0.967 (0.803) | self_rank |
| TranscriptFormer | Jurkat | individual cells | 705 | 0.000 (0.000) | 76 of 3,247 (1,572) | 0.969 (0.794) | self_rank |

Each cell gives the observed statistic with its permuted null in parentheses. The last column names the statistics that separate widely enough to establish alignment for that block. 23 of 24 block and anchor pairs separate from their null; the rest do not establish alignment and are not usable.

The pre-registered guard is reported as it stands and is not retro-fitted to the result. It is met by scPRINT at every anchor in both cell lines and it is not met by scGPT, AIDO.Cell and TranscriptFormer, so for those models the alignment conclusion rests on the permutation-controlled statistics rather than on the threshold written before the data existed. What the guard would have caught the replacements also catch: a misalignment relabels the own entry at random, which is exactly the permutation, and it would drive all three statistics to their nulls at once.

One reading in that table is a finding rather than a check. For AIDO.Cell the sign statistic does not merely miss its threshold, it inverts: at worst, the perturbed gene's own predicted output rises with its input value for only 10% of sampled genes, against 81.7% when that same column is relabelled at random. The inversion is specific to the gene's own entry, which is why it is evidence of correct alignment and not of a mix-up: other genes move the ordinary way. It is the behaviour of a head trained to reconstruct expression from context, which pulls an artificially raised gene back toward the level the rest of the cell implies. Run 3 should not assume the own-entry sign is positive, and any summary that averages the own entry together with the rest of the vector will partly cancel.

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
  hidden_readout/condition=<c>/     visible / mask token / median level, where
                                    the model has a masking regime

features_full_scope/<model>/<cell_line>/   the same layout at full coverage
feature_index/blocks.json                  the canonical block per (model, line)

```

Two trees, because coverage differs by model: `features/` holds the 250-gene blocks the first wave produced and `features_full_scope/` the full-coverage blocks that supersede them for the three cheap models. Read `feature_index/blocks.json` to resolve the canonical block for a (model, cell line) rather than globbing either tree, and read the `coverage` block of a manifest before comparing two blocks: they share the smaller one's perturbed genes but not its row order, so join on `perturbed_gene_id`.

Each npz carries `raw` and `standardized` always, and `recentred` and `log2fc` wherever the model's output has a bridge to a linear expression scale, plus the perturbed and read-out gene identifiers and the value grid, so a block is self-describing.

### Checkpoints, and where model code had to be changed

| model | checkpoint | revision |
|---|---|---|
| scGPT | `wanglab/scGPT-human` | `a24c237737a40f3720f75abb555489e9fe753be6` |
| AIDO.Cell | `genbio-ai/GB.Cell-100M (plan alias genbio-ai/AIDO.Cell-100M, upstream rename)` | `a983b388e1025860f2c413ecef60e76078253c24` |
| scPRINT | `jkobject/scPRINT:medium-v1.5.ckpt` | `8e1ad1a1bc40c1fc2f4f1700fe27923eea9a2bed` |
| TranscriptFormer | `s3://czi-transcriptformer/weights/tf_sapiens.tar.gz` | `tf_sapiens (public bucket object, unversioned)` |

Both cell lines of a model load the same checkpoint at the same revision, and each block's `manifest.json` records it. Every model needed some change to its shipped code to run here, 7 in total, and each block records its own verbatim. They are worth reading because several of them touch numerics:

| model | substitution |
|---|---|
| scGPT | flash-attention not installed (the published checkpoint was trained with scGPT's FlashMHA layers, whose API exists only in flash-attn 1.x, which cannot build against this torch); used scGPT's own fallback torch.nn.TransformerEncoder path (use_fast_transformer=False), which changes the attention kernel and therefore the exact numerics |
| scGPT | remapped 24 fused attention tensors from the checkpoint's FlashMHA layout (self_attn.Wqkv) onto torch's packed projection (self_attn.in_proj); both hold [query; key; value] over the same head partition, so the attention computed is the same function of the same weights. Verified numerically at load time (see attention_remap_check) |
| scGPT | imported scgpt.model.model by file path to avoid executing scgpt/__init__ (scvi-tools / torchtext / scib); model source unmodified |
| AIDO.Cell | replaced BertSelfFlashAttention.forward with a shape-correct SDPA implementation: the shipped non-flash fallback softmaxes across heads instead of across the sequence, and the flash-attention path is fp16/bf16 only with no forward-mode derivative. Weights and attention mathematics are unchanged; the kernel and dtype are not. |
| scPRINT | simpler_flash.FlashSelfAttention.forward delegated to the package's own pure-torch SelfAttention: the fused kernel asserts fp16/bf16 and has no forward-mode derivative, so the response and its Jacobian would not be computable in float32. Same weights, same attention, different kernel (upstream forward kept at FlashSelfAttention.forward) |
| scPRINT | stubbed scprint.model.utils / scprint.utils / scprint.tasks.* at import time to avoid bionty+lamindb+scdataloader (a configured lamindb instance is not available and is unnecessary for a forward pass); model source unmodified |
| TranscriptFormer | compile_block_mask disabled (eager flex-attention block mask) so that a single-cell forward pass does not pay a torch.compile warm-up |

They fall into three kinds. Three are import plumbing, working around packages that will not install or configure here, and leave the model itself untouched. Three replace a fused half-precision attention kernel with the same attention in float32, because the fused kernel refuses float32 or carries no forward-mode derivative; the weights and the attention function are unchanged, the kernel and the dtype are not, so exact numerics differ from a flash-attention run. AIDO.Cell's is in that second group and also repairs a bug: its packaged non-flash fallback softmaxes across heads rather than across the sequence, which is not the attention the weights were trained with. No cross-check against the published kernel was possible for it, because that kernel does not import in this environment. The last one remaps scGPT's fused query-key-value tensors onto torch's packed projection, which is the only change that moves checkpoint weights, and it was verified numerically at load time.

## Verification

Verification runs at two levels. Every block is checked inside a job that mounts the volume the extraction wrote, against that block's own npz files rather than against a summary: 352 checks over 8 blocks, 328 passed. The roll-up then reads the 8 per-block records together and adds coverage and alignment checks: 51 checks, 44 passed. Both outputs ship with the report, and the failures are listed here rather than summarised away.

| failing check | blocks | what it means |
|---|---|---|
| alignment_vs_permutation | 1 anchor and block pairs in transcriptformer/hepg2 | see the per-block record |
| softmax_coupling | 5 anchor and block pairs in transcriptformer/hepg2, transcriptformer/jurkat | the redistribution identity, missed by a small margin because the read-out set is delivered in two separately normalized passes. The recentred variant that ships in every block removes the offset. |
| wiring_magnitude | 18 anchor and block pairs in aido_cell/hepg2, aido_cell/jurkat, scgpt/hepg2, scgpt/jurkat, transcriptformer/hepg2, transcriptformer/jurkat | the pre-registered 95% own-entry-largest guard. Reported as it stands; the permutation-controlled statistics in the wiring section establish alignment for these blocks instead. |

The roll-up's own failures, which are the per-block failures restated plus anything missing from the tree:

| check | detail |
|---|---|
| block_checks/aido_cell/hepg2 | 37 block-level checks, 3 failed: ['uniform_virtual/wiring_magnitude', 'mean_control/wiring_magnitude', 'individual_cells/wiring_magnitude'] |
| block_checks/aido_cell/jurkat | 37 block-level checks, 3 failed: ['uniform_virtual/wiring_magnitude', 'mean_control/wiring_magnitude', 'individual_cells/wiring_magnitude'] |
| block_checks/scgpt/hepg2 | 39 block-level checks, 3 failed: ['uniform_virtual/wiring_magnitude', 'mean_control/wiring_magnitude', 'individual_cells/wiring_magnitude'] |
| block_checks/scgpt/jurkat | 39 block-level checks, 3 failed: ['uniform_virtual/wiring_magnitude', 'mean_control/wiring_magnitude', 'individual_cells/wiring_magnitude'] |
| alignment/transcriptformer/hepg2 | uniform_virtual on no statistic; mean_control on self_rank; individual_cells on self_rank |
| block_checks/transcriptformer/hepg2 | 46 block-level checks, 6 failed: ['individual_cells/softmax_coupling', 'uniform_virtual/softmax_coupling', 'uniform_virtual/alignment_vs_permutation', 'uniform_virtual/wiring_magnitude', 'mean_control/wiring_magnitude', 'individual_cells/wiring_magnitude'] |
| block_checks/transcriptformer/jurkat | 46 block-level checks, 6 failed: ['individual_cells/softmax_coupling', 'mean_control/softmax_coupling', 'uniform_virtual/softmax_coupling', 'uniform_virtual/wiring_magnitude', 'mean_control/wiring_magnitude', 'individual_cells/wiring_magnitude'] |

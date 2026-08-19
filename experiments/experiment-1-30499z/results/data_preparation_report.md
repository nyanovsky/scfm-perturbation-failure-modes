> This Markdown file is a reading copy. The primary artifact is the self-contained HTML report, where every figure is embedded inline and interactive.

# Replogle-Nadig effect matrices and the four-model response contract

This run measured perturbation effects for two CRISPRi Perturb-seq cell lines and wrote down, with measurements rather than assertions, what four single-cell foundation models emit when asked about a cell. In HepG2 1,982 perturbations and in Jurkat 2,201 survive filtering, their fold changes agree with an independent published reanalysis at pooled Spearman 0.9251 over 8,464,986 gene-level entries, and each of the four models now has a per-gene response vector and a Jacobian column over a shared 7,444-gene universe on one real control cell.

The most consequential number here is not an agreement statistic but a limit. Splitting each perturbation's cells in two and comparing the two halves against each other, the weaker half of the perturbations barely reproduce themselves: median agreement is 0.0961 and 0.1247 in the two lowest HepG2 quartiles against 0.7031 in the highest. Any later comparison of model predictions has to be read against that limit rather than against a perfect score.

Each of the three areas below is written in two parts: first the quantity and the exact procedure that produced it, including sample sizes and thresholds, then the measured values with what each one implies. Every number is read at build time out of the artifact files the run produced, which are listed at the end.

_Generated 2026-08-19 02:08 UTC._

## The effect matrices

### Definition, procedure, and thresholds

An effect matrix is one table per cell line. **A row is one CRISPRi perturbation**: a single target gene, with every cell carrying any guide against that target pooled together, so the unit is the target gene and not the individual guide. **A column is one gene in the assay's measured panel.** The entry at a row and column is how much that gene's expression changed in cells carrying that perturbation, relative to non-targeting control cells.

Each cell is first normalized to counts per 10,000 over the measured panel (CP10K), and a perturbation's expression profile is the mean of those normalized per-cell vectors, not a normalized sum of raw counts. The change is reported in two forms: `delta`, the difference of mean CP10K, and `log2fc`, the base-2 logarithm of their ratio with a small constant (0.01 CP10K) added to both sides so a zero control level cannot produce an infinite value.

Two control estimators are computed for each form, which is why `effects.npz` holds four matrices. **The batch-matched estimator is the primary one** (`delta_within`, `lfc_within`). It computes the mean control profile separately in each sequencing batch (`gem_group`), then builds the reference for a perturbation as the average of those per-batch control means weighted by the share of that perturbation's own cells in each batch. The batch-pooled estimator (`delta_pooled`, `lfc_pooled`) references every perturbation against one control mean over all control cells, and is kept as a secondary column for comparison. Row order follows `perturbations.parquet` and column order follows `genes.parquet`; the matrices carry no labels of their own, so the indices mean nothing without those two tables.

The batch-matched reference is weighted rather than computed strictly within batch because these libraries are thin: the median perturbation here has 1.6 cells per batch it appears in (HepG2). A strictly within-batch contrast would estimate each batch's effect from one or two perturbed cells, drop every batch that happens to hold no control cell, and then average those noisy pieces. Weighting the control side by the perturbation's batch composition removes the same first-order confound, which is that a perturbation over-represented in one batch would otherwise inherit that batch's baseline, while keeping all of the perturbation's cells on the numerator side. The two estimators are compared directly in the cross-check below, so the size of the correction is a measured number rather than an assumption.

Differential expression is scored differently on purpose. For each gene, a **Welch t-test on log1p(CP10K)** compares that perturbation's cells against **pooled** non-targeting controls, and Benjamini-Hochberg control at 0.05 is applied across expressed genes; `n_de_genes` counts the genes that pass. The test uses pooled controls because a t-test needs a variance and a sample size on both sides, and the batch-matched reference is a weighted mean of per-batch means, a point estimate with no per-cell spread attached; the individual batches hold far too few control cells to estimate a per-gene variance. Pooling all 4,916 HepG2 and 12,004 Jurkat control cells gives the most stable variance and the largest degrees of freedom, and batch composition mainly shifts the control mean, which the effect estimate already corrects, rather than inflating the within-gene variance. The consequence to keep in mind is that the differential-expression count and the reported fold change are computed against slightly different references, so they are two related screens rather than two views of one number.

A gene counts as **expressed** when its pooled control level is at least 0.5 CP10K; effect sizes, the differential-expression correction, and every ceiling are restricted to those genes. **On-target knockdown** is one minus the ratio of the perturbed to the batch-matched control level at the target gene's own column, and a perturbation whose knockdown falls below 0.30 is flagged rather than dropped, as is one whose target gene is not in the measured panel at all.

- Cell filters: at least 2,000 UMI, at least 500 genes detected, mitochondrial fraction at most 0.15.
- Perturbation filters: at least 25 quality-passing cells spread over at least 2 sequencing batches.
- Expressed genes: pooled control level at least 0.5 CP10K.
- Differential expression: Benjamini-Hochberg FDR below 0.05 across expressed genes.
- Seed 42 for every split and sample.

**The effect-size strata are equal-count quartiles, cut per cell line.** All retained perturbations in a cell line are ranked by `effect_l2_lfc`, the Euclidean norm of their batch-matched log2 fold-change vector over expressed genes, with ties broken by first occurrence. The ranks are then divided into 4 groups of equal size: the lowest quarter of ranks becomes `S1_lowest` and the highest becomes `S4_highest`. Because the cut is on ranks within a cell line, the stratum boundaries are not fixed effect-size values and are not the same in the two cell lines; the realized ranges are in the table below, and a stratum label should never be read as an absolute effect size across cell lines.

**Figure 1: Cells surviving each filter step, both cell lines** ([interactive render](figures/filter_funnel/filter_funnel.html), [plot data](figures/filter_funnel/data.json))

Cells remaining after each filter in the order they are applied, from the raw deposited matrix to the final working set.

Per-cell quality control removes almost nothing, because the deposited matrices are already filtered: 3,137 HepG2 cells go to the mitochondrial-fraction cut and 631 Jurkat cells to the minimum-count cut. The attrition that matters is at the perturbation level, where requiring 25 cells across 2 batches takes 2,393 HepG2 perturbations down to 1,982.

### Results and what they imply

|  | HepG2 | Jurkat |
|---|---|---|
| cells × measured genes | 145,473 × 9,624 | 262,956 × 8,882 |
| source file size | 5.614 GB | 9.366 GB |
| sequencing batches | 56 | 55 |
| cells after quality control | 135,368 | 259,335 |
| non-targeting control cells | 4,916 | 12,004 |
| retained perturbations | 1,982 | 2,201 |
| expressed genes (CP10K ≥ 0.5) | 3,119 of 9,624 | 3,469 of 8,882 |
| median on-target knockdown | 0.8976 | 0.8866 |
| perturbations below the knockdown floor | 228 | 306 |
| target gene not in the measured panel | 187 | 223 |
| median differentially expressed genes | 36 | 61 |
| median cells per perturbation | 51 | 88 |
| processing runtime | 169 s | 279 s |

Both cell lines pass the checks a Perturb-seq effect matrix has to pass before it is worth modelling: knockdown is nearly complete at the median, the retained perturbations each have tens of cells across at least two batches, and a third of the measured panel is expressed highly enough to carry a fold change. Jurkat is the deeper of the two datasets in cells but the noisier one per perturbation, which shows up in the ceilings below.

**Figure 2: On-target knockdown per perturbation** ([interactive render](figures/knockdown_distribution/knockdown_distribution.html), [plot data](figures/knockdown_distribution/data.json))

How much of the target gene's control-level expression each perturbation removed, as one histogram per cell line. The marked line is the 0.30 detection floor below which a perturbation is flagged rather than dropped.

The mass sits near complete knockdown (median 0.8976 in HepG2, 0.8866 in Jurkat), so the CRISPRi perturbation did what it claims for most targets, and the 228 HepG2 and 306 Jurkat perturbations below the floor are identifiable by name.

**Figure 3: On-target knockdown against transcriptome-wide effect size** ([interactive render](figures/knockdown_vs_effect/knockdown_vs_effect.html), [plot data](figures/knockdown_vs_effect/data.json))

Each retained perturbation's knockdown of its own target against the L2 norm of its log2 fold-change vector over expressed genes, which is the size of its response across the rest of the transcriptome.

The two are nearly unrelated: Spearman 0.0726 in HepG2 and -0.1225 in Jurkat. Silencing a gene efficiently does not make its downstream response large, so selecting perturbations by knockdown strength does not select the ones a model has a chance of predicting.

That has a practical consequence for how a later run picks its evaluation set. Perturbations with a large measurable response cannot be selected by knockdown strength; they have to be selected by effect size or by stratum, which is what the strata below exist for. Knockdown remains the check that the intended perturbation happened, not a proxy for how much it did.

| stratum | HepG2 n | HepG2 effect-size range | HepG2 median DE genes | HepG2 median cells | Jurkat n | Jurkat effect-size range | Jurkat median DE genes | Jurkat median cells |
|---|---|---|---|---|---|---|---|---|
| S1_lowest | 496 | 3.92 – 12.19 | 3 | 71 | 551 | 3.27 – 10.65 | 4 | 139 |
| S2 | 495 | 12.20 – 15.11 | 5 | 42 | 550 | 10.66 – 13.90 | 31 | 88 |
| S3 | 496 | 15.11 – 20.83 | 79 | 40 | 550 | 13.90 – 18.57 | 130 | 72 |
| S4_highest | 495 | 20.89 – 50.27 | 984 | 54 | 550 | 18.58 – 45.98 | 448 | 55 |

The realized boundaries show why the strata are worth carrying: the effect norm spans roughly an order of magnitude from the lowest to the highest quartile, and the number of differentially expressed genes spans two. They also show the trap in the labels, since the same label covers different absolute ranges in the two cell lines, so `S4_highest` in Jurkat is not the same experimental object as `S4_highest` in HepG2.

**Figure 4: Effect size against differentially expressed gene count, by stratum** ([interactive render](figures/effect_size_vs_de/effect_size_vs_de.html), [plot data](figures/effect_size_vs_de/data.json))

Every retained perturbation, with effect size on the x-axis and its number of differentially expressed genes on a logarithmic y-axis, shifted by one so perturbations with none of them remain visible. Colour marks the equal-count effect-size quartile.

The two measures of a perturbation's strength agree in ordering but not in scale: median differentially expressed genes rise from 3 in the lowest HepG2 quartile to 984 in the highest, while the effect norm varies continuously inside every quartile.

## The noise ceilings

### Definition, procedure, and thresholds

A noise ceiling answers a question about the data, not about any model: **if the same perturbation were measured twice at this sequencing depth, how similar would the two measurements be?** Each perturbation's quality-passing cells are divided into two sides, the effect vector is computed independently on each side using exactly the pipeline described above, and the two vectors are compared across expressed genes. Whatever agreement they reach is the reproducible part of the measurement; the rest is sampling noise that no predictor can recover, because it is not a property of the perturbation at all.

Two split forms are reported, and both keep the control cells of the two sides disjoint so shared control noise cannot inflate the agreement. `split_half_cells` splits the perturbation's cells in half by a seeded assignment and references each half against its own half of the non-targeting control cells. `split_by_batch` alternates the perturbation's sequencing batches into two sets, largest batch first, and references each set against the control cells of its own batches, so disjointness holds by construction rather than by the seed. A side with fewer than 5 cells is not scored, which is why some perturbations carry a ceiling for one split form and not the other.

Three metrics are computed per split, chosen to match the ways a later run will score a model against these effects: Pearson correlation across expressed genes on log2 fold changes (`pearson_lfc`), cosine similarity on the linear CP10K deltas (`cosine_delta`), and the overlap of the top 50 genes ranked by absolute log2 fold change (`topk_overlap`). Per-perturbation values live in `perturbations.parquet`; `noise_ceiling_summary.parquet` holds the median and the quartiles per stratum.

The overlap metric is reported over a grid of k rather than at one k. For every perturbation and both split forms, the two sides' top-k sets are compared at k = 10, 50, 100, 500, and at an adaptive k defined below. The expensive part of this measurement is rebuilding the two split-half effect vectors from the raw count matrices; once they exist, another k costs microseconds, so the whole grid is computed and any k can be read off it. The k = 50 column is the `topk_overlap` column already in `perturbations.parquet`, recomputed and identical to it. Per-perturbation values live in `topk_overlap_grid.parquet`, per-stratum medians in `topk_overlap_grid_summary.parquet`, and the split-half vectors themselves in `split_half_vectors.npz` (log2FC and delta, both sides, both split forms, float32 over expressed genes), so a later question about another k or another metric is a file read rather than another pass over the raw data. The stage re-reads the raw count matrices, because those vectors were not kept the first time, and it reproduces every published ceiling column exactly on the way through (largest absolute difference 0 over both split forms and all three metrics), so nothing already reported changes meaning.

**Overlap has a floor that moves with k, so raw overlap is not comparable across k.** Two independent sets of size k drawn from n expressed genes share k/n of their members by chance. In HepG2, with 3,119 expressed genes, that chance level is 0.003206 at k = 10 and 0.1603 at k = 500; in Jurkat, with 3,469 expressed genes, 0.002883 and 0.1441. A top-500 overlap of 0.30 is therefore under twice what two random draws reach, while a top-10 overlap of 0.30 is roughly a hundred times chance. Every cell of the grid is reported twice for that reason: the raw overlap, and the excess over chance, (overlap - chance) / (1 - chance), which is 0 when the two halves agree no better than two random draws and 1 when they select exactly the same genes.

A fifth column sets k per perturbation rather than per stratum: k is that perturbation's own `n_de_genes`, clipped to [10, 500]. The strata are rank quartiles cut inside a cell line, so a stratum label does not fix a response size: the median S3 perturbation has 79 differentially expressed genes in HepG2 against 130 in Jurkat, and the median S4 perturbation 984 against 448. Asking both lines for their top 500 genes asks a much harder question of Jurkat's top stratum than of HepG2's. A k tied to the perturbation's own response asks both the same question, at the cost of a chance level that changes from row to row, so the adaptive column carries its own chance value per perturbation.

Two diagnostics travel with every cell, because the overlap can be quietly meaningless without them. Selecting the top k with `argpartition` breaks ties arbitrarily, which matters at small k where many genes sit near zero log2 fold change, so `tie_fraction` records the share of the selected set within numerical tolerance (relative 1e-06) of the k-th value; its floor is 1/k because the k-th value ties itself, and membership is only actually contested when a gene outside the set falls inside that tolerance too. Selection is on absolute log2 fold change with no expression or significance filter, so a small-k set can be dominated by low-expression genes whose ratios are large and unreliable; `median_control_cp10k_selected` records the median pooled control level of the selected genes, to be read against the median over all expressed genes (1.0108 CP10K in HepG2, 1.0232 in Jurkat).

**A model score has to be read as a fraction of this number, not on its own.** If two independent measurements of the same perturbation agree at 0.30, then a prediction that correlates with a measured effect vector at 0.30 has reached the level at which the experiment agrees with itself, and a prediction at 0.30 in a stratum whose ceiling is 0.70 has reached less than half of it. Reporting the raw correlation alone makes the first look as weak as the second. The ceilings are per stratum and per cell line for exactly this reason: they are the denominators.

### Results and what they imply

| stratum | HepG2 split by cell | HepG2 split by batch | HepG2 cosine on delta | Jurkat split by cell | Jurkat split by batch | Jurkat cosine on delta |
|---|---|---|---|---|---|---|
| S1_lowest | 0.0961 | 0.0934 | 0.2995 | 0.1041 | 0.0992 | 0.3765 |
| S2 | 0.1247 | 0.1184 | 0.4255 | 0.1906 | 0.1930 | 0.6885 |
| S3 | 0.2980 | 0.2910 | 0.7082 | 0.2933 | 0.2946 | 0.8080 |
| S4_highest | 0.7031 | 0.7044 | 0.9229 | 0.4347 | 0.4305 | 0.8997 |

Agreement rises with effect size in both cell lines, in both split forms, and in both metrics, and the two split forms land on nearly the same value in every cell of the table. That is what a working ceiling estimate looks like, and it means the ordering is a property of the perturbations rather than of the way the halves were drawn.

**The bottom two strata are effectively unpredictable at this sequencing depth.** Their log2 fold-change vectors reproduce at 0.0961 and 0.1247 in HepG2 and 0.1041 and 0.1906 in Jurkat, which is close enough to zero that the measured effect vector is mostly noise. That covers 991 of the 1,982 HepG2 perturbations and 1,101 of the 2,201 Jurkat perturbations, about half of each dataset. Two consequences follow for everything built on these matrices. A model that scores near zero there has not been shown to fail, and a model that scores slightly above another one has not been shown to be better; the measurement cannot separate them. Any later claim about model quality on this data therefore has to be restricted to the upper strata, or stated as a fraction of the stratum's ceiling, and a headline number pooled over all four strata will be dominated by perturbations whose truth is unmeasured.

Cosine similarity on the linear deltas stays high everywhere (0.9229 in the top HepG2 stratum) because it is dominated by the few highly-expressed genes with large absolute changes. It is the easier metric and should not be quoted as though it were the correlation.

**Figure 5: Replicate agreement of a perturbation with itself, by stratum** ([interactive render](figures/noise_ceilings/noise_ceilings.html), [plot data](figures/noise_ceilings/data.json))

Median agreement between two independent halves of the same perturbation, for each effect-size quartile, each metric, each cell line, and both split forms. Solid bars split the cells; hatched bars split the sequencing batches, which forces disjoint control cells.

Reproducibility rises steeply with effect size and stays low in the bottom half: median agreement on log2 fold changes is 0.0961 in the lowest HepG2 quartile against 0.7031 in the highest. The two split forms give nearly the same answer, so the ceiling is not an artifact of sharing control cells.

**Figure 6: Replicate agreement against cells per perturbation** ([interactive render](figures/ceiling_vs_cells/ceiling_vs_cells.html), [plot data](figures/ceiling_vs_cells/data.json))

The same split-half agreement, now per perturbation against how many cells that perturbation has, on a logarithmic x-axis, with the median of each cell-count bin drawn over the points.

More cells help, but effect size decides: agreement correlates with cell count at Spearman 0.3836 in HepG2 and with effect size at 0.7243. The weakest quartile is not the shallowest one, so its low agreement is a small-signal problem rather than an undersampling problem.

The low ceiling in the weak strata is not a matter of too few cells. Agreement correlates with effect size at Spearman 0.7243 (HepG2) and 0.5291 (Jurkat) but with cell count at only 0.3836 and 0.2679, and cell count runs *against* effect size (-0.2427 in HepG2, -0.5980 in Jurkat): the weakest quartile has a median of 71 cells in HepG2 against 54 in the strongest. The weak perturbations were measured more deeply and still did not reproduce, so their response is small relative to cell-level variation rather than merely undersampled, and sequencing more cells of the same perturbations is not an obvious route to a usable signal there.

#### The top-k overlap grid

Median overlap of the two halves' top-k gene sets, split by cell, with excess over chance in parentheses and each stratum's own k in bold. The chance row is the overlap two independent draws of the same size reach, and the adaptive column uses each perturbation's own differentially expressed gene count as its k. Splitting by sequencing batch instead of by cell moves no cell of this table by more than 0.0200.

| cell line | stratum | k = 10 | k = 50 | k = 100 | k = 500 | adaptive k |
|---|---|---|---|---|---|---|
| HepG2 | chance level | 0.003206 | 0.01603 | 0.03206 | 0.1603 | 0.01154 (median) |
| HepG2 | S1_lowest | **0.2000 (0.1974)** | 0.1000 (0.0853) | 0.1100 (0.0805) | 0.2380 (0.0925) | 0.1722 (0.1558) at k = 10 |
| HepG2 | S2 | 0.1000 (0.0971) | **0.1000 (0.0853)** | 0.1100 (0.0805) | 0.2400 (0.0949) | 0.1304 (0.1130) at k = 10 |
| HepG2 | S3 | 0.2000 (0.1974) | 0.1600 (0.1463) | **0.1700 (0.1425)** | 0.2760 (0.1378) | 0.2000 (0.1846) at k = 79 |
| HepG2 | S4_highest | 0.2000 (0.1974) | 0.3800 (0.3699) | 0.4400 (0.4215) | **0.5120 (0.4188)** | 0.5120 (0.4188) at k = 500 |
| Jurkat | chance level | 0.002883 | 0.01441 | 0.02883 | 0.1441 | 0.01758 (median) |
| Jurkat | S1_lowest | **0.1000 (0.0974)** | 0.0600 (0.0463) | 0.0800 (0.0527) | 0.2160 (0.0840) | 0.1000 (0.0974) at k = 10 |
| Jurkat | S2 | 0.2000 (0.1977) | **0.1000 (0.0868)** | 0.1000 (0.0733) | 0.2260 (0.0957) | 0.1290 (0.1031) at k = 31 |
| Jurkat | S3 | 0.2000 (0.1977) | 0.1600 (0.1477) | **0.1600 (0.1351)** | 0.2460 (0.1190) | 0.1943 (0.1469) at k = 130 |
| Jurkat | S4_highest | 0.2000 (0.1977) | 0.1700 (0.1579) | 0.1900 (0.1660) | **0.2980 (0.1798)** | 0.2854 (0.1873) at k = 448.5 |

**Reading across a row shows why the chance level has to be quoted with the number.** The weakest HepG2 quartile appears to improve from 0.2000 at k = 10 to 0.2380 at k = 500, which looks like a better measurement at larger k. It is not: chance rises from 0.003206 to 0.1603 over the same span, so the excess over chance stays near 0.0925 throughout. Two thirds of that top-500 overlap is what two random draws would have produced. The same row in Jurkat reads 0.2160 against a chance level of 0.1441.

**Reading down a column shows which k can see the effect-size ordering.** At k = 50 and above, the top stratum separates cleanly from the bottom one in both cell lines: in HepG2 the excess runs 0.0805 to 0.4215 at k = 100, and in Jurkat 0.0527 to 0.1660. At k = 10 it cannot: a per-perturbation overlap of ten genes takes only the eleven values 0, 0.1, ..., 1, so the median collapses onto that grid and three of the four HepG2 strata land on exactly 0.2000. k = 10 is a legitimate question about the few strongest genes, but it is too coarse to rank perturbations by, and a model compared at k = 10 on a single perturbation is being scored on a ten-way coin toss.

**Figure 7: Top-k overlap against the chance level it has to beat** ([interactive render](figures/topk_overlap_grid/topk_overlap_grid.html), [plot data](figures/topk_overlap_grid/data.json))

Median overlap between the two split halves' top-k gene sets, over k = 10, 50, 100 and 500 on a logarithmic axis, by effect-size stratum and cell line. The dashed line is the chance level k/n for n expressed genes; the open square marks the k quoted for that stratum and the cross the median adaptive k. The lower panels show the same numbers as excess over chance, where 0 is chance and 1 is an identical set.

Raw overlap climbs with k in every stratum, and most of that climb is the chance level climbing underneath it: in the weakest HepG2 quartile the top-500 overlap of 0.2380 sits against a chance level of 0.1603, while the strongest quartile reaches 0.5120. In the excess panels the curves are nearly flat, so the ranking of the strata, not the value of k, is what the metric is telling you.

**The adaptive column is the one that compares the two cell lines.** Each perturbation is asked for as many genes as it actually moved, so the median k is 79 in HepG2's S3 against 130 in Jurkat's, and 500 against 448.5 in S4. On that footing the top HepG2 stratum reproduces its own response at an excess of 0.4188 against 0.1873 in Jurkat, so the S4 label covers a substantially more reproducible set of perturbations in HepG2 than in Jurkat. In the bottom two strata the adaptive k collapses to the floor of 10 in HepG2, because the median perturbation there has fewer than ten differentially expressed genes at all, which is another way of saying the same thing the correlation ceiling says: there is little there to reproduce.

| cell line | k | median tie fraction | selections with a contested boundary | median control level of the selected genes (CP10K) |
|---|---|---|---|---|
| HepG2 | 10 | 0.1 | 0.05% | 0.7652 |
| HepG2 | 50 | 0.02 | 0.00% | 0.7165 |
| HepG2 | 100 | 0.01 | 0.10% | 0.7184 |
| HepG2 | 500 | 0.002 | 0.35% | 0.7702 |
| HepG2 | adaptive | 0.02778 | 0.25% | 0.7649 |
| Jurkat | 10 | 0.1 | 0.05% | 0.6538 |
| Jurkat | 50 | 0.02 | 0.05% | 0.6632 |
| Jurkat | 100 | 0.01 | 0.00% | 0.6794 |
| Jurkat | 500 | 0.002 | 0.27% | 0.7632 |
| Jurkat | adaptive | 0.01639 | 0.09% | 0.6861 |

The two diagnostics land differently. Ties are not a practical problem in this data: the median tie fraction is exactly the 1/k floor in every cell, so the only value tied with the k-th value is the k-th value itself, and the share of selections where any gene outside the set is indistinguishable from the boundary peaks at 1.01% across every cell of the grid. The expression diagnostic is the one worth carrying: the selected genes sit below the typical expressed gene everywhere, from 0.6202 to 0.8907 CP10K against about 1 CP10K over all expressed genes, and the lean is strongest exactly where the signal is weakest (0.6202 CP10K for the lowest Jurkat stratum at k = 10 against 0.8907 for the highest HepG2 stratum at k = 500). Ranking by absolute log2 fold change with no expression filter does pull in genes whose ratios are large because their denominators are small, so a later run that wants a top-k comparison to be about biology rather than about counting noise should either add an expression floor or read this column as the caveat it is.

None of this changes a number that was already reported: the k = 50 column is the published `topk_overlap` column, recomputed to the last bit, and the correlation and cosine ceilings are untouched. What the grid adds is the ability to quote an overlap honestly. A top-k overlap without its k, its chance level and its expressed-gene count is not interpretable, and the same overlap value can mean a strong result at k = 10 and a near-chance result at k = 500.

## Agreement with an independent reanalysis

### Definition, procedure, and thresholds

The reference release `tahoebio/replogle-nadig-de-rhaister` holds a separate differential-expression analysis of the same GEO series, computed per sequencing batch and published as `pdex/all_pdex.parquet` over a 6,546-gene panel. Its per-batch results are pooled the same way ours are, as `log2(sum_b n_b*target_mean_b / sum_b n_b*ref_mean_b)`, and then compared entry by entry on log2 fold changes for every (cell line, perturbation, gene) triple both sides share. The comparison was pre-registered with a gate: pooled Spearman at least 0.70, with the number of compared pairs reported alongside it. The same module also compares our two control estimators against each other, which is the measurement that says how much the batch-matched correction actually changes.

### Results and what they imply

| quantity | value |
|---|---|
| pooled Spearman against the reference | 0.9251 |
| gene-level entries compared | 8,464,986 |
| shared perturbations | 2,877 (HepG2 1,340, Jurkat 1,537) |
| median agreement within one perturbation | 0.9449 |
| batch-matched against batch-pooled, log2FC | 0.9963 |
| batch-matched against batch-pooled, delta | 0.9971 |

Two independently written pipelines starting from the same raw matrices agree on the fold changes, which rules out the class of error that silently reindexes genes, mixes up controls, or normalizes twice. It does not validate the choices the two pipelines share, so it is a check on implementation rather than on method.

| stratum | perturbations | median Spearman vs reference |
|---|---|---|
| S1_lowest | 663 | 0.9191 |
| S2 | 741 | 0.9300 |
| S3 | 713 | 0.9519 |
| S4_highest | 760 | 0.9772 |

Agreement with the reference rises with effect size in the same way the self-agreement does, from 0.9191 in the weakest quartile to 0.9772 in the strongest. The residual disagreement in the weak strata is the same noise the ceilings measure, seen from the other side: two analyses of an effect that barely reproduces cannot agree closely about it either.

The two control estimators agree at median Pearson 0.9963 on log2 fold changes, so the batch-matched correction is small on most perturbations. It is kept as the primary estimate because where the two disagree, the disagreement is exactly the batch-composition confound that the weighting removes; a later run that prefers the pooled column can read it from the same file and should say which one it used.

**Figure 8: Agreement with the published reanalysis, per gene and per perturbation** ([interactive render](figures/reference_crosscheck/reference_crosscheck.html), [plot data](figures/reference_crosscheck/data.json))

Left: a random sample of per-gene log2 fold changes, ours against the published reference release, with the identity line. Right: the distribution of per-perturbation agreement, ours against the reference and our two control estimators against each other.

Pooled Spearman against the reference is 0.9251 over 8,464,986 shared entries, and our two estimators agree with each other at median Pearson 0.9963. The scatter is tight along the identity line rather than along a different slope, so the two pipelines agree on scale as well as on ordering.

## The four-model response contract

### Definition, procedure, and thresholds

The shared gene universe holds **7,444 genes**: measured in both cell lines (7,632 genes qualify on that count alone) and addressable in all four model vocabularies. Identifiers are resolved to Ensembl gene ids through the assay's own var table first, then HGNC approved symbols, then HGNC aliases and previous symbols, and the route taken is recorded for every token in `gene_universe/vocabulary_<model>.parquet`. Round-tripping the 10,874 gene names in the series fails 1 time in the Ensembl to symbol to Ensembl direction and 0 times in the reverse direction; unmapped vocabulary tokens are counted, not dropped silently.

The demonstration is deliberately narrow: one real non-targeting control cell from HepG2 (barcode `AAACCCAAGCTGGAGT-49`, 12,129 total counts and 3,637 genes detected) is passed to each model through that model's own published inference path, and two things are recorded. The **response vector** is the model's native per-gene output for the universe genes, whose meaning differs per model and is written down in `contract.json`. The **Jacobian column** is the derivative of every gene's response with respect to the observed count of one source gene (`ENSG00000198712`, MT-CO2, observed count 197), which is the quantity a later run needs if it wants to ask how one gene's abundance moves the model's opinion about the others.

Each Jacobian column is computed by the most exact route the model's graph allows and then cross-checked against a second route or step size, with the Pearson correlation and the relative L2 difference between the two recorded in `demo.json`. Where the graph is differentiable end to end, the column is a forward-mode automatic derivative; where the input is discretized or the autodiff pass does not fit in memory, it is a central finite difference at a stated step. The step is part of the measurement, not an implementation detail, so it is recorded with the value.

Symmetry is measured rather than assumed. A run that reads one column of a response matrix and treats the matrix as symmetric is relying on a property no model documents, so each model was additionally asked for a second column at the gene its source gene moves most, and the two cross entries were compared. **A verdict is recorded only when the Jacobian estimate itself is stable**, meaning Pearson at least 0.99 and relative L2 difference at most 0.5 between two step sizes; otherwise step sensitivity and genuine asymmetry are not separable, and the contract says so instead of guessing.

**The per-gene values are not comparable across models, and a consuming run has to do something about it.** The four outputs are four different quantities: scGPT returns a predicted expression bin value on its 51-bin training scale, AIDO.Cell returns its masked-language-model regression target on a log1p CP10K scale, and scPRINT and TranscriptFormer both return a share of the cell's counts, because their decoders end in a softmax across the gene positions supplied in the same forward pass. On this one cell those ranges are scgpt 25.56 to 36.16, aido_cell -2.09 to 2.527, scprint 0.1863 to 141.9, transcriptformer 1.552e-04 to 0.01033. A difference or ratio between two models' raw values has no defined meaning, and for the two softmax models even a single model's values move when the supplied gene set changes, for reasons that have nothing to do with biology. A later run therefore has to declare one comparison space and stay in it: rank the genes within a cell per model, or standardize within a model, or better, have each model produce its own contrast between a perturbed and an unperturbed pass in its own units and compare those contrasts by correlation or rank overlap. It also has to hold the supplied gene set fixed across the passes it compares.

**Asymmetry constrains how a response matrix may be indexed.** The object a later run builds is directional: entry (i, j) is the change in gene j's response caused by a change in gene i, and the two indices are not interchangeable. Where the measurement below is decisive it comes out asymmetric, so a run may not fill the other half of the matrix by transposition, may not average the two triangles, and may not use one column to stand in for the corresponding row. Getting both directions costs two passes per pair. For TranscriptFormer the restriction is structural: with a causal mask only the entries where the source precedes the target are reachable at all, and the rest are exactly zero by construction rather than small by measurement.

**Figure 9: Vocabulary size and assay coverage for the four models** ([interactive render](figures/model_gene_coverage/model_gene_coverage.html), [plot data](figures/model_gene_coverage/data.json))

For each model: how many tokens its vocabulary holds, how many resolve to a distinct Ensembl gene id, and how many of the genes measured in each cell line it can address. The axis is logarithmic because the vocabularies differ by a factor of three; the marked line is the shared universe.

All four models can address the shared 7,444 genes, but they get there from very different vocabularies, and scGPT's 21,338 unmapped vocabulary tokens are the reason the universe is defined by intersection rather than by any one model's gene list.

### Results and what they imply

| model | vocabulary | identifier | fixed input order | covers HepG2 measured | covers Jurkat measured |
|---|---|---|---|---|---|
| scgpt | 60,697 | symbol | no | 9,522 / 9,624 | 8,780 / 8,882 |
| aido_cell | 19,264 | symbol | yes | 9,296 / 9,624 | 8,548 / 8,882 |
| scprint | 44,756 | ensembl | yes | 9,297 / 9,624 | 8,546 / 8,882 |
| transcriptformer | 23,830 | ensembl | no | 9,297 / 9,624 | 8,549 / 8,882 |

The vocabularies differ by a factor of three and use different identifier kinds, so the intersection, not any single model's gene list, is what all four can be asked about. Two of the four also require a fixed input order, which means the universe has to carry each model's token positions rather than only its gene ids.

| model | per-gene value | information flow | ordering matters | Jacobian as run |
|---|---|---|---|---|
| scgpt | predicted binned expression value for that gene in this cell, on the 51-bin training scale (not counts, not CP10K) | bidirectional full attention over gene tokens; no positional encoding over genes | no | central_finite_difference(step=0.5)_on_smooth_binning |
| aido_cell | one scalar per gene position (the model's masked-language-model regression target, i.e. predicted high-read-depth expression on the log1p CP10K scale); the last two positions are read-depth tokens and are dropped | bidirectional encoder (18 layers) over the full fixed gene order | no | central_finite_difference(step=1.97) |
| scprint | predicted share of the cell's counts for that gene: the expression decoder ends in a softmax across the supplied gene positions, conditioned on the requested read depth | bidirectional flash transformer over the supplied gene set plus cell-embedding tokens | no | forward_mode_autodiff_jvp |
| transcriptformer | predicted share of the cell's total counts assigned to that gene, conditioned on the gene's own identity and on the identity and count of every gene earlier in the ordering | autoregressive, causal mask over the gene sequence | yes | central_finite_difference(step=1)_source_first_ordering |

The recorded Jacobian route is the one that actually ran, which is not always the one the contract declares as preferred: scGPT's forward-mode pass over 7,444 genes exhausted the GPU, so its column is a central finite difference. Reading the declared field instead of the recorded one would misdescribe the number.

**Figure 10: Each model's per-gene response on the same control cell** ([interactive render](figures/model_responses_on_one_cell/model_responses_on_one_cell.html), [plot data](figures/model_responses_on_one_cell/data.json))

One panel per model: the response the model returns for each of the 7,444 universe genes, against that gene's observed count in the cell. Vertical scales are per panel and are not comparable.

The four panels have different vertical ranges and different shapes because the models emit four different quantities, and each response tracks the observed counts only loosely (Spearman 0.3569 to 0.5878), so none of them is simply echoing its input.

| model | second column route | d target / d source | d source / d target | ratio | verdict |
|---|---|---|---|---|---|
| scgpt | central_finite_difference(step=0.4831)_on_smooth_binning | +2.8408e-02 | -2.0489e-02 | -0.7212 | asymmetric |
| aido_cell | central_finite_difference(step=0.02) | -7.8509e-04 | -4.5300e-04 | 0.577 | not determined |
| scprint | forward_mode_autodiff_jvp | +4.9628e-03 | +6.8167e-03 | 1.374 | asymmetric |
| transcriptformer | central_finite_difference(step=1)_source_first_ordering | +5.7457e-06 | -6.1287e-05 | -10.67 | not determined |

Where the measurement is decisive it comes out asymmetric (scgpt, scprint), and for scGPT the two cross entries do not even share a sign. The response matrix is directional in fact and not only in principle, so the indexing rule stated above is a requirement rather than a caution.

For aido_cell the finite-difference Jacobian is itself step-dependent on this cell, so the contract records symmetry as not determined. The practical reading is narrower than it sounds: the direction of that column is reproducible, so ranking genes by sensitivity is defensible, while anything that depends on the magnitude needs a step-size study first.

For transcriptformer the cross-entry probe is also step-limited, but symmetry is settled by the architecture rather than by this measurement, as the ordering probe below shows directly.

| model | Jacobian route | cross-check route | Pearson | relative L2 gap | column L2 |
|---|---|---|---|---|---|
| scgpt | central_finite_difference(step=0.5)_on_smooth_binning | second central finite difference at step=1 | 0.9993 | 0.1616 | 0.1855 |
| aido_cell | central_finite_difference(step=1.97) | second central finite difference at step=1 | 0.9991 | 1.379 | 0.0107 |
| scprint | forward_mode_autodiff_jvp | central_finite_difference(step=1.73) | 1 | 2.629e-04 | 0.7169 |
| transcriptformer | central_finite_difference(step=1)_source_first_ordering | second central finite difference at step=2 | 0.8714 | 0.9803 | 1.289e-05 |

Only scPRINT's column is an exact derivative, agreeing with an independent finite difference to a relative L2 difference of 2.629e-04. The other three are finite differences over integer counts, which is the honest description of what a discretized or masked input allows: the direction of the column reproduces across step sizes, its magnitude does not, and TranscriptFormer's per-gene sensitivities on this cell are close to numerical noise (column L2 1.289e-05).

The ordering probe settles TranscriptFormer directly. Reading the same target gene twice from the same cell with 254 context genes between them, raising the source gene's count from 197 to 296 changes the target's predicted share by +1.565e-03 (on a base of 1.271e-02) when the source sits ahead of the target, and by exactly +0.0 when it sits behind. Half of a TranscriptFormer response matrix is therefore not small, it is unreachable, and a run that averages the two triangles is averaging measurements with structural zeros.

### Substitutions

Where a published inference path could not be run as shipped, the substitution is recorded in the model's `contract.json` rather than hidden:

**scgpt**

- flash-attention not installed (the published checkpoint was trained with scGPT's FlashMHA layers, whose API exists only in flash-attn 1.x, which cannot build against this torch); used scGPT's own fallback torch.nn.TransformerEncoder path (use_fast_transformer=False), which changes the attention kernel and therefore the exact numerics
- remapped 24 fused attention tensors from the checkpoint's FlashMHA layout (self_attn.Wqkv) onto torch's packed projection (self_attn.in_proj); both hold [query; key; value] over the same head partition, so the attention computed is the same function of the same weights. Verified numerically at load time (see attention_remap_check)
- imported scgpt.model.model by file path to avoid executing scgpt/__init__ (scvi-tools / torchtext / scib); model source unmodified
- forward-mode AD unavailable (OutOfMemoryError: CUDA out of memory. Tried to allocate 2.70 GiB. GPU 0 has a total capacity of 79.18 GiB of which 2.31 MiB is free. Process 1 has 79.17 GiB memory in use. Of the allocated memory 76.32 GiB is allocated by PyTorch, and 2.19 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)); used central finite differences instead

**aido_cell**

- replaced BertSelfFlashAttention.forward with a shape-correct SDPA implementation: the shipped non-flash fallback softmaxes across heads instead of across the sequence, and the flash-attention path is fp16/bf16 only with no forward-mode derivative. Weights and attention mathematics are unchanged; the kernel and dtype are not.
- forward-mode AD unavailable (NotImplementedError); used central finite differences

**scprint**

- simpler_flash.FlashSelfAttention.forward delegated to the package's own pure-torch SelfAttention: the fused kernel asserts fp16/bf16 and has no forward-mode derivative, so the response and its Jacobian would not be computable in float32. Same weights, same attention, different kernel (upstream forward kept at FlashSelfAttention.forward)
- stubbed scprint.model.utils / scprint.utils / scprint.tasks.* at import time to avoid bionty+lamindb+scdataloader (a configured lamindb instance is not available and is unnecessary for a forward pass); model source unmodified

**transcriptformer**

- compile_block_mask disabled (eager flex-attention block mask) so that a single-cell forward pass does not pay a torch.compile warm-up

### The four stacks as run

The plan called for one pinned environment per model, on the assumption that the four dependency sets are mutually unsatisfiable. They are not quite: pinning torch first and installing the four model packages with `--no-deps`, since each would otherwise drag torch back to the version it was built against, gives one stack (`envs/model_stack.sh`) that runs all four. Every model here ran on torch 2.5.1+cu121 in the same image, each in its own job, and each records its own resolved package set in `environment.json`, so a later run can tell whether two results came from the same stack.

| model | image | device | checkpoint load | forward | Jacobian |
|---|---|---|---|---|---|
| scgpt | docker.io/library/python:3.11-slim | cuda | 0.85 s | 0.463 s | 1.30 s |
| aido_cell | docker.io/library/python:3.11-slim | cuda | 2.02 s | 1.145 s | 3.81 s |
| scprint | docker.io/library/python:3.11-slim | cuda | 3.75 s | 0.336 s | 0.65 s |
| transcriptformer | docker.io/library/python:3.11-slim | cuda | 7.41 s | 0.891 s | 1.03 s |

## The file contract later runs read

```
effects/<cell_line>/effects.npz              delta and log2FC matrices, both estimators
effects/<cell_line>/perturbations.parquet    per-perturbation metadata, knockdown, DE, strata, ceilings
effects/<cell_line>/genes.parquet            gene index, pooled control level, expressed flag
effects/<cell_line>/noise_ceiling_summary.parquet   per-stratum medians and quartiles
effects/<cell_line>/topk_overlap_grid.parquet        per-perturbation overlap at every k, with chance and diagnostics
effects/<cell_line>/topk_overlap_grid_summary.parquet  per-stratum medians of that grid
effects/<cell_line>/split_half_vectors.npz   the two split halves' log2FC and delta vectors
effects/<cell_line>/topk_overlap_grid_qc.json  grid config, chance levels, reproduction of the published ceilings
effects/<cell_line>/qc.json                  provenance, filter funnel, config, environment
crosscheck/summary.json                      agreement with the published reanalysis
gene_universe/gene_universe.parquet          one row per shared gene, with each model's token position
gene_universe/vocabulary_<model>.parquet     every vocabulary token, its gene id, and how it resolved
gene_universe/coverage.json                  coverage and the identifier audit
models/<model>/response_vector.parquet       per-gene response on the demonstration cell
models/<model>/jacobian_column.parquet       d response / d source gene count
models/<model>/contract.json                 the declared response contract
models/<model>/environment.json              pinned packages, image, device
models/<model>/demo.json                     what was run, timings, checks, symmetry probe
```

`uv run python -m scfm_pfm.verify --all` re-checks every claim on this page against those files: matrix shapes and finiteness, knockdown and differential-expression ranges, the four strata, ceilings bounded in (0, 1] and ordered across strata, the reference Spearman against its gate, vocabulary positions inside each model's vocabulary, and one finite response vector per model with its contract fields filled in, including a symmetry field that is either answered or explicitly measured and inconclusive. For the overlap grid it also re-derives the chance levels from the expressed-gene count, re-checks the excess-over-chance definition and the adaptive k on every row, confirms that the k = 50 column still equals the published `topk_overlap` column, and recomputes the overlap from the persisted split-half vectors to confirm they are the vectors that were scored.

## Provenance

| artifact | source | revision or checksum |
|---|---|---|
| HepG2 matrix | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_hepg2_raw_singlecell_01.h5ad | md5 dfc676e4186b8c8c173ff14e1194cc8a |
| Jurkat matrix | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_jurkat_raw_singlecell_01.h5ad | md5 dae9fab2a8b0430099a76ce60d0c5405 |
| scgpt vocabulary | wanglab/scGPT-human | a24c237737a40f3720f75abb555489e9fe753be6 |
| aido_cell vocabulary | https://files.pythonhosted.org/packages/65/3c/4f36684dad8b4cc5b3145dea2f05f25fd8a3ae29934f90e30f31f1ca377b/modelgenerator-0.1.3.post0.tar.gz | 6fb102f3bf76d75a04cf93ca57f55bf5 |
| scprint vocabulary | jkobject/scPRINT | 8e1ad1a1bc40c1fc2f4f1700fe27923eea9a2bed |
| transcriptformer vocabulary | czi-transcriptformer | 59fe97ed61265c99e1059514fcf6ff2a |
| scgpt checkpoint | wanglab/scGPT-human | a24c237737a40f3720f75abb555489e9fe753be6 |
| aido_cell checkpoint | genbio-ai/GB.Cell-100M (plan alias genbio-ai/AIDO.Cell-100M, upstream rename) | a983b388e1025860f2c413ecef60e76078253c24 |
| scprint checkpoint | jkobject/scPRINT:medium-v1.5.ckpt | 8e1ad1a1bc40c1fc2f4f1700fe27923eea9a2bed |
| transcriptformer checkpoint | s3://czi-transcriptformer/weights/tf_sapiens.tar.gz | tf_sapiens (public bucket object, unversioned) |
| reference release | tahoebio/replogle-nadig-de-rhaister | pdex/all_pdex.parquet |

Source series: GEO GSE264667. Seed 42 throughout. The configuration lives in `src/scfm_pfm/config.py` and is copied into every `qc.json`, so a later run can see exactly which thresholds produced these matrices.

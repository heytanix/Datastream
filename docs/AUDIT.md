# Audit of the schema-2.0 matcher (Milestone 1)

Scope: `datastream/mvfp.py`, `similarity.py`, `matcher.py`, `mv_engine.py`, plus `bench_mv.py`, `store.py`, `models.py`, tests.
Nothing was changed during the audit. Every "confirmed" item below was reproduced by `python -m datastream.research.audit_checks`
(raw numbers in `results/research/audit.json`, seed 12345). Baseline benchmark results are frozen in `results/baseline/`.

## 1. Current execution flow

```
drift signal (Page-Hinkley on errors | label-free feature shift D_X > 0.25)            mv_engine._stable
  -> collect >= 40 post-drift observations, retry every 10, give up at 200              mv_engine._recovering
  -> F_X = extract_feature_fp(X, t)                                                      mvfp
  -> Level 1: vectorised distance on (median, robust sigma, q10, q90), keep top-5        matcher.level1
  -> Level 2: per candidate D = weighted views, noise-floored                            similarity.compare
  -> decision: INSUFFICIENT / NEW / CANDIDATE / HIGH, heuristic confidence               matcher.match
  -> shadow validation of <= 3 candidates incl. the active concept on labelled samples  mv_engine._resolve
  -> accept (recall / false alarm) | reject -> rollback if provisional -> train new
  -> accepted: merge window into stored fingerprint, update intra variability            matcher.update
```

## 2. Mathematical description of what is implemented

Window `X in R^{n x d}`, optional timestamps `t` (rows stable-sorted by `t`). NaN/inf = missing.

**Fingerprint** (per numeric feature `j`; NaN = unavailable, never 0)
`stat_j` = {mean, std, median, min, max, q01, q10, q25, q75, q90, q99, skew, missing}; `dist_j` = 21-point quantile sketch `Q_j(p), p in {0, .05, ..., 1}`;
`corr` = Spearman correlation of each feature pair (average ranks, NaN filled by the median, constant columns -> NaN);
`time_j` = {trend = corr(x, t), acf1/5/10 (need n >= 3 lag + 10 and regular sampling), volatility = sd(dx)/(sqrt(2) sd(x)), rate of change, season = max_{2<=l<=min(n/3,40)} acf_l, abrupt = share of |dx - med| > 3 MAD}.

**Effective sample size** (`FeatureFingerprint.effective_n`): `rho = clip(median_j acf1_j, 0, 0.95)`, `n_eff = max(5, n (1-rho)/(1+rho))`. One global rho for every feature and every statistic.

**Noise scale**: `nu = sqrt(1/n_eff_a + 1/n_eff_b)`, floor `kappa * nu` with one `kappa = 1`, subtracted once from an aggregate raw distance, never made feature- or statistic-specific.

**Per-view distances**, all mapped by `D = 1 - exp(-max(0, raw - kappa nu) / tau)`:
* `D_stat` per feature: `raw = (2 loc + spread + 0.5 skew + miss) / 4.5` with `loc` = mean of `|Delta|/s` over {median, q10, q25, q75, q90} clipped to [0,5] (`s` = pooled IQR/1.349), `spread = |log(sigma_a/sigma_b)|`, `skew = |Delta|/1.7`, `miss = 5 |Delta|`. Categorical: `(3 TV + 5 |Delta miss|)/2`.
* `D_dist` per feature: `W1 = mean_p |Q_a(p) - Q_b(p)| / s` and `PSI` with 10 bins at the deciles of `a` (asymmetric), each floored (`kappa nu` and `9 kappa nu^2`) and normalised; averaged. Plus Spearman term `|Delta r|` (weight 0.25).
* `D_time` per feature: mean over available statistics of `|Delta|` (abrupt x5), floored once, normalised.
* `D_pred`: mean of 2|Delta error|, 2|Delta ECE|, residual-quantile gap, class-recall gap, confusion gap; floored with `0.5 nu`.
* Features -> groups (default one group per column) -> `D = (1-0.3) mean + 0.3 max`.

**Combination**: `D = sum_{k available} w_k D_k / sum_{k available} w_k`, `w = (0.4, 0.3, 0.1, 0.2)`; no view -> `D = 1`. `completeness = available feature weight / total feature weight`.

**Level 1**: per concept a cached `(d, 4)` matrix; score = mean of `|Delta median|/s, |Delta q10|/s, |Delta q90|/s, |log sigma ratio|`; top-5 by score.

**Intra-concept variability**: `intra <- D` on first update, then `0.8 intra + 0.2 D` where `D` compares the new window with the stored (already merged) fingerprint.

**Thresholds**: `thr_cand = max(0.32, 2.5 intra)`, `thr_high = max(0.18, 1.5 intra)`, margin `D_2 - D_1 >= 0.06`, `conf >= 0.35`.

**Confidence** `c = (1 - D/thr_cand)(1/2 + 1/2 min(1, margin/0.15)) min(1, n/100)(1/2 + completeness/2)(1 - 1/2 min(1, intra/thr_cand)) * (0.85 if no D_pred)`. Hand-built; not calibrated.

**Validation**: up to 3 candidates (incl. the active concept) scored on the labelled post-drift buffer; accept if `acc >= 0.70`, `acc >= acc_old + 0`, `D <= thr`, `D_pred <= 0.6` (if stored predictive fingerprint); pick the highest accuracy. HIGH matches may be served provisionally; failed validation restores the old model object.

**Update**: `std, skew, quantiles, corr, temporal <- (w_a x_a + w_b x_b) / (w_a + w_b)` with `w_a = min(count, 1000)`, `w_b = n`; min/max by min/max; count capped at 2000.

**Eviction**: when `> max_concepts` (50) drop `argmin (uses, last_used_t)` among all except the concept just added.

**Persistence**: the repository serialises fingerprints, never models (models live in `MVEngine.models`, in memory). `LogReg.to_dict` exists but nothing saves or loads it. The pipeline's SQLite store only keeps v1 snapshots.

## 3. Findings

### Confirmed defects (reproduced)
| # | Finding | Evidence |
|---|---|---|
| D1 | **Merge is mathematically wrong for std and quantiles.** Weighted averaging of std ignores between-window mean differences, and averaging quantile functions is not the mixture quantile. | Merging N(0,1) (n=300) with N(5,1) (n=100): pooled std 2.356 vs merged 0.988; q50 0.458 vs 1.257; q90 5.240 vs 2.477. Means are exact. Impact is small while merged windows come from one concept, but the concept memory has no guard against merging drifted windows. |
| D2 | **One global AR(1) factor is applied to statistics it does not describe.** | Observed SE inflation at rho = 0.9 (predicted 4.36 by AR(1)): mean 4.25 (fits), median 3.57, q90 2.91, log-scale 2.87, Spearman 2.73, **lag-1 ACF 0.54** (it *shrinks*; the legacy rule enlarges the floor). Median-over-features rho also hides feature-specific dependence. |
| D3 | **Candidate selection bias in validation.** The best of m candidates is chosen and judged on the same labelled samples. | m = 3 models of identical true accuracy 0.80, n = 30: selected model scores 0.860 on the selection set (+0.060 optimism). |
| D4 | **Worst-group term lets one unstable feature dominate.** | 10 features, one heavy-tailed and non-stationary but irrelevant: D_same q95 0.303 vs 0.115 with mean-only; AUC (same vs 0.5-sigma shift) 0.956 vs 1.000. |

### Potential statistical weaknesses
| # | Finding | Evidence / status |
|---|---|---|
| W1 | `intra` is seeded from one window distance. | Relative SD of the estimate 0.62 after 1 update vs 0.23 after 20; thresholds scale with it (`2.5 intra`), mean only, no upper quantile or sample count. |
| W2 | **Tail-shape blindness.** Level 1 uses median/sigma/q10/q90; `D_stat` excludes q01/q99/std; the quantile sketch has only the 0% and 100% tail points. | Heavy-tail concept (6% x4 outliers, same middle quantiles) vs base: Level 2 `D = 0.094`, below every threshold; at Level 1 base and heavy-tail swap ranks. |
| W3 | Noise floor is a single `kappa` on an aggregate; same constant for location, quantile, correlation and temporal terms. | Does **not** reduce ranking power here (AUC 0.664 vs 0.651 at 0.2 sigma, 0.917 vs 0.902 at 0.4 sigma, n = 40), but a 0.2 sigma shift at n = 40 stays below the same-concept q95 either way, so it is unresolvable by threshold. |
| W4 | PSI is asymmetric (bins from `a` only) and computed from a 21-point sketch (resolution 5%); it and W1 are both functions of the same sketch, so they are not independent evidence. | To be measured in M3. |
| W5 | `D_time` is a mean of unnormalised absolute differences across statistics with different units (rates, bounded correlations, ratios). | To be measured in M3. |
| W6 | Fixed view weights and thresholds (0.4/0.3/0.1/0.2, 0.18/0.32) were chosen from one synthetic family. | To be measured in M4/M5. |
| W7 | Confidence is a product of unrelated heuristics and is not calibrated; similarity, evidence quality and confidence are partly mixed. | To be measured in M5. |
| W8 | Validation uses an absolute accuracy bar (0.70), point estimates, no uncertainty, no task-specific metric, no minimum detectable gain. | To be addressed in M6. |
| W9 | Eviction ignores model performance and recurrence period; a genuine concept with `uses = 1` is first in line. | Survivors in the test: old frequent concept + the two newest; the rule behaves as written, but is not a recurrence-aware policy. |
| W10 | Temporal/correlation merge by averaging has no mathematical justification. | To be addressed in M7. |

### Performance bottlenecks
* `FeatureFingerprint` extraction costs about 1.2 ms per 100 samples x 4 features (3.4 ms at 10 features) vs 0.04 ms for v1; recomputed every 100 stable samples. Two quantile calls and a per-column Python loop dominate.
* `level1` re-stacks the concept vectors on every call (O(C) allocation); there is no persistent index.
* `compare()` recomputes robust sigmas and floors for every pair.

### Already implemented correctly (verified by tests)
* F_X cannot see labels (no `y` parameter); unavailable statistics are NaN and excluded; an all-missing window is distance 1, not 0.
* Constant features, outliers (robust statistics), missing values, categorical frequencies, irregular timestamps.
* Weights renormalise over available views; completeness is reported; thresholds and weights are configurable.
* Shadow validation does not mutate stored models; the active concept is a candidate; rollback restores the previous model.
* Serialisation round trip and v1 migration.

### Research improvements (hypotheses to test)
Statistic-specific effective sample size; empirical same-concept noise profiles; symmetric binning and alternative divergences; standardised temporal statistics; richer Level 1; adaptive top-k and full-scan fallback; robust group aggregation; calibrated confidence; paired/bootstrap validation with selection-bias control; mergeable summaries.

## 4. Prioritised plan

| Milestone | Work | Success criterion (can fail) |
|---|---|---|
| M2 | Per-feature, statistic-specific ESS; empirical block-bootstrap same-concept noise profile; configurable floors | D_same q95 more stable across n in {20..160}, rho, noise, d, with different-concept AUC not lower |
| M3 | Symmetric PSI, JS, energy; tail statistic; standardised temporal; Fisher-z correlations | Pick a metric set by AUC and cost, keep the legacy if nothing beats it |
| M4 | Richer Level 1, adaptive top-k, full-scan fallback, cached index; robust group aggregation; weight modes | Recall@k on look-alike libraries; latency vs 10..1000 concepts |
| M5 | Windowed variability tracker; empirical/conformal thresholds; similarity / evidence / confidence split; calibrated logistic model | Brier, ECE on held-out streams; false-match at high confidence |
| M6 | Paired bootstrap acceptance; Bonferroni or split-confirm for multiple candidates; task-aware metrics | Reduce false recalls without losing recalled-model accuracy |
| M7 | Chan-merged moments, mergeable quantile sketch, per-window history medians for correlation/temporal, versioned records, recurrence-aware eviction | Merge error vs exact; backward-compatible loading |
| M8 | Ablations, 5+ seeds, scaling to 1000 concepts | Report whatever happens, including negative results |
| M9 | Docs, dashboard, final report | |

## 5. Resolution status (added after Milestones 2-8)

| Finding | Status | Where |
|---|---|---|
| D1 merge of std / quantiles | Fixed (opt-in `merge_mode="mergeable"`): exact moments, sketch error 0.01 sigma | `memory.py`, `tests/test_memory.py` |
| D2 one global AR(1) factor for every statistic | Statistic-specific ESS implemented; **it did not improve matching** (same-concept q95 0.249 vs 0.210). The legacy global rule stays the default. An empirical same-concept profile calibrates far better offline but is not used by the engine | `noise.py`, `results/research/e1_noise.md` |
| D3 candidate-selection bias | Fixed (opt-in): Bonferroni and split-confirm protocols hold false accepts at about nominal (4-8%); no accuracy gain on streams | `validation.py` |
| D4 worst-group term | Fixed (opt-in `group_agg="mean"`): AUC >= 0.95 in four scenarios incl. d=30 | `similarity.py` |
| W1 intra from one window | Windowed variability tracker + conformal thresholds implemented; **hurt match recall inside the engine**, so legacy thresholds stay the default | `matcher.py` |
| W2 tail blindness | Tail-ratio term (AUC 0.87 -> 0.94); Level 1 `shape_corr` helps autocorrelated look-alikes but hurts heavy tails | `similarity.py`, `mvfp.py` |
| W3 single kappa | Confirmed invalid across views; analytic floor kept, profile available | `noise.py` |
| W4 asymmetric / redundant PSI | Symmetric binning added; W1 and PSI are complementary (rank correlation 0.63); JS and energy add nothing | `similarity.py` |
| W5 unnormalised temporal | Standardised (Fisher-z / log / arcsine, standard-error units): periodicity AUC 0.74 -> 1.00 | `similarity.py` |
| W6 fixed weights | Calibrated weights on separate seeds: held-out top-1 0.865 vs 0.774; not adopted as default | `exp_retrieval.py` |
| W7 confidence | Split into similarity / evidence quality / score; heuristic is badly calibrated (Brier 0.426); calibrated model Brier 0.199, offline only | `matcher.py` |
| W8 absolute 0.70 bar | Task-aware paired validation (opt-in); a significance-based floor was too strict and was replaced by a point floor | `validation.py` |
| W9 eviction | Recurrence-aware score did **not** beat least-used; LRU is worse | `matcher.py` |
| W10 temporal / correlation merge | Per-window history with median instead of averaging | `memory.py` |
| Performance | Sketch compression 3.5x faster after profiling; fingerprint extraction O(d^2) is the remaining bottleneck at high dimension (14.8 ms at d=30) | `results/research/e12_scaling.md` |
| Eviction of the active concept (found only on real data, ELEC2) | Fixed: the new and the caller's concept are both protected; regression tests added | `matcher.py`, `tests/test_memory.py` |

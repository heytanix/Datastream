# Datastream: research report on uncertainty-aware multi-view matching

All numbers below are produced by code in `datastream/research/` and stored in `results/research/`; the appendix tables are generated from those files. Sections 1-8 use synthetic data; section 9 adds four real datasets (single deterministic runs, no significance tests).

## 1. Verdict on the central hypothesis

> An uncertainty-aware multi-view fingerprint matching mechanism can recognise recurring data-generating concepts more reliably than conventional statistical fingerprint matching, while reducing unnecessary retraining without sacrificing predictive performance.

The hypothesis was tested at two levels, and the answers differ.

**Schema-2.0 multi-view matching versus the original v1 fingerprint: partly supported.** On fresh seeds 11-20 (10 seeds, 18 scenarios, exact paired permutation tests, Holm-corrected), schema-2.0 MV had significantly higher accuracy in 5 of 18 scenarios and was never significantly worse. It is robust to missing features, where v1 collapses (`feature_availability`: v1 trains 8.4 models and recalls none, MV trains 3.0 and recalls 4). Match recall was significantly better in 1 scenario, model counts significantly lower in 2 and higher in 1. In the other scenarios there is no significant difference. It costs more compute in all 18 (about 2.5x per sample).

**The bundle of improvements added in milestones M2-M7 versus schema 2.0: not supported.** On the same fresh seeds the improved system never significantly improved any outcome metric. It was significantly *worse* on accuracy in 3 scenarios (gradual, incremental, ab_gradual: -0.7 to -1.4 points), significantly worse on match recall in 2 (gradual -16 points, incremental -22 points), and slower in 17 of 18. The variant that keeps schema-2.0 thresholds (F) is statistically indistinguishable from the baseline on every outcome except accuracy on gradual and incremental drift (-1.0 to -1.6 points), with significantly fewer models in one scenario (`diff_x_same_y`, -1.7). The hypothesis could have been confirmed by this experiment and was not.

Why the experiment was fair: the variant to test (F) was chosen from an ablation on seeds 1-5 and then confirmed on seeds 11-20 that played no part in the choice; every comparison uses the same streams for both systems.

## 2. What each milestone found

| Milestone | Idea | Verdict | Evidence |
|---|---|---|---|
| M1 audit | Find defects before changing anything | 4 defects confirmed, 10 weaknesses listed | `AUDIT.md`: merging std 0.99 vs true 2.36; selection optimism +0.060; worst-group term inflates D_same q95 0.30 vs 0.12 |
| M2 | Statistic-specific effective sample size | **Not supported** | same-concept q95 0.249 vs 0.210, false flags 11.4% vs 6.4% (legacy global AR(1) is better) |
| M2 | Empirical same-concept noise profile | Supported offline, **not wired into the engine** | worst-cell q95 0.28 vs 0.64; false flags 0.1% vs 6.4%; mean AUC 0.856 vs 0.874 |
| M2 | One kappa for every view | **Not valid** | legacy same-concept q95: statistics 0.19, distribution 0.32, temporal 0.05 (temporal is almost entirely floored away) |
| M2 | Floor hides small real changes | **Not observed** | ranking power (AUC) is the same with or without the floor |
| M3 | Wasserstein and PSI are redundant | **Refuted**: complementary | rank correlation 0.63; W1 weak on shape (uniform 0.60, bimodal 0.63), PSI weak on scale (0.87) and Laplace (0.74); together mean AUC 0.92 |
| M3 | JS / energy distance add value | **No** | JS = PSI within noise (0.923 vs 0.924); energy worse (0.83-0.85) |
| M3 | Symmetric (pooled-bin) PSI | Marginally better, inside the noise | 0.924 vs 0.918, SE of AUC about 0.03 |
| M3 | Tail term | Supported | heavy tail with unchanged core: AUC 0.87 -> 0.94; bimodal 0.77 -> 0.95 |
| M3 | Standardised temporal statistics | Supported | periodicity AUC 0.74 -> 1.00; ordering, trend 1.00; abrupt-vs-gradual stays hard (0.62) |
| M3 | Fisher-z correlation distance | Supported | e.g. d=30, n=40, r=0.3: AUC 0.83 -> 0.94 |
| M4 | Richer Level 1 | **Mixed** | `shape_corr` R@1 0.68 vs 0.53 but heavy-tail R@5 1.00 -> 0.71; `shape` alone is worse (R@5 0.87 vs 0.95) |
| M4 | Adaptive top-k | Supported | candidate recall 0.996 with 5.5 candidates on average, equal to a full scan |
| M4 | Calibrated view weights | Supported on held-out seeds | top-1 0.865 vs 0.774; equal thirds 0.857; stat-dist dependence 0.73, time independent |
| M4 | Robust group aggregation | Supported: plain mean | AUC >= 0.95 in four scenarios (also at d=30) vs 0.70-0.93 for the worst-group blend |
| M5 | Per-concept conformal thresholds | Supported standalone, **hurt in the engine** | better operating point in 3 of 4 conditions offline; in the engine F (legacy thresholds) recovers match recall 75.6% -> 87.1% |
| M5 | Heuristic confidence as a probability | **Not calibrated** | Brier 0.426 vs 0.249 for the base rate; ECE 0.44 |
| M5 | Calibrated logistic probability | Supported offline | Brier 0.199, ECE 0.064, AUROC 0.76; precision 1.00 at p >= 0.9 (n = 22) |
| M6 | Paired CI validation | Supported in simulation, **no benefit on streams** | false-accept 4-5% (Bonferroni) vs 72-78% (legacy point rule); power 71% vs 100% at n=60 for a +0.20 gain; streams: no accuracy gain |
| M6 | Significant-majority floor | Too strict | lost recalls (retrieval recall 73% -> 31%); replaced by a point-estimate floor |
| M7 | Mergeable summaries | Supported (correctness) | std and skew exact; quantile error 0.01 sigma vs 0.35-0.72 sigma; cost 1.2 ms vs 0.6 ms per update, 9.7 vs 1.5 KB per concept |
| M7 | Recurrence-aware eviction | **Not supported** | score = least-used (64.0 vs 64.0 re-creations); LRU worse (83.7); grace period worse (112.7) |
| M8 | Confidence gating | **No effect** | variant G identical to D on every metric |
| M8 | Noise correction | **Clearly needed** | removing it: false-match 0.5% -> 6.2%, models 5.3 -> 7.5 |
| M8 | Predictive validation | **Clearly needed** | removing it: post-drift accuracy 78.9% -> 72.0%; overlapping scenario precision 40%, false-match 50% |

## 3. The improved matching pipeline

```
drift signal (error alarm | label-free feature shift)
 -> feature fingerprint F_X (statistics, distributions, temporal; NaN = unavailable; no labels)
 -> Level 1: cached vectorised index, adaptive k (3..10), full-scan fallback when the shortlist says "new"
 -> Level 2: per-view distances
      stat : robust location/spread/skew/missing (+ optional tail ratio), categorical TV
      dist : Wasserstein-1 + PSI on a symmetric bin set (optional JS / energy), rank correlations (optional Fisher-z)
      time : transformed statistics (Fisher-z, log, arcsine) in standard-error units
 -> noise and uncertainty: analytic floor with global or per-statistic ESS, or an empirical same-concept profile
 -> aggregation: mean over feature groups (alternatives: median, trimmed, quantile, weighted, mean+worst)
 -> weights: availability-normalised (default), fixed with a missing prior, or calibrated on separate data
 -> candidate ranking; thresholds: legacy max(t, k x intra) or per-concept conformal quantiles
 -> three separate quantities: similarity (1 - D), evidence quality, confidence score; optional calibrated probability
 -> structured explanation (MatchResult.explanation())
 -> validation: point rule (default) | paired CI with Bonferroni / split-confirm | none (ablation only)
 -> safe recall: shadow evaluation, provisional activation, rollback, false-alarm path
 -> repository update: legacy averaging | mergeable moments + sketches + per-window history; eviction policy; versioned records
```

Every stage is configuration; all defaults reproduce schema 2.0 exactly (verified by a regression freeze in `tests/test_noise.py` and by identical benchmark output). The improved configuration used in the ablation is `datastream.research.presets.ablation_variants()["D full multi-view (improved)"]`.

## 4. Defects found by this work and fixed

1. Merging concept fingerprints by weighted averaging gave std 0.99 instead of 2.36 and q90 2.48 instead of 5.24 (now exact moments and a mergeable sketch).
2. An all-missing window scored distance 0 (a perfect match); now unavailable (found earlier, by a test).
3. A candidate list could be empty under gradual drift (found by the benchmark).
4. My first "unresolvable window" rule flagged 100% of windows because it tested the un-floored noise level; it is now a power criterion (can a 1-sigma change be detected at this window size?), and is only applied to the statistics and distribution views (the probe is a location shift, so it has no power on the temporal view by construction).
5. The first trivial-predictor floor in paired validation rejected valid recalls on short windows; replaced by a point-estimate floor after measuring the cause.
7. On real data (ELEC2) the repository could evict the active concept once it exceeded its 50-concept cap (details in section 9); fixed.
6. v1's fingerprint produced NaN on streams with missing features; the learner now imputes means. The v1 engine is left as the baseline and remains broken on that scenario, which is reported.

## 5. Statistical protocol

* Seeds: 1-5 for exploration and ablation; 11-20 (not seen during design) for the confirmation. Mean and standard deviation over seeds everywhere.
* Calibration versus evaluation: thresholds, view weights and the confidence calibrator are fitted on separate seeds / separate RNG streams from the data they are evaluated on.
* Significance: exact paired sign-flip permutation test on per-seed differences (2^10 patterns, smallest attainable p = 2/1024), two-sided, Holm-corrected across the 18 scenarios per metric. No significance claim is made anywhere else.
* AUC differences smaller than about 0.03 are inside Monte-Carlo noise (60 pairs) and are called ties.
* Bootstrap intervals in paired validation are mildly anti-conservative at n <= 40 (measured false-accept 6-9% at nominal 5%).

## 6. Performance and memory

Fingerprint extraction: 1.09 ms (40 samples, 4 features), 1.20 ms (100, 4), 3.4 ms (100, 10), 14.8 ms (100, 30): the correlation term is O(d^2), so high-dimensional streams need `max_corr_pairs` or fewer correlation pairs. v1: 0.05 ms.
Whole engine on `abrupt_recurring`: v1 19 us per sample, MV schema 2.0 49 us, improved 77 us (empirical thresholds add about 18 us per sample, mergeable memory about 5 us).
Repository: Level 1 grows from 0.07 ms (10 concepts) to 0.37 ms (1000); the whole match stays at 2-4 ms because only the shortlist is scored by Level 2, while an exhaustive scan costs 6.6 ms (10 concepts) to 649 ms (1000). The default cap of 50 concepts is therefore a retention limit, not a latency requirement. Memory per concept: 136 bytes (v1), 1.5 KB (MV 2.0), 9.7 KB (mergeable, d=4; 29 KB at d=10); 1000 concepts of the 2.0 kind take 7.9 MB.

## 7. What is implemented and tested, and what is not

| Item | Status |
|---|---|
| Audit with reproducible measurements | implemented, measured |
| Statistic-specific ESS, empirical noise profile (`noise.py`) | implemented and tested; profile is **not used by the engine** |
| Divergences, tail term, standardised temporal, Fisher-z, aggregation and weight modes (`similarity.py`) | implemented and tested |
| Level 1 representations, adaptive k, cached index, fallback scan | implemented and tested |
| Variability tracker, conformal thresholds, three-quantity result, calibrator, explanations | implemented and tested; calibrator is **not applied inside the engine** |
| Task-aware paired validation, selection protocols (`validation.py`); paired mode in the engine | implemented and tested; engine supports classification only |
| Mergeable memory, versioned records, eviction policies (`memory.py`) | implemented and tested; 2.0 and v1 dumps load |
| 18 controlled scenarios, ablation, paired tests, scaling | implemented and run |
| Dataset loaders (ARFF, Intel Berkeley, UCI load diagrams) | implemented, unit-tested on toy files, **and run on the four real datasets** (single runs) |
| Regression and anomaly validation | implemented and tested in `validation.py`; not usable inside the engine (its learner is a classifier) |
| Spectral temporal features, rolling-stability features | **not implemented** |
| KLL sketch | **not implemented**; a t-digest-style sketch is used (accuracy 0.3% of range after 40 merges at 34 centroids) |

## 8. Known limitations

* Synthetic results are low-dimensional (d = 3-10); the four real runs are single deterministic runs with chosen feature subsets and (for two of them) derived labels, so they illustrate behaviour but do not establish generalisation. Airlines shows adaptation can hurt.
* The improved bundle was not better than schema 2.0 on these streams. Individual components are worth keeping for specific reasons (noise correction, predictive validation, exact merging, calibrated probabilities offline), others are not (per-statistic ESS, score eviction, in-engine empirical thresholds, paired validation as an accuracy gain).
* Offline operating points do not transfer automatically to the engine: thresholds that give a better standalone trade-off lose recalls inside a pipeline where validation already removes false matches.
* With about 10 same-concept observations a conformal threshold cannot certify alpha = 0.02 (it needs at least 49); the implementation widens conservatively.
* Concepts that differ only in p(y|x) with identical p(x) (SEA, A vs A') need labels; feature-only fingerprints cannot separate them. Windows inside gradual or incremental transitions are mixtures and are matched poorly by every variant.
* A calibrated probability is only as good as its calibration data; this one was fitted on one synthetic family.
* Statistical validation needs labels and enough of them: at n = 30 the power for a +0.20 improvement is 27-39% for the controlled protocols.

## 9. Real datasets

Downloaded with your approval into `data/` (git-ignored): `electricity.arff` (OpenML 151, 3.1 MB), `airlines.arff` (OpenML 1169, 19.2 MB), the Intel Berkeley lab `data.txt.gz` (34.4 MB, 150.9 MB unpacked) and the UCI `LD2011_2014.txt` (261 MB zipped, 711 MB unpacked). Archives were inspected before extraction. Results are in appendix A11 (`results/research/e14_real.json`).

**What was run.** Natural streams, in file order, one deterministic run per variant (so no variance and no significance test is possible): ELEC2 (all 45,312 rows; `date` dropped because it is a monotone counter), Airlines (first 100,000 rows; features Airline code, Flight, DayOfWeek, Time, Length; airports omitted; the file is not strictly chronological), Intel Berkeley mote 1 (first 43,017 usable readings; raw glitches kept) and UCI load diagrams (6 clients, first 70,000 intervals). Intel and UCI have no class label: the labels are derived (temperature rises over the next 30 readings; total demand rises next interval) and prove nothing about the existence of concepts. Constructed recurrence: two real chronological blocks of 6,000 rows replayed as A B A B A; the repetition is an exact replay.

**What it shows (single runs; differences of 1-2 points cannot be called significant).**
* ELEC2: every adaptive strategy beats the static model (0.650); MV 2.0 0.758, improved 0.765, v1 0.742, retrain 0.744 (majority 0.575). Retrain trains 189 models, the improved system 67.
* Airlines: **every adaptive strategy is worse than the static model** (0.575-0.590 vs 0.641, which equals the majority rate). With these weak features, adapting on short windows adds noise; static wins. MV 2.0 trains 488 models.
* Intel (derived label): v1 is best (0.784), MV 2.0 0.750, improved 0.766; the improved system raises 79 false alarms.
* UCI (derived label): all adaptive variants are about 0.80 (static 0.763); the first half is trivially 1.000.
* Constructed recurrence is mixed: on ELEC2 MV 2.0 recalls best (match recall 0.714, false-match 1.2%, versus v1 0.631 and 4.3%) and the improved system is worse (0.595); on Intel the improved system recalls best (0.661 against 0.084 for MV 2.0 and 0.534 for v1); on UCI almost nothing is recalled by anyone (MV 0.286, v1 0).

No variant wins consistently on real data, and the synthetic conclusions neither transfer cleanly nor contradict. The result that matters most is the bug below.

**A defect that only real data exposed.** On ELEC2 the engine created more than 50 concepts. `ConceptRepository.add` protected only one concept from eviction, so when the engine passed the previous concept as the protected one the newly created (about to be active) concept could be evicted at once, and the next lookup raised `KeyError`. No synthetic scenario ever exceeded the cap. Fixed (the new concept and the caller's concept are both protected); regression tests added, including an engine run with a cap of 2.

## 10. Reproduce

```bash
.venv/bin/python -m unittest discover -s tests                      # 185 tests
.venv/bin/python -m datastream.research.audit_checks                # M1
.venv/bin/python -m datastream.research.exp_noise                   # M2
.venv/bin/python -m datastream.research.exp_dist_time               # M3
.venv/bin/python -m datastream.research.exp_retrieval               # M4
.venv/bin/python -m datastream.research.exp_thresholds              # M5
.venv/bin/python -m datastream.research.exp_validation              # M6
.venv/bin/python -m datastream.research.exp_memory                  # M7
.venv/bin/python -m datastream.research.exp_ablation                # M8, about 9 minutes
.venv/bin/python -m datastream.research.exp_paired                  # M8 confirmation, about 8 minutes
.venv/bin/python -m datastream.research.exp_scaling                 # M8 scaling
.venv/bin/python -m datastream.research.exp_real                    # real data (needs data/ populated), about 4 minutes
.venv/bin/python -m datastream.research.build_report                # regenerate docs/RESEARCH.md
```


---

# Appendix: all measured tables (generated from `results/research/`)

## A1. Milestone 1: audit measurements
See [`AUDIT.md`](AUDIT.md) and `results/research/audit.json`.

## A2. Noise model and effective sample size (E1)

| config | mean q95 of D_same | worst-cell q95 | range of q95 (stability) | mean AUC | worst AUC | mean D_diff median | false-flag @0.32 | false-flag @0.18 |
|---|---|---|---|---|---|---|---|---|
| legacy | 0.210 | 0.638 | 0.577 | 0.874 | 0.388 | 0.259 | 0.064 | 0.177 |
| per_stat | 0.249 | 0.716 | 0.644 | 0.878 | 0.415 | 0.305 | 0.114 | 0.279 |
| profile | 0.058 | 0.277 | 0.277 | 0.856 | 0.421 | 0.128 | 0.001 | 0.010 |
| none | 0.498 | 0.872 | 0.627 | 0.876 | 0.461 | 0.545 | 0.707 | 0.985 |

mean q95 of D_same by rho:

| rho | legacy | per_stat | profile | none |
|---|---|---|---|---|
| -0.5 | 0.139 | 0.174 | 0.034 | 0.409 |
| 0.0 | 0.153 | 0.160 | 0.036 | 0.421 |
| 0.5 | 0.177 | 0.219 | 0.053 | 0.490 |
| 0.9 | 0.372 | 0.445 | 0.109 | 0.673 |

mean q95 of D_same by n:

| n | legacy | per_stat | profile | none |
|---|---|---|---|---|
| 20 | 0.315 | 0.353 | 0.074 | 0.653 |
| 40 | 0.239 | 0.282 | 0.067 | 0.551 |
| 80 | 0.167 | 0.209 | 0.046 | 0.444 |
| 160 | 0.120 | 0.154 | 0.045 | 0.345 |

mean q95 of D_same by d:

| d | legacy | per_stat | profile | none |
|---|---|---|---|---|
| 2 | 0.235 | 0.276 | 0.085 | 0.513 |
| 4 | 0.207 | 0.244 | 0.058 | 0.495 |
| 10 | 0.188 | 0.227 | 0.031 | 0.487 |

mean q95 of D_same by noise:

| noise | legacy | per_stat | profile | none |
|---|---|---|---|---|
| 0.0 | 0.208 | 0.265 | 0.067 | 0.517 |
| 0.5 | 0.214 | 0.255 | 0.054 | 0.502 |
| 1.0 | 0.209 | 0.228 | 0.053 | 0.477 |


## A3. Distribution metrics, tail term, temporal statistics, correlations (E2, E3)

### Distribution metrics (D_dist view, n=80, AUC same-vs-different by family)

| metric | uniform | bimodal | laplace | skewed | heavy_tail | shift_0.5 | scale_1.4 | mean AUC | same q95 | µs/compare |
|---|---|---|---|---|---|---|---|---|---|---|
| A wasserstein | 0.60 | 0.63 | 0.87 | 0.99 | 0.99 | 0.99 | 0.92 | 0.857 | 0.193 | 339 |
| B psi (a-deciles, schema 2.0) | 0.76 | 0.99 | 0.74 | 1.00 | 0.99 | 0.95 | 0.87 | 0.899 | 0.266 | 387 |
| B' psi (pooled bins) | 0.76 | 0.99 | 0.76 | 1.00 | 0.99 | 0.96 | 0.92 | 0.911 | 0.285 | 503 |
| C wasserstein+psi (schema 2.0) | 0.75 | 0.96 | 0.82 | 1.00 | 1.00 | 0.98 | 0.91 | 0.918 | 0.226 | 394 |
| C' wasserstein+psi (pooled) | 0.74 | 0.97 | 0.84 | 1.00 | 1.00 | 0.98 | 0.94 | 0.924 | 0.230 | 510 |
| D wasserstein+JS | 0.74 | 0.97 | 0.84 | 1.00 | 1.00 | 0.98 | 0.93 | 0.923 | 0.217 | 514 |
| JS only | 0.75 | 0.99 | 0.75 | 1.00 | 0.99 | 0.96 | 0.91 | 0.909 | 0.260 | 512 |
| E wasserstein+energy | 0.59 | 0.64 | 0.87 | 0.98 | 0.99 | 0.99 | 0.88 | 0.849 | 0.162 | 373 |
| energy only | 0.56 | 0.65 | 0.85 | 0.97 | 0.98 | 0.99 | 0.78 | 0.827 | 0.132 | 364 |

Spearman correlation of Wasserstein and PSI distances over all pairs: 0.629

### Tail term in D_stat (AUC of the statistics view)

| case | AUC | same q95 |
|---|---|---|
| heavy_tail_core tail_weight=0.0 | 0.87 | 0.066 |
| heavy_tail_core tail_weight=1.0 | 0.94 | 0.083 |
| heavy_tail_core tail_weight=2.0 | 0.94 | 0.115 |
| heavy_tail tail_weight=0.0 | 1.00 | 0.066 |
| heavy_tail tail_weight=1.0 | 1.00 | 0.083 |
| heavy_tail tail_weight=2.0 | 1.00 | 0.115 |
| laplace tail_weight=0.0 | 0.96 | 0.066 |
| laplace tail_weight=1.0 | 0.99 | 0.083 |
| laplace tail_weight=2.0 | 0.99 | 0.115 |
| bimodal tail_weight=0.0 | 0.77 | 0.066 |
| bimodal tail_weight=1.0 | 0.95 | 0.083 |
| bimodal tail_weight=2.0 | 0.98 | 0.115 |

### Temporal tasks (n=120)

| task | AUC legacy time | same q95 legacy time | AUC standardised time (legacy ESS) | same q95 standardised time (legacy ESS) | AUC standardised time (per-stat ESS) | same q95 standardised time (per-stat ESS) | combined AUC stat+dist only | combined AUC stat+dist+time (w=0.1, legacy time) | combined AUC stat+dist+time (w=0.3, standardised) |
|---|---|---|---|---|---|---|---|---|---|
| ordering (iid vs AR 0.9) | 1.00 | 0.00 | 1.00 | 0.20 | 1.00 | 0.21 | 0.51 | 0.87 | 1.00 |
| periodicity (period 12 vs 25) | 0.74 | 0.00 | 1.00 | 0.00 | 1.00 | 0.16 | 0.50 | 0.52 | 1.00 |
| trend (up vs down, same quantiles) | 0.98 | 0.00 | 1.00 | 0.10 | 1.00 | 0.25 | 0.49 | 0.98 | 1.00 |
| trend (flat vs up) | 1.00 | 0.00 | 1.00 | 0.15 | 1.00 | 0.16 | 0.78 | 0.94 | 1.00 |
| abrupt vs gradual (step vs ramp) | 0.50 | 0.00 | 0.63 | 0.08 | 0.62 | 0.24 | 0.53 | 0.52 | 0.63 |

### Correlation comparison (AUC of D_dist, same marginals / different dependence)

| case | AUC | same q95 |
|---|---|---|
| d=4 n=40 corr=0.3 abs | 0.64 | 0.256 |
| d=4 n=40 corr=0.3 fisher | 0.72 | 0.298 |
| d=4 n=40 corr=0.6 abs | 0.85 | 0.293 |
| d=4 n=40 corr=0.6 fisher | 0.89 | 0.304 |
| d=4 n=80 corr=0.3 abs | 0.80 | 0.211 |
| d=4 n=80 corr=0.3 fisher | 0.83 | 0.222 |
| d=4 n=80 corr=0.6 abs | 0.98 | 0.214 |
| d=4 n=80 corr=0.6 fisher | 0.99 | 0.231 |
| d=10 n=40 corr=0.3 abs | 0.72 | 0.250 |
| d=10 n=40 corr=0.3 fisher | 0.81 | 0.266 |
| d=10 n=40 corr=0.6 abs | 0.96 | 0.255 |
| d=10 n=40 corr=0.6 fisher | 0.99 | 0.275 |
| d=10 n=80 corr=0.3 abs | 0.86 | 0.199 |
| d=10 n=80 corr=0.3 fisher | 0.91 | 0.206 |
| d=10 n=80 corr=0.6 abs | 1.00 | 0.178 |
| d=10 n=80 corr=0.6 fisher | 1.00 | 0.181 |
| d=30 n=40 corr=0.3 abs | 0.83 | 0.262 |
| d=30 n=40 corr=0.3 fisher | 0.94 | 0.267 |
| d=30 n=40 corr=0.6 abs | 1.00 | 0.272 |
| d=30 n=40 corr=0.6 fisher | 1.00 | 0.277 |
| d=30 n=80 corr=0.3 abs | 0.97 | 0.199 |
| d=30 n=80 corr=0.3 fisher | 0.99 | 0.202 |
| d=30 n=80 corr=0.6 abs | 1.00 | 0.189 |
| d=30 n=80 corr=0.6 fisher | 1.00 | 0.197 |

missing values + a constant feature handled: True


## A4. Level-1 retrieval, view ablation, weights, aggregation (E4)

### Level 1 candidate recall (42 concepts, 6 look-alike families x 7; 3 seeds, mean±std)

| representation | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| basic | 0.528±0.017 | 0.845±0.005 | 0.948±0.010 | 0.998±0.003 |
| shape | 0.401±0.016 | 0.712±0.017 | 0.871±0.014 | 0.970±0.015 |
| shape_corr | 0.683±0.018 | 0.899±0.008 | 0.956±0.012 | 0.994±0.000 |
| full | 0.683±0.018 | 0.899±0.008 | 0.956±0.012 | 0.994±0.000 |

Recall@5 by look-alike family:

| representation | core | heavy_tail | bimodal | correlated | autocorr | shifted | wider |
|---|---|---|---|---|---|---|---|
| basic | 1.00 | 1.00 | 1.00 | 1.00 | 0.64 | 1.00 | 1.00 |
| shape | 0.99 | 0.43 | 1.00 | 1.00 | 0.69 | 1.00 | 0.99 |
| shape_corr | 1.00 | 0.71 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |
| full | 1.00 | 0.71 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |

Recall@5 by query kind:

| representation | plain | shifted | missing_feature | noisy |
|---|---|---|---|---|
| basic | 0.95 | 0.94 | 0.96 | 0.94 |
| shape | 0.87 | 0.89 | 0.85 | 0.88 |
| shape_corr | 0.96 | 0.99 | 0.91 | 0.96 |
| full | 0.96 | 0.99 | 0.91 | 0.96 |

### End-to-end top-1 (shortlist then Level 2)

| configuration | top-1 | mean candidates | candidate recall |
|---|---|---|---|
| legacy L2 | full scan | 0.722±0.020 | all | 1.000 |
| legacy L2 | basic fixed k=5 (schema 2.0) | 0.704±0.022 | 5.0 | 0.948 |
| legacy L2 | basic adaptive | 0.722±0.020 | 5.5 | 0.996 |
| legacy L2 | shape_corr fixed k=5 | 0.774±0.029 | 5.0 | 0.956 |
| legacy L2 | shape_corr adaptive | 0.760±0.030 | 7.6 | 0.990 |
| legacy L2 | full adaptive | 0.760±0.030 | 7.6 | 0.990 |
| improved L2 | full scan | 0.774±0.025 | all | 1.000 |
| improved L2 | basic fixed k=5 (schema 2.0) | 0.752±0.028 | 5.0 | 0.948 |
| improved L2 | basic adaptive | 0.774±0.025 | 5.5 | 0.996 |
| improved L2 | shape_corr fixed k=5 | 0.823±0.034 | 5.0 | 0.956 |
| improved L2 | shape_corr adaptive | 0.817±0.034 | 7.6 | 0.990 |
| improved L2 | full adaptive | 0.817±0.034 | 7.6 | 0.990 |

### View ablation (improved Level 2, full scan, held-out seeds)

| views | top-1 |
|---|---|
| stat only | 0.349 |
| dist only | 0.488 |
| time only | 0.067 |
| stat+dist | 0.484 |
| stat+time | 0.683 |
| dist+time | 0.865 |
| stat+dist+time (0.4/0.3/0.1) | 0.774 |
| equal thirds | 0.857 |

Calibrated weights (grid search on separate calibration seeds): [0.4, 0.2, 0.4] -> calibration top-1 0.861, held-out 0.865 (fixed 0.4/0.3/0.1 held-out 0.774)

View dependence (Spearman of distances): {'stat-dist': 0.7294917409400425, 'stat-time': -0.03924613429486858, 'dist-time': -0.13834391643157765}

### Group aggregation (AUC of D_stat; same-concept q95 in brackets)

| scenario | mean_worst (0.7/0.3, schema 2.0) | mean | median | trimmed (20%) | quantile 0.75 | quantile 0.9 |
|---|---|---|---|---|---|---|
| A: unstable noise feature; 9 features shift 0.5 | 0.87 (0.35) | 1.00 (0.12) | 1.00 (0.01) | 1.00 (0.03) | 1.00 (0.09) | 1.00 (0.25) |
| B: unstable noise feature; ONE feature shifts 1.5 | 0.72 (0.36) | 0.97 (0.12) | 0.60 (0.02) | 0.79 (0.03) | 0.80 (0.11) | 0.99 (0.27) |
| C: no unstable feature; ONE feature shifts 1.5 | 1.00 (0.09) | 1.00 (0.04) | 0.54 (0.02) | 0.69 (0.02) | 0.69 (0.06) | 0.90 (0.14) |
| D: no unstable feature; 9 features shift 0.5 | 1.00 (0.08) | 1.00 (0.04) | 1.00 (0.02) | 1.00 (0.02) | 1.00 (0.06) | 1.00 (0.12) |


## A5. Thresholds and confidence calibration (E7, E8)

### Threshold policies (improved Level 2; calibration windows from separate RNG streams; 2 seeds; mean)


**d=4 n=40**

| policy | correct recall | wrong concept | missed match | false match (novel) |
|---|---|---|---|---|
| fixed 0.10 | 0.518 | 0.042 | 0.440 | 0.173 |
| fixed 0.15 | 0.738 | 0.077 | 0.185 | 0.476 |
| fixed 0.20 | 0.815 | 0.125 | 0.060 | 0.815 |
| fixed 0.25 | 0.839 | 0.155 | 0.006 | 0.952 |
| fixed 0.30 | 0.839 | 0.155 | 0.006 | 0.988 |
| fixed 0.40 | 0.845 | 0.155 | 0.000 | 1.000 |
| legacy 2.5 x intra (floor 0.32) | 0.845 | 0.155 | 0.000 | 1.000 |
| conformal per-concept alpha=0.02 | 0.750 | 0.083 | 0.167 | 0.530 |
| conformal global pool alpha=0.02 | 0.833 | 0.143 | 0.024 | 0.905 |
| conformal per-concept alpha=0.05 | 0.750 | 0.083 | 0.167 | 0.530 |
| conformal global pool alpha=0.05 | 0.804 | 0.125 | 0.071 | 0.821 |
| conformal per-concept alpha=0.1 | 0.667 | 0.060 | 0.274 | 0.315 |
| conformal global pool alpha=0.1 | 0.774 | 0.101 | 0.125 | 0.696 |

**d=4 n=80**

| policy | correct recall | wrong concept | missed match | false match (novel) |
|---|---|---|---|---|
| fixed 0.10 | 0.690 | 0.042 | 0.268 | 0.250 |
| fixed 0.15 | 0.810 | 0.077 | 0.113 | 0.577 |
| fixed 0.20 | 0.869 | 0.089 | 0.042 | 0.815 |
| fixed 0.25 | 0.887 | 0.107 | 0.006 | 0.958 |
| fixed 0.30 | 0.893 | 0.107 | 0.000 | 0.988 |
| fixed 0.40 | 0.893 | 0.107 | 0.000 | 1.000 |
| legacy 2.5 x intra (floor 0.32) | 0.893 | 0.107 | 0.000 | 0.988 |
| conformal per-concept alpha=0.02 | 0.839 | 0.065 | 0.095 | 0.452 |
| conformal global pool alpha=0.02 | 0.869 | 0.095 | 0.036 | 0.839 |
| conformal per-concept alpha=0.05 | 0.839 | 0.065 | 0.095 | 0.452 |
| conformal global pool alpha=0.05 | 0.857 | 0.089 | 0.054 | 0.690 |
| conformal per-concept alpha=0.1 | 0.774 | 0.054 | 0.173 | 0.298 |
| conformal global pool alpha=0.1 | 0.810 | 0.077 | 0.113 | 0.524 |

**d=10 n=40**

| policy | correct recall | wrong concept | missed match | false match (novel) |
|---|---|---|---|---|
| fixed 0.10 | 0.393 | 0.000 | 0.607 | 0.048 |
| fixed 0.15 | 0.708 | 0.024 | 0.268 | 0.387 |
| fixed 0.20 | 0.827 | 0.095 | 0.077 | 0.756 |
| fixed 0.25 | 0.881 | 0.113 | 0.006 | 0.917 |
| fixed 0.30 | 0.887 | 0.113 | 0.000 | 0.982 |
| fixed 0.40 | 0.887 | 0.113 | 0.000 | 1.000 |
| legacy 2.5 x intra (floor 0.32) | 0.887 | 0.113 | 0.000 | 0.988 |
| conformal per-concept alpha=0.02 | 0.839 | 0.054 | 0.107 | 0.506 |
| conformal global pool alpha=0.02 | 0.875 | 0.113 | 0.012 | 0.869 |
| conformal per-concept alpha=0.05 | 0.839 | 0.054 | 0.107 | 0.506 |
| conformal global pool alpha=0.05 | 0.827 | 0.101 | 0.071 | 0.780 |
| conformal per-concept alpha=0.1 | 0.750 | 0.000 | 0.250 | 0.214 |
| conformal global pool alpha=0.1 | 0.815 | 0.077 | 0.107 | 0.649 |

**d=10 n=80**

| policy | correct recall | wrong concept | missed match | false match (novel) |
|---|---|---|---|---|
| fixed 0.10 | 0.589 | 0.042 | 0.369 | 0.119 |
| fixed 0.15 | 0.792 | 0.060 | 0.149 | 0.506 |
| fixed 0.20 | 0.893 | 0.071 | 0.036 | 0.696 |
| fixed 0.25 | 0.923 | 0.077 | 0.000 | 0.875 |
| fixed 0.30 | 0.923 | 0.077 | 0.000 | 0.982 |
| fixed 0.40 | 0.923 | 0.077 | 0.000 | 1.000 |
| legacy 2.5 x intra (floor 0.32) | 0.923 | 0.077 | 0.000 | 1.000 |
| conformal per-concept alpha=0.02 | 0.905 | 0.054 | 0.042 | 0.381 |
| conformal global pool alpha=0.02 | 0.905 | 0.077 | 0.018 | 0.762 |
| conformal per-concept alpha=0.05 | 0.905 | 0.054 | 0.042 | 0.381 |
| conformal global pool alpha=0.05 | 0.869 | 0.065 | 0.065 | 0.637 |
| conformal per-concept alpha=0.1 | 0.821 | 0.048 | 0.131 | 0.226 |
| conformal global pool alpha=0.1 | 0.821 | 0.060 | 0.119 | 0.589 |

### Confidence calibration (held-out seeds)

calibration matches 735, test matches 502, base rate of correct best candidate: cal 0.533, test 0.540

constant-base-rate Brier: 0.2485

| score | Brier | ECE | AUROC |
|---|---|---|---|
| heuristic_score | 0.4256 | 0.4408 | 0.648 |
| calibrated_probability | 0.1988 | 0.0642 | 0.759 |

Reliability (calibrated probability):

| bin | n | mean predicted | observed |
|---|---|---|---|
| 0.1-0.2 | 19 | 0.163 | 0.474 |
| 0.2-0.3 | 76 | 0.260 | 0.289 |
| 0.3-0.4 | 78 | 0.354 | 0.256 |
| 0.4-0.5 | 74 | 0.452 | 0.405 |
| 0.5-0.6 | 78 | 0.548 | 0.538 |
| 0.6-0.7 | 50 | 0.638 | 0.800 |
| 0.7-0.8 | 59 | 0.745 | 0.763 |
| 0.8-0.9 | 46 | 0.849 | 0.891 |
| 0.9-1.0 | 22 | 0.943 | 1.000 |

Reliability (heuristic score):

| bin | n | mean predicted | observed |
|---|---|---|---|
| 0.0-0.1 | 300 | 0.044 | 0.447 |
| 0.1-0.2 | 137 | 0.140 | 0.591 |
| 0.2-0.3 | 50 | 0.237 | 0.820 |
| 0.3-0.4 | 14 | 0.355 | 1.000 |
| 0.4-0.5 | 1 | 0.426 | 1.000 |

Precision and false-match rate above a confidence cut-off:

| score | cut-off | n accepted | precision | share of novel-concept queries accepted |
|---|---|---|---|---|
| heuristic_score | 0.3 | 15 | 1.000 | 0.000 |
| heuristic_score | 0.5 | 0 | n/a | 0.000 |
| heuristic_score | 0.7 | 0 | n/a | 0.000 |
| heuristic_score | 0.9 | 0 | n/a | 0.000 |
| calibrated_probability | 0.3 | 407 | 0.590 | 0.746 |
| calibrated_probability | 0.5 | 255 | 0.745 | 0.306 |
| calibrated_probability | 0.7 | 127 | 0.850 | 0.088 |
| calibrated_probability | 0.9 | 22 | 1.000 | 0.000 |

Calibrator coefficients (standardised features): {'D': -0.3856298482108158, 'margin_norm': 0.64813865889035, 'log_n_eff': -0.491631310620451, 'completeness': 0.0, 'var_ratio': 0.08041839910629082, 'has_pred': 0.0, 'bias': 0.20787973570684115}


## A6. Validation protocols (E9)

### Selection protocols (simulation, 250 trials per cell, 3 candidates)

| scenario | protocol | accept rate | accepts the truly better candidate | optimism of the reported metric |
|---|---|---|---|---|
| null (no candidate better) | n=30 | legacy point rule (acc>=0.70, >= baseline, pick best) | 0.780 | 0.000 | +0.090 |
| null (no candidate better) | n=30 | pick_best + paired CI (biased) | 0.128 | 0.000 | +0.133 |
| null (no candidate better) | n=30 | bonferroni + paired CI | 0.048 | 0.000 | +0.150 |
| null (no candidate better) | n=30 | split_confirm + paired CI | 0.080 | 0.000 | +0.113 |
| null (no candidate better) | n=60 | legacy point rule (acc>=0.70, >= baseline, pick best) | 0.744 | 0.000 | +0.063 |
| null (no candidate better) | n=60 | pick_best + paired CI (biased) | 0.124 | 0.000 | +0.096 |
| null (no candidate better) | n=60 | bonferroni + paired CI | 0.040 | 0.000 | +0.108 |
| null (no candidate better) | n=60 | split_confirm + paired CI | 0.044 | 0.000 | +0.064 |
| null (no candidate better) | n=100 | legacy point rule (acc>=0.70, >= baseline, pick best) | 0.716 | 0.000 | +0.048 |
| null (no candidate better) | n=100 | pick_best + paired CI (biased) | 0.112 | 0.000 | +0.073 |
| null (no candidate better) | n=100 | bonferroni + paired CI | 0.052 | 0.000 | +0.084 |
| null (no candidate better) | n=100 | split_confirm + paired CI | 0.060 | 0.000 | +0.067 |
| small gain (+0.05 for one candidate) | n=30 | legacy point rule (acc>=0.70, >= baseline, pick best) | 0.856 | 0.464 | +0.070 |
| small gain (+0.05 for one candidate) | n=30 | pick_best + paired CI (biased) | 0.128 | 0.084 | +0.132 |
| small gain (+0.05 for one candidate) | n=30 | bonferroni + paired CI | 0.064 | 0.032 | +0.146 |
| small gain (+0.05 for one candidate) | n=30 | split_confirm + paired CI | 0.092 | 0.068 | +0.063 |
| small gain (+0.05 for one candidate) | n=60 | legacy point rule (acc>=0.70, >= baseline, pick best) | 0.872 | 0.524 | +0.053 |
| small gain (+0.05 for one candidate) | n=60 | pick_best + paired CI (biased) | 0.216 | 0.156 | +0.075 |
| small gain (+0.05 for one candidate) | n=60 | bonferroni + paired CI | 0.080 | 0.060 | +0.081 |
| small gain (+0.05 for one candidate) | n=60 | split_confirm + paired CI | 0.064 | 0.048 | +0.050 |
| small gain (+0.05 for one candidate) | n=100 | legacy point rule (acc>=0.70, >= baseline, pick best) | 0.832 | 0.600 | +0.031 |
| small gain (+0.05 for one candidate) | n=100 | pick_best + paired CI (biased) | 0.256 | 0.184 | +0.060 |
| small gain (+0.05 for one candidate) | n=100 | bonferroni + paired CI | 0.136 | 0.104 | +0.065 |
| small gain (+0.05 for one candidate) | n=100 | split_confirm + paired CI | 0.140 | 0.116 | +0.036 |
| large gain (+0.20 for one candidate) | n=30 | legacy point rule (acc>=0.70, >= baseline, pick best) | 0.992 | 0.984 | +0.005 |
| large gain (+0.20 for one candidate) | n=30 | pick_best + paired CI (biased) | 0.572 | 0.568 | +0.029 |
| large gain (+0.20 for one candidate) | n=30 | bonferroni + paired CI | 0.272 | 0.268 | +0.041 |
| large gain (+0.20 for one candidate) | n=30 | split_confirm + paired CI | 0.388 | 0.388 | +0.007 |
| large gain (+0.20 for one candidate) | n=60 | legacy point rule (acc>=0.70, >= baseline, pick best) | 1.000 | 1.000 | +0.001 |
| large gain (+0.20 for one candidate) | n=60 | pick_best + paired CI (biased) | 0.852 | 0.852 | +0.007 |
| large gain (+0.20 for one candidate) | n=60 | bonferroni + paired CI | 0.708 | 0.708 | +0.011 |
| large gain (+0.20 for one candidate) | n=60 | split_confirm + paired CI | 0.540 | 0.540 | +0.007 |
| large gain (+0.20 for one candidate) | n=100 | legacy point rule (acc>=0.70, >= baseline, pick best) | 1.000 | 1.000 | -0.002 |
| large gain (+0.20 for one candidate) | n=100 | pick_best + paired CI (biased) | 0.972 | 0.972 | -0.000 |
| large gain (+0.20 for one candidate) | n=100 | bonferroni + paired CI | 0.916 | 0.916 | +0.002 |
| large gain (+0.20 for one candidate) | n=100 | split_confirm + paired CI | 0.824 | 0.824 | -0.003 |

### Paired validation inside the engine (5 seeds, mean±std)


**abrupt_recurring**

| variant | accuracy | post-drift acc | models trained | recalls | precision | retrieval recall | false-match | recall success |
|---|---|---|---|---|---|---|---|---|
| legacy point (acc>=0.70, no uncertainty) | 91.4±0.2% | 81.9±0.5% | 4.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired non-inferiority + bonferroni + significant-majority floor | 91.4±0.2% | 81.9±0.5% | 4.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired non-inferiority + bonferroni + point floor (default) | 91.4±0.2% | 81.9±0.5% | 4.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired non-inferiority + bonferroni, no floor | 91.4±0.2% | 81.9±0.5% | 4.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired superiority(0.02) + bonferroni + point floor | 91.4±0.2% | 81.9±0.5% | 4.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired non-inferiority + pick_best (biased) + point floor | 91.4±0.2% | 81.9±0.5% | 4.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired non-inferiority + split_confirm + point floor | 90.8±0.2% | 78.6±0.4% | 4.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |

**noisy_recurring**

| variant | accuracy | post-drift acc | models trained | recalls | precision | retrieval recall | false-match | recall success |
|---|---|---|---|---|---|---|---|---|
| legacy point (acc>=0.70, no uncertainty) | 74.3±0.2% | 67.4±0.5% | 6.60±1.62 | 6.40±0.80 | 100.0±0.0% | 73.1±14.9% | 0.0±0.0% | 82.4±10.6% |
| paired non-inferiority + bonferroni + significant-majority floor | 74.2±0.3% | 66.7±1.2% | 12.40±2.24 | 3.60±1.62 | 100.0±0.0% | 30.6±14.5% | 0.0±0.0% | 86.7±19.4% |
| paired non-inferiority + bonferroni + point floor (default) | 74.3±0.2% | 67.1±0.7% | 6.20±1.47 | 5.20±1.33 | 100.0±0.0% | 70.7±17.9% | 0.0±0.0% | 83.8±14.9% |
| paired non-inferiority + bonferroni, no floor | 74.3±0.2% | 67.1±0.7% | 6.00±1.67 | 5.20±1.33 | 100.0±0.0% | 73.2±20.8% | 0.0±0.0% | 83.8±14.9% |
| paired superiority(0.02) + bonferroni + point floor | 74.2±0.2% | 67.1±0.7% | 6.40±1.36 | 4.80±1.60 | 100.0±0.0% | 66.1±19.8% | 0.0±0.0% | 89.1±15.6% |
| paired non-inferiority + pick_best (biased) + point floor | 74.3±0.2% | 67.1±0.7% | 6.20±1.47 | 5.20±1.33 | 100.0±0.0% | 70.7±17.9% | 0.0±0.0% | 83.8±14.9% |
| paired non-inferiority + split_confirm + point floor | 74.1±0.3% | 65.9±1.2% | 7.80±2.79 | 5.00±1.55 | 100.0±0.0% | 60.3±23.0% | 0.0±0.0% | 66.0±14.2% |

**similar_x_different_y**

| variant | accuracy | post-drift acc | models trained | recalls | precision | retrieval recall | false-match | recall success |
|---|---|---|---|---|---|---|---|---|
| legacy point (acc>=0.70, no uncertainty) | 90.9±0.2% | 79.1±0.3% | 3.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired non-inferiority + bonferroni + significant-majority floor | 90.9±0.2% | 79.1±0.3% | 3.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired non-inferiority + bonferroni + point floor (default) | 90.9±0.2% | 79.1±0.3% | 3.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired non-inferiority + bonferroni, no floor | 90.9±0.2% | 79.1±0.3% | 3.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired superiority(0.02) + bonferroni + point floor | 90.9±0.2% | 79.1±0.3% | 3.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired non-inferiority + pick_best (biased) + point floor | 90.9±0.2% | 79.1±0.3% | 3.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |
| paired non-inferiority + split_confirm + point floor | 89.9±0.1% | 73.6±0.4% | 3.00±0.00 | 7.00±0.00 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 100.0±0.0% |

**sea**

| variant | accuracy | post-drift acc | models trained | recalls | precision | retrieval recall | false-match | recall success |
|---|---|---|---|---|---|---|---|---|
| legacy point (acc>=0.70, no uncertainty) | 85.8±0.3% | 82.8±1.4% | 1.60±0.49 | 0.00±0.00 | n/a | n/a | 0.0±0.0% | n/a |
| paired non-inferiority + bonferroni + significant-majority floor | 85.8±0.3% | 83.1±1.2% | 3.60±1.02 | 0.40±0.80 | 0.0±0.0% | 0.0±0.0% | 6.7±13.3% | 0.0±0.0% |
| paired non-inferiority + bonferroni + point floor (default) | 85.8±0.3% | 82.7±1.4% | 1.80±0.75 | 0.00±0.00 | n/a | 0.0±0.0% | 0.0±0.0% | n/a |
| paired non-inferiority + bonferroni, no floor | 85.8±0.3% | 82.8±1.4% | 1.60±0.49 | 0.00±0.00 | n/a | n/a | 0.0±0.0% | n/a |
| paired superiority(0.02) + bonferroni + point floor | 85.8±0.3% | 82.7±1.4% | 1.80±0.75 | 0.00±0.00 | n/a | 0.0±0.0% | 0.0±0.0% | n/a |
| paired non-inferiority + pick_best (biased) + point floor | 85.8±0.3% | 82.7±1.4% | 1.80±0.75 | 0.00±0.00 | n/a | 0.0±0.0% | 0.0±0.0% | n/a |
| paired non-inferiority + split_confirm + point floor | 85.7±0.3% | 82.3±1.6% | 1.40±0.49 | 0.00±0.00 | n/a | n/a | 0.0±0.0% | n/a |

**gradual**

| variant | accuracy | post-drift acc | models trained | recalls | precision | retrieval recall | false-match | recall success |
|---|---|---|---|---|---|---|---|---|
| legacy point (acc>=0.70, no uncertainty) | 88.4±0.6% | 72.6±1.2% | 13.00±0.63 | 6.20±0.75 | 100.0±0.0% | 40.7±2.8% | 0.0±0.0% | 88.1±10.9% |
| paired non-inferiority + bonferroni + significant-majority floor | 88.2±0.6% | 72.7±1.4% | 15.40±1.20 | 3.60±1.02 | 100.0±0.0% | 24.0±7.1% | 0.0±0.0% | 91.0±11.1% |
| paired non-inferiority + bonferroni + point floor (default) | 88.0±0.9% | 72.6±1.2% | 14.40±1.85 | 4.60±1.50 | 96.0±8.0% | 30.0±10.7% | 1.1±2.2% | 88.7±15.7% |
| paired non-inferiority + bonferroni, no floor | 87.8±0.8% | 72.5±1.2% | 15.00±2.10 | 4.00±1.79 | 100.0±0.0% | 26.9±12.6% | 0.0±0.0% | 96.7±6.7% |
| paired superiority(0.02) + bonferroni + point floor | 87.9±0.9% | 72.6±1.2% | 15.60±1.85 | 3.40±1.62 | 100.0±0.0% | 22.8±11.1% | 0.0±0.0% | 95.0±10.0% |
| paired non-inferiority + pick_best (biased) + point floor | 88.0±1.0% | 72.6±1.2% | 14.40±1.02 | 4.80±0.75 | 100.0±0.0% | 31.7±5.4% | 0.0±0.0% | 92.0±9.8% |
| paired non-inferiority + split_confirm + point floor | 88.0±0.6% | 72.5±1.0% | 16.20±1.60 | 3.60±1.02 | 96.0±8.0% | 22.1±5.8% | 1.0±2.0% | 82.0±15.7% |


## A7. Mergeable memory and eviction (E10)

### Merge accuracy: mean |error| / pooled std (10 windows of 100; lower is better)

| location drift between windows | method | std | q10 | q90 | skew |
|---|---|---|---|---|---|
| location spread 0.0 | legacy averaging | 0.0063 | 0.0285 | 0.0244 | 0.0131 |
| location spread 0.0 | mergeable | 0.0000 | 0.0118 | 0.0124 | 0.0000 |
| location spread 0.5 | legacy averaging | 0.1009 | 0.1379 | 0.1404 | 0.0556 |
| location spread 0.5 | mergeable | 0.0000 | 0.0143 | 0.0119 | 0.0000 |
| location spread 1.0 | legacy averaging | 0.2580 | 0.3532 | 0.3464 | 0.1548 |
| location spread 1.0 | mergeable | 0.0000 | 0.0125 | 0.0118 | 0.0000 |
| location spread 2.0 | legacy averaging | 0.5068 | 0.6786 | 0.7165 | 0.2678 |
| location spread 2.0 | mergeable | 0.0000 | 0.0117 | 0.0095 | 0.0000 |

### Update cost and memory

| case | µs per update | bytes per concept |
|---|---|---|
| d=4 legacy | 586 | 1456 |
| d=4 mergeable | 4034 | 9744 |
| d=10 legacy | 818 | 3784 |
| d=10 mergeable | 9106 | 29416 |

### Eviction under Zipf recurrence (25 concepts, repository cap 8, 300 events, 3 seeds)

| policy | hit rate when concept is stored | needed after eviction | concepts created |
|---|---|---|---|
| legacy (uses, last_used) | 1.000±0.000 | 64.0±4.2 | 88.7±4.0 |
| lru | 1.000±0.000 | 83.7±6.8 | 108.3±6.6 |
| score (recurrence + recency) | 1.000±0.000 | 64.0±3.7 | 88.7±3.4 |
| score + grace 2 | 1.000±0.000 | 112.7±0.9 | 137.3±1.2 |


## A8. End-to-end ablation, seeds 1-5 (E11)

### Core variants on every scenario

**ab_gradual**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 86.3±0.7 | 73.4±2.1 | 2.8±0.4 | 0.2±0.4 | 100.0±0.0 | 20.0±40.0 | 0.0±0.0 | 80.0±40.0 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 87.6±0.8 | 74.0±1.5 | 4.0±0.0 | 0.8±0.4 | 100.0±0.0 | 26.7±13.3 | 0.0±0.0 | 73.3±13.3 | 65.0±8.2 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 86.0±0.5 | 73.3±2.1 | 3.2±0.4 | 0.0±0.0 | n/a | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | 53.3±45.2 | 100.0±0.0 | 100.0±0.0 | n/a |

**aba**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 90.9±0.3 | 80.9±0.9 | 2.0±0.0 | 1.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 90.8±0.4 | 80.0±0.8 | 2.0±0.0 | 1.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 90.8±0.4 | 80.0±0.8 | 2.0±0.0 | 1.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |

**aba_noisy**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 81.7±0.5 | 71.6±2.4 | 2.2±0.4 | 0.8±0.4 | 100.0±0.0 | 80.0±40.0 | 0.0±0.0 | 20.0±40.0 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 81.9±0.6 | 72.3±1.7 | 2.2±0.4 | 0.8±0.4 | 100.0±0.0 | 80.0±40.0 | 0.0±0.0 | 20.0±40.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 81.8±0.6 | 72.6±1.8 | 2.6±0.5 | 0.8±0.4 | 100.0±0.0 | 60.0±37.4 | 0.0±0.0 | 40.0±37.4 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |

**abca**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 90.7±0.4 | 79.2±0.9 | 3.0±0.0 | 1.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 90.7±0.4 | 79.1±0.7 | 3.0±0.0 | 1.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 90.6±0.5 | 78.6±0.9 | 3.2±0.4 | 0.8±0.4 | 100.0±0.0 | 80.0±40.0 | 0.0±0.0 | 20.0±40.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |

**abrupt_recurring**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 91.5±0.2 | 82.4±0.6 | 4.0±0.0 | 7.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 91.4±0.2 | 81.9±0.5 | 4.0±0.0 | 7.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 91.3±0.2 | 81.6±0.6 | 4.4±0.5 | 6.6±0.5 | 100.0±0.0 | 94.3±7.0 | 0.0±0.0 | 5.7±7.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |

**diff_x_same_y**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 92.7±0.3 | 92.6±0.5 | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 92.9±0.1 | 92.2±0.7 | 2.8±0.7 | 2.2±0.7 | 100.0±0.0 | 73.3±24.9 | 0.0±0.0 | 26.7±24.9 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 92.7±0.3 | 92.4±0.7 | 1.2±0.4 | 0.2±0.4 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |

**feature_availability**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 89.5±0.3 | 80.8±0.8 | 8.4±0.8 | 0.0±0.0 | n/a | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 89.5±0.2 | 80.4±0.4 | 3.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 89.5±0.2 | 80.4±0.4 | 3.2±0.4 | 4.0±0.0 | 100.0±0.0 | 96.0±8.0 | 0.0±0.0 | 4.0±8.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |

**gradual**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 85.9±0.5 | 72.2±1.2 | 9.2±1.5 | 2.4±1.0 | 100.0±0.0 | 32.5±15.7 | 0.0±0.0 | 67.5±15.7 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 88.4±0.6 | 72.6±1.2 | 13.0±0.6 | 6.2±0.7 | 100.0±0.0 | 40.7±2.8 | 0.0±0.0 | 59.3±2.8 | 72.0±6.0 | 88.9±5.8 | 97.5±3.1 | n/a |
| D full multi-view (improved) | 86.5±0.7 | 71.9±1.0 | 11.8±1.0 | 3.4±1.4 | 83.3±14.9 | 26.1±8.9 | 4.1±3.4 | 73.9±8.9 | 60.9±8.3 | 87.8±6.5 | 94.5±4.5 | n/a |

**hyperplane**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 89.6±0.2 | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 89.7±0.2 | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 89.7±0.2 | n/a | 1.2±0.4 | 0.0±0.0 | n/a | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |

**imbalanced**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 90.9±0.4 | 79.9±1.3 | 3.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 90.6±0.4 | 78.6±1.0 | 3.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 90.5±0.3 | 77.9±1.1 | 3.6±0.8 | 3.6±0.5 | 100.0±0.0 | 87.0±16.6 | 0.0±0.0 | 13.0±16.6 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |

**incremental**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 85.6±0.6 | 74.7±1.3 | 10.2±2.4 | 3.8±1.0 | 76.0±24.8 | 33.5±12.1 | 7.3±6.7 | 66.5±12.1 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 87.8±0.4 | 75.4±1.0 | 12.4±1.2 | 11.8±2.1 | 61.3±5.3 | 46.1±8.1 | 19.6±4.2 | 53.9±8.1 | 60.1±6.9 | 83.8±2.3 | 97.3±3.6 | n/a |
| D full multi-view (improved) | 87.3±0.3 | 75.5±0.6 | 12.6±1.9 | 5.2±2.5 | 86.7±12.5 | 33.1±12.4 | 5.7±6.1 | 66.9±12.4 | 58.0±5.1 | 78.3±10.5 | 84.3±4.7 | n/a |

**noisy_recurring**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 73.9±0.4 | 65.0±1.2 | 10.4±3.4 | 4.6±1.4 | 96.0±8.0 | 44.7±18.3 | 1.7±3.3 | 55.3±18.3 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 74.3±0.2 | 67.4±0.5 | 6.6±1.6 | 6.4±0.8 | 100.0±0.0 | 73.1±14.9 | 0.0±0.0 | 26.9±14.9 | 99.1±1.7 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 74.1±0.3 | 66.4±0.9 | 10.8±2.6 | 3.6±1.2 | 100.0±0.0 | 36.5±15.9 | 0.0±0.0 | 63.5±15.9 | 99.0±2.0 | 100.0±0.0 | 100.0±0.0 | n/a |

**overlapping**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 91.2±0.1 | 81.4±1.2 | 4.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 91.1±0.1 | 81.1±0.8 | 4.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 72.0±28.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 91.0±0.2 | 80.8±0.4 | 4.2±0.4 | 3.8±0.4 | 100.0±0.0 | 95.0±10.0 | 0.0±0.0 | 5.0±10.0 | 82.0±15.7 | 100.0±0.0 | 100.0±0.0 | n/a |

**same_marginal_diff_corr**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 93.2±0.1 | 93.4±0.7 | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 93.2±0.2 | 93.4±0.7 | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 93.2±0.2 | 93.4±0.7 | 1.2±0.4 | 0.0±0.0 | n/a | n/a | 0.0±0.0 | n/a | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |

**sea**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 85.7±0.3 | 82.5±1.6 | 2.0±0.0 | 0.2±0.4 | 0.0±0.0 | n/a | 10.0±20.0 | n/a | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 85.8±0.3 | 82.8±1.4 | 1.6±0.5 | 0.0±0.0 | n/a | n/a | 0.0±0.0 | n/a | 80.0±40.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 85.8±0.3 | 82.9±1.3 | 2.2±1.5 | 0.4±0.8 | 50.0±0.0 | 50.0±50.0 | 11.1±15.7 | 50.0±50.0 | 90.0±20.0 | 100.0±0.0 | 100.0±0.0 | n/a |

**similar_x_different_y**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 91.1±0.2 | 80.2±0.7 | 3.0±0.0 | 7.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 90.9±0.2 | 79.1±0.3 | 3.0±0.0 | 7.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 68.9±18.2 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 90.8±0.3 | 78.7±1.0 | 3.4±0.8 | 6.6±0.8 | 100.0±0.0 | 94.3±11.4 | 0.0±0.0 | 5.7±11.4 | 63.2±16.5 | 100.0±0.0 | 100.0±0.0 | n/a |

**temporal_only**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 93.2±0.2 | 93.4±0.5 | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 93.1±0.4 | 93.3±0.6 | 1.6±0.5 | 0.8±0.7 | 16.7±23.6 | 100.0±0.0 | 44.4±7.9 | 0.0±0.0 | 61.1±43.7 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 92.9±0.4 | 92.5±1.1 | 3.0±1.1 | 4.0±1.5 | 90.0±20.0 | 75.0±27.9 | 6.7±13.3 | 25.0±27.9 | 97.1±5.7 | 100.0±0.0 | 100.0±0.0 | n/a |

**varied_sizes**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % | cand. R@3 % | cand. R@5 % | high-conf. precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 91.1±0.2 | 82.2±0.9 | 3.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 90.9±0.3 | 81.4±0.6 | 3.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |
| D full multi-view (improved) | 90.9±0.3 | 81.4±0.6 | 3.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | n/a |

### Ablation

#### Macro-average over the eight ablation scenarios (mean of per-scenario means)

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % |
|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 88.4 | 79.4 | 5.4 | 3.6 | 99.3 | 68.2 | 0.2 | 31.8 | n/a |
| MV schema 2.0 (baseline) | 88.7 | 79.3 | 4.7 | 4.8 | 100.0 | 87.7 | 0.0 | 12.3 | 89.0 |
| A statistical only | 88.4 | 78.9 | 5.0 | 4.2 | 97.4 | 78.0 | 0.8 | 22.0 | 88.0 |
| B stat + distribution | 88.6 | 79.1 | 5.2 | 4.4 | 99.5 | 80.7 | 0.2 | 19.3 | 91.0 |
| C stat + dist + temporal | 88.4 | 79.1 | 4.9 | 4.2 | 97.7 | 81.0 | 0.5 | 19.0 | 88.3 |
| D full multi-view (improved) | 88.4 | 78.9 | 5.3 | 4.0 | 97.6 | 75.6 | 0.5 | 24.4 | 88.1 |
| E = D without noise correction | 88.5 | 78.5 | 7.5 | 11.3 | 92.7 | 75.6 | 6.2 | 24.4 | 88.2 |
| F = D without adaptive thresholds | 88.4 | 79.2 | 4.2 | 4.4 | 99.4 | 87.1 | 0.2 | 12.9 | 88.5 |
| G = D without confidence gating | 88.4 | 78.9 | 5.3 | 4.0 | 97.6 | 75.6 | 0.5 | 24.4 | 88.1 |
| H = D without predictive validation | 87.0 | 72.0 | 3.9 | 3.7 | 86.8 | 85.5 | 8.1 | 14.5 | 95.2 |
| J = D + paired validation | 88.3 | 78.9 | 5.5 | 3.6 | 99.1 | 72.0 | 0.2 | 28.0 | 87.6 |
| K = D with legacy fingerprint merge | 88.3 | 78.8 | 5.4 | 3.8 | 98.6 | 73.6 | 0.3 | 26.4 | 86.3 |

**abrupt_recurring**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % |
|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 91.5±0.2 | 82.4±0.6 | 4.0±0.0 | 7.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | n/a |
| MV schema 2.0 (baseline) | 91.4±0.2 | 81.9±0.5 | 4.0±0.0 | 7.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 |
| A statistical only | 91.3±0.2 | 81.5±0.6 | 4.6±0.5 | 6.4±0.5 | 100.0±0.0 | 91.4±7.0 | 0.0±0.0 | 8.6±7.0 | 100.0±0.0 |
| B stat + distribution | 91.3±0.3 | 81.5±0.8 | 4.6±0.8 | 6.4±0.8 | 100.0±0.0 | 91.4±11.4 | 0.0±0.0 | 8.6±11.4 | 100.0±0.0 |
| C stat + dist + temporal | 91.4±0.2 | 81.7±0.6 | 4.2±0.4 | 6.8±0.4 | 100.0±0.0 | 97.1±5.7 | 0.0±0.0 | 2.9±5.7 | 100.0±0.0 |
| D full multi-view (improved) | 91.3±0.2 | 81.6±0.6 | 4.4±0.5 | 6.6±0.5 | 100.0±0.0 | 94.3±7.0 | 0.0±0.0 | 5.7±7.0 | 100.0±0.0 |
| E = D without noise correction | 90.9±0.3 | 80.0±1.4 | 5.8±1.3 | 11.4±4.4 | 97.4±3.3 | 87.4±8.0 | 1.9±2.3 | 12.6±8.0 | 98.9±0.9 |
| F = D without adaptive thresholds | 91.4±0.2 | 81.9±0.5 | 4.0±0.0 | 7.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 |
| G = D without confidence gating | 91.3±0.2 | 81.6±0.6 | 4.4±0.5 | 6.6±0.5 | 100.0±0.0 | 94.3±7.0 | 0.0±0.0 | 5.7±7.0 | 100.0±0.0 |
| H = D without predictive validation | 91.4±0.2 | 81.7±0.6 | 4.2±0.4 | 6.8±0.4 | 100.0±0.0 | 97.1±5.7 | 0.0±0.0 | 2.9±5.7 | 100.0±0.0 |
| J = D + paired validation | 91.3±0.2 | 81.6±0.6 | 4.4±0.5 | 6.6±0.5 | 100.0±0.0 | 94.3±7.0 | 0.0±0.0 | 5.7±7.0 | 100.0±0.0 |
| K = D with legacy fingerprint merge | 91.3±0.2 | 81.6±0.6 | 4.4±0.5 | 6.6±0.5 | 100.0±0.0 | 94.3±7.0 | 0.0±0.0 | 5.7±7.0 | 100.0±0.0 |

**noisy_recurring**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % |
|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 73.9±0.4 | 65.0±1.2 | 10.4±3.4 | 4.6±1.4 | 96.0±8.0 | 44.7±18.3 | 1.7±3.3 | 55.3±18.3 | n/a |
| MV schema 2.0 (baseline) | 74.3±0.2 | 67.4±0.5 | 6.6±1.6 | 6.4±0.8 | 100.0±0.0 | 73.1±14.9 | 0.0±0.0 | 26.9±14.9 | 99.1±1.7 |
| A statistical only | 74.2±0.3 | 66.5±0.7 | 8.4±2.1 | 4.8±1.2 | 100.0±0.0 | 53.8±15.5 | 0.0±0.0 | 46.2±15.5 | 99.0±1.9 |
| B stat + distribution | 74.2±0.3 | 66.6±1.0 | 8.4±2.9 | 4.4±1.6 | 100.0±0.0 | 53.4±24.6 | 0.0±0.0 | 46.6±24.6 | 99.1±1.8 |
| C stat + dist + temporal | 74.2±0.3 | 67.0±0.8 | 7.8±2.8 | 5.0±0.9 | 100.0±0.0 | 60.9±18.9 | 0.0±0.0 | 39.1±18.9 | 99.0±2.0 |
| D full multi-view (improved) | 74.1±0.3 | 66.4±0.9 | 10.8±2.6 | 3.6±1.2 | 100.0±0.0 | 36.5±15.9 | 0.0±0.0 | 63.5±15.9 | 99.0±2.0 |
| E = D without noise correction | 74.1±0.5 | 66.7±1.5 | 14.2±1.9 | 23.4±5.4 | 100.0±0.0 | 68.9±7.2 | 0.0±0.0 | 31.1±7.2 | 99.2±0.6 |
| F = D without adaptive thresholds | 74.3±0.2 | 67.4±0.5 | 6.4±1.7 | 6.0±0.6 | 100.0±0.0 | 73.9±16.6 | 0.0±0.0 | 26.1±16.6 | 99.0±1.9 |
| G = D without confidence gating | 74.1±0.3 | 66.4±0.9 | 10.8±2.6 | 3.6±1.2 | 100.0±0.0 | 36.5±15.9 | 0.0±0.0 | 63.5±15.9 | 99.0±2.0 |
| H = D without predictive validation | 74.2±0.3 | 66.9±0.9 | 6.6±1.4 | 4.6±1.0 | 100.0±0.0 | 64.6±16.3 | 0.0±0.0 | 35.4±16.3 | 98.9±2.2 |
| J = D + paired validation | 74.1±0.3 | 66.4±0.9 | 10.8±2.5 | 3.0±1.3 | 100.0±0.0 | 31.9±14.5 | 0.0±0.0 | 68.1±14.5 | 99.0±2.0 |
| K = D with legacy fingerprint merge | 74.1±0.3 | 66.3±0.9 | 10.8±2.2 | 3.8±1.2 | 100.0±0.0 | 37.0±14.2 | 0.0±0.0 | 63.0±14.2 | 99.0±2.0 |

**gradual**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % |
|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 85.9±0.5 | 72.2±1.2 | 9.2±1.5 | 2.4±1.0 | 100.0±0.0 | 32.5±15.7 | 0.0±0.0 | 67.5±15.7 | n/a |
| MV schema 2.0 (baseline) | 88.4±0.6 | 72.6±1.2 | 13.0±0.6 | 6.2±0.7 | 100.0±0.0 | 40.7±2.8 | 0.0±0.0 | 59.3±2.8 | 72.0±6.0 |
| A statistical only | 86.9±0.5 | 72.0±1.2 | 12.0±0.9 | 4.2±1.0 | 81.7±9.7 | 29.8±6.2 | 5.3±2.7 | 70.2±6.2 | 67.4±7.8 |
| B stat + distribution | 88.3±0.7 | 72.7±1.0 | 13.8±0.4 | 5.8±0.4 | 96.7±6.7 | 36.3±2.6 | 1.1±2.1 | 63.7±2.6 | 69.6±2.8 |
| C stat + dist + temporal | 86.7±0.9 | 72.1±1.2 | 11.8±1.5 | 3.8±1.6 | 83.8±14.9 | 28.7±11.3 | 4.1±3.4 | 71.3±11.3 | 62.3±7.7 |
| D full multi-view (improved) | 86.5±0.7 | 71.9±1.0 | 11.8±1.0 | 3.4±1.4 | 83.3±14.9 | 26.1±8.9 | 4.1±3.4 | 73.9±8.9 | 60.9±8.3 |
| E = D without noise correction | 88.8±0.6 | 73.3±0.7 | 13.2±2.2 | 9.8±3.7 | 98.3±3.3 | 50.2±7.3 | 0.9±1.8 | 49.8±7.3 | 86.6±1.2 |
| F = D without adaptive thresholds | 86.4±0.7 | 71.9±1.0 | 9.6±0.5 | 3.4±1.2 | 96.0±8.0 | 35.8±8.9 | 1.5±3.1 | 64.2±8.9 | 64.1±10.4 |
| G = D without confidence gating | 86.5±0.7 | 71.9±1.0 | 11.8±1.0 | 3.4±1.4 | 83.3±14.9 | 26.1±8.9 | 4.1±3.4 | 73.9±8.9 | 60.9±8.3 |
| H = D without predictive validation | 86.5±0.5 | 70.4±0.7 | 9.4±1.0 | 7.0±1.4 | 67.3±21.8 | 45.3±11.4 | 15.0±10.2 | 54.7±11.4 | 62.8±10.7 |
| J = D + paired validation | 86.3±0.6 | 72.0±1.1 | 12.8±1.3 | 1.8±1.5 | 93.8±10.8 | 15.2±10.9 | 1.4±2.9 | 84.8±10.9 | 56.7±8.2 |
| K = D with legacy fingerprint merge | 86.3±0.7 | 72.0±1.2 | 11.6±0.8 | 3.0±1.7 | 90.0±13.3 | 24.8±10.6 | 2.8±3.4 | 75.2±10.6 | 62.3±7.6 |

**similar_x_different_y**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % |
|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 91.1±0.2 | 80.2±0.7 | 3.0±0.0 | 7.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | n/a |
| MV schema 2.0 (baseline) | 90.9±0.2 | 79.1±0.3 | 3.0±0.0 | 7.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 68.9±18.2 |
| A statistical only | 90.8±0.3 | 78.8±0.5 | 3.4±0.5 | 6.6±0.5 | 100.0±0.0 | 94.3±7.0 | 0.0±0.0 | 5.7±7.0 | 61.8±11.4 |
| B stat + distribution | 90.7±0.4 | 78.5±1.5 | 3.6±1.2 | 6.4±1.2 | 100.0±0.0 | 91.4±17.1 | 0.0±0.0 | 8.6±17.1 | 67.5±14.4 |
| C stat + dist + temporal | 90.8±0.3 | 78.7±1.0 | 3.4±0.8 | 6.6±0.8 | 100.0±0.0 | 94.3±11.4 | 0.0±0.0 | 5.7±11.4 | 63.2±16.5 |
| D full multi-view (improved) | 90.8±0.3 | 78.7±1.0 | 3.4±0.8 | 6.6±0.8 | 100.0±0.0 | 94.3±11.4 | 0.0±0.0 | 5.7±11.4 | 63.2±16.5 |
| E = D without noise correction | 90.4±0.1 | 77.4±0.9 | 5.6±2.2 | 11.8±5.3 | 100.0±0.0 | 84.0±5.2 | 0.0±0.0 | 16.0±5.2 | 64.9±4.0 |
| F = D without adaptive thresholds | 90.9±0.2 | 79.1±0.3 | 3.0±0.0 | 7.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 63.2±16.5 |
| G = D without confidence gating | 90.8±0.3 | 78.7±1.0 | 3.4±0.8 | 6.6±0.8 | 100.0±0.0 | 94.3±11.4 | 0.0±0.0 | 5.7±11.4 | 63.2±16.5 |
| H = D without predictive validation | 84.1±0.1 | 45.4±0.7 | 2.0±0.0 | 2.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 |
| J = D + paired validation | 90.8±0.3 | 78.7±1.0 | 3.4±0.8 | 6.6±0.8 | 100.0±0.0 | 94.3±11.4 | 0.0±0.0 | 5.7±11.4 | 63.2±16.5 |
| K = D with legacy fingerprint merge | 90.7±0.4 | 78.5±1.5 | 3.6±1.2 | 6.4±1.2 | 100.0±0.0 | 91.4±17.1 | 0.0±0.0 | 8.6±17.1 | 62.5±8.3 |

**overlapping**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % |
|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 91.2±0.1 | 81.4±1.2 | 4.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | n/a |
| MV schema 2.0 (baseline) | 91.1±0.1 | 81.1±0.8 | 4.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 72.0±28.0 |
| A statistical only | 91.0±0.2 | 80.8±1.3 | 4.2±0.4 | 3.8±0.4 | 100.0±0.0 | 95.0±10.0 | 0.0±0.0 | 5.0±10.0 | 76.0±15.9 |
| B stat + distribution | 91.1±0.1 | 81.1±0.8 | 4.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 92.0±16.0 |
| C stat + dist + temporal | 91.0±0.2 | 80.8±0.4 | 4.2±0.4 | 3.8±0.4 | 100.0±0.0 | 95.0±10.0 | 0.0±0.0 | 5.0±10.0 | 82.0±15.7 |
| D full multi-view (improved) | 91.0±0.2 | 80.8±0.4 | 4.2±0.4 | 3.8±0.4 | 100.0±0.0 | 95.0±10.0 | 0.0±0.0 | 5.0±10.0 | 82.0±15.7 |
| E = D without noise correction | 90.6±0.2 | 79.6±0.6 | 10.0±5.2 | 10.0±6.4 | 95.0±7.7 | 64.0±12.1 | 2.8±4.3 | 36.0±12.1 | 66.1±11.0 |
| F = D without adaptive thresholds | 91.1±0.1 | 81.1±0.8 | 4.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 82.0±15.7 |
| G = D without confidence gating | 91.0±0.2 | 80.8±0.4 | 4.2±0.4 | 3.8±0.4 | 100.0±0.0 | 95.0±10.0 | 0.0±0.0 | 5.0±10.0 | 82.0±15.7 |
| H = D without predictive validation | 86.4±0.3 | 59.8±1.7 | 1.2±0.4 | 1.0±2.0 | 40.0±0.0 | 100.0±0.0 | 50.0±0.0 | 0.0±0.0 | 100.0±0.0 |
| J = D + paired validation | 91.0±0.2 | 80.8±0.4 | 4.2±0.4 | 3.8±0.4 | 100.0±0.0 | 95.0±10.0 | 0.0±0.0 | 5.0±10.0 | 82.0±15.7 |
| K = D with legacy fingerprint merge | 91.0±0.2 | 80.8±0.4 | 4.2±0.4 | 3.8±0.4 | 100.0±0.0 | 95.0±10.0 | 0.0±0.0 | 5.0±10.0 | 67.0±36.8 |

**imbalanced**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % |
|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 90.9±0.4 | 79.9±1.3 | 3.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | n/a |
| MV schema 2.0 (baseline) | 90.6±0.4 | 78.6±1.0 | 3.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 |
| A statistical only | 90.6±0.3 | 78.3±0.7 | 3.2±0.4 | 3.8±0.4 | 100.0±0.0 | 95.0±10.0 | 0.0±0.0 | 5.0±10.0 | 100.0±0.0 |
| B stat + distribution | 90.7±0.4 | 78.6±1.0 | 3.2±0.4 | 4.0±0.0 | 100.0±0.0 | 96.0±8.0 | 0.0±0.0 | 4.0±8.0 | 100.0±0.0 |
| C stat + dist + temporal | 90.6±0.3 | 78.3±0.7 | 3.2±0.4 | 3.8±0.4 | 100.0±0.0 | 95.0±10.0 | 0.0±0.0 | 5.0±10.0 | 100.0±0.0 |
| D full multi-view (improved) | 90.5±0.3 | 77.9±1.1 | 3.6±0.8 | 3.6±0.5 | 100.0±0.0 | 87.0±16.6 | 0.0±0.0 | 13.0±16.6 | 100.0±0.0 |
| E = D without noise correction | 90.4±0.4 | 78.0±1.3 | 4.8±1.6 | 8.6±6.3 | 100.0±0.0 | 82.3±4.7 | 0.0±0.0 | 17.7±4.7 | 98.7±1.6 |
| F = D without adaptive thresholds | 90.6±0.4 | 78.6±1.0 | 3.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 |
| G = D without confidence gating | 90.5±0.3 | 77.9±1.1 | 3.6±0.8 | 3.6±0.5 | 100.0±0.0 | 87.0±16.6 | 0.0±0.0 | 13.0±16.6 | 100.0±0.0 |
| H = D without predictive validation | 90.6±0.3 | 78.3±0.7 | 3.2±0.4 | 3.8±0.4 | 100.0±0.0 | 95.0±10.0 | 0.0±0.0 | 5.0±10.0 | 100.0±0.0 |
| J = D + paired validation | 90.4±0.4 | 77.6±1.2 | 4.2±1.2 | 3.4±0.5 | 100.0±0.0 | 77.0±20.4 | 0.0±0.0 | 23.0±20.4 | 100.0±0.0 |
| K = D with legacy fingerprint merge | 90.4±0.4 | 77.2±1.1 | 4.0±0.9 | 3.2±0.7 | 100.0±0.0 | 77.0±20.4 | 0.0±0.0 | 23.0±20.4 | 100.0±0.0 |

**same_marginal_diff_corr**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % |
|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 93.2±0.1 | 93.4±0.7 | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a |
| MV schema 2.0 (baseline) | 93.2±0.2 | 93.4±0.7 | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | 100.0±0.0 |
| A statistical only | 93.2±0.2 | 93.4±0.7 | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | 100.0±0.0 |
| B stat + distribution | 93.2±0.2 | 93.4±0.7 | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | 100.0±0.0 |
| C stat + dist + temporal | 93.2±0.2 | 93.4±0.7 | 1.2±0.4 | 0.0±0.0 | n/a | n/a | 0.0±0.0 | n/a | 100.0±0.0 |
| D full multi-view (improved) | 93.2±0.2 | 93.4±0.7 | 1.2±0.4 | 0.0±0.0 | n/a | n/a | 0.0±0.0 | n/a | 100.0±0.0 |
| E = D without noise correction | 93.0±0.1 | 93.0±0.6 | 1.8±1.2 | 5.6±7.0 | 51.0±17.7 | 82.3±2.3 | 43.9±17.6 | 17.7±2.3 | 91.5±17.0 |
| F = D without adaptive thresholds | 93.2±0.2 | 93.4±0.7 | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | 100.0±0.0 |
| G = D without confidence gating | 93.2±0.2 | 93.4±0.7 | 1.2±0.4 | 0.0±0.0 | n/a | n/a | 0.0±0.0 | n/a | 100.0±0.0 |
| H = D without predictive validation | 93.2±0.2 | 93.4±0.7 | 1.2±0.4 | 0.0±0.0 | n/a | n/a | 0.0±0.0 | n/a | 100.0±0.0 |
| J = D + paired validation | 93.2±0.2 | 93.4±0.7 | 1.2±0.4 | 0.0±0.0 | n/a | n/a | 0.0±0.0 | n/a | 100.0±0.0 |
| K = D with legacy fingerprint merge | 93.2±0.2 | 93.4±0.7 | 1.2±0.4 | 0.0±0.0 | n/a | n/a | 0.0±0.0 | n/a | 100.0±0.0 |

**feature_availability**

| variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % | false-match % | missed-match % | cand. R@1 % |
|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 89.5±0.3 | 80.8±0.8 | 8.4±0.8 | 0.0±0.0 | n/a | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 | n/a |
| MV schema 2.0 (baseline) | 89.5±0.2 | 80.4±0.4 | 3.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 |
| A statistical only | 89.5±0.2 | 80.3±0.6 | 3.6±0.5 | 3.8±0.4 | 100.0±0.0 | 87.0±10.8 | 0.0±0.0 | 13.0±10.8 | 100.0±0.0 |
| B stat + distribution | 89.5±0.2 | 80.4±0.4 | 3.2±0.4 | 4.0±0.0 | 100.0±0.0 | 96.0±8.0 | 0.0±0.0 | 4.0±8.0 | 100.0±0.0 |
| C stat + dist + temporal | 89.5±0.2 | 80.4±0.4 | 3.2±0.4 | 4.0±0.0 | 100.0±0.0 | 96.0±8.0 | 0.0±0.0 | 4.0±8.0 | 100.0±0.0 |
| D full multi-view (improved) | 89.5±0.2 | 80.4±0.4 | 3.2±0.4 | 4.0±0.0 | 100.0±0.0 | 96.0±8.0 | 0.0±0.0 | 4.0±8.0 | 100.0±0.0 |
| E = D without noise correction | 89.3±0.1 | 80.3±0.7 | 5.0±1.4 | 9.6±4.2 | 100.0±0.0 | 85.8±7.6 | 0.0±0.0 | 14.2±7.6 | 100.0±0.0 |
| F = D without adaptive thresholds | 89.5±0.2 | 80.4±0.4 | 3.0±0.0 | 4.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 | 100.0±0.0 |
| G = D without confidence gating | 89.5±0.2 | 80.4±0.4 | 3.2±0.4 | 4.0±0.0 | 100.0±0.0 | 96.0±8.0 | 0.0±0.0 | 4.0±8.0 | 100.0±0.0 |
| H = D without predictive validation | 89.5±0.2 | 80.4±0.3 | 3.2±0.4 | 4.2±0.4 | 100.0±0.0 | 96.7±6.7 | 0.0±0.0 | 3.3±6.7 | 100.0±0.0 |
| J = D + paired validation | 89.5±0.2 | 80.4±0.4 | 3.2±0.4 | 4.0±0.0 | 100.0±0.0 | 96.0±8.0 | 0.0±0.0 | 4.0±8.0 | 100.0±0.0 |
| K = D with legacy fingerprint merge | 89.5±0.2 | 80.4±0.4 | 3.2±0.4 | 4.0±0.0 | 100.0±0.0 | 96.0±8.0 | 0.0±0.0 | 4.0±8.0 | 100.0±0.0 |

### Cost (abrupt_recurring)

| variant (abrupt_recurring) | µs / sample | fingerprint ms | match ms | bytes / concept |
|---|---|---|---|---|
| v1 (legacy engine) | 19.4±0.0 | n/a | n/a | 136.0±0.0 |
| MV schema 2.0 (baseline) | 48.9±0.1 | 1.34±0.00 | 1.65±0.03 | 1456.0±0.0 |
| A statistical only | 76.7±0.1 | 1.27±0.00 | 2.25±0.04 | 9640.8±147.2 |
| B stat + distribution | 77.2±0.8 | 1.28±0.01 | 2.28±0.08 | 9542.7±279.6 |
| C stat + dist + temporal | 77.2±0.2 | 1.28±0.01 | 2.26±0.01 | 9707.2±118.0 |
| D full multi-view (improved) | 77.2±0.3 | 1.28±0.00 | 2.26±0.03 | 9703.7±115.9 |
| E = D without noise correction | 81.1±1.7 | 1.29±0.00 | 2.32±0.09 | 8775.2±816.0 |
| F = D without adaptive thresholds | 59.2±0.1 | 1.35±0.00 | 2.06±0.02 | 9768.8±11.1 |
| G = D without confidence gating | 77.1±0.2 | 1.28±0.00 | 2.25±0.03 | 9703.7±115.9 |
| H = D without predictive validation | 78.5±1.6 | 1.31±0.02 | 2.31±0.07 | 9707.2±118.0 |
| J = D + paired validation | 79.1±1.2 | 1.29±0.01 | 2.29±0.07 | 9703.7±115.9 |
| K = D with legacy fingerprint merge | 71.8±0.2 | 1.27±0.01 | 2.24±0.04 | 1456.0±0.0 |

### Delayed labels (abrupt_recurring)

| labels delayed by | variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % |
|---|---|---|---|---|---|---|---|
| 50 | MV schema 2.0 (baseline) | 88.7±0.6 | 72.6±0.8 | 5.4±0.5 | 5.6±0.5 | 100.0±0.0 | 80.0±7.0 |
| 50 | D full multi-view (improved) | 88.5±0.8 | 72.2±1.2 | 5.8±0.4 | 5.2±0.4 | 100.0±0.0 | 74.3±5.7 |
| 200 | MV schema 2.0 (baseline) | 84.1±0.6 | 49.6±1.1 | 13.0±0.9 | 8.0±0.6 | 100.0±0.0 | 47.1±4.1 |
| 200 | D full multi-view (improved) | 83.8±0.7 | 48.6±1.0 | 15.4±1.0 | 5.8±1.0 | 100.0±0.0 | 33.6±4.7 |


## A9. Fresh-seed confirmation, seeds 11-20, exact paired permutation tests (E13)


### Scorecard: scenarios (of 18) where the difference is significant after Holm correction


**F = D without adaptive thresholds  vs  MV schema 2.0 (baseline)**

| metric | significantly better | significantly worse | no significant difference |
|---|---|---|---|
| accuracy | 0 | 2 | 16 |
| post_drift_acc | 0 | 0 | 17 |
| models_trained | 1 | 0 | 17 |
| retrieval_precision | 0 | 0 | 12 |
| retrieval_recall | 0 | 0 | 13 |
| false_match_rate | 0 | 0 | 14 |
| missed_match_rate | 0 | 0 | 13 |
| candidate_recall@1 | 0 | 0 | 16 |
| us_per_sample | 0 | 17 | 1 |

**D full multi-view (improved)  vs  MV schema 2.0 (baseline)**

| metric | significantly better | significantly worse | no significant difference |
|---|---|---|---|
| accuracy | 0 | 3 | 15 |
| post_drift_acc | 0 | 0 | 17 |
| models_trained | 0 | 0 | 18 |
| retrieval_precision | 0 | 0 | 12 |
| retrieval_recall | 0 | 2 | 11 |
| false_match_rate | 0 | 0 | 15 |
| missed_match_rate | 0 | 2 | 11 |
| candidate_recall@1 | 0 | 0 | 17 |
| us_per_sample | 0 | 17 | 1 |

**MV schema 2.0 (baseline)  vs  v1 (legacy engine)**

| metric | significantly better | significantly worse | no significant difference |
|---|---|---|---|
| accuracy | 5 | 0 | 13 |
| post_drift_acc | 1 | 0 | 16 |
| models_trained | 2 | 1 | 15 |
| retrieval_precision | 0 | 0 | 11 |
| retrieval_recall | 1 | 0 | 12 |
| false_match_rate | 0 | 1 | 13 |
| missed_match_rate | 1 | 0 | 12 |
| candidate_recall@1 | 0 | 0 | 0 |
| us_per_sample | 0 | 18 | 0 |


**F = D without adaptive thresholds  versus  MV schema 2.0 (baseline)** (per-scenario mean difference, exact paired sign-flip test over 10 seeds, Holm-adjusted p; * = adjusted p < 0.05)


*accuracy*

| scenario | MV | F | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 0.915 | 0.915 | +0.000 | 1.000 |
| noisy_recurring | 0.742 | 0.742 | +0.000 | 1.000 |
| gradual | 0.877 | 0.862 | -0.016* | 0.035 |
| incremental | 0.877 | 0.867 | -0.010* | 0.035 |
| similar_x_different_y | 0.910 | 0.910 | +0.000 | 1.000 |
| temporal_only | 0.934 | 0.934 | -0.000 | 1.000 |
| sea | 0.863 | 0.863 | +0.000 | 1.000 |
| hyperplane | 0.894 | 0.894 | +0.000 | 1.000 |
| aba | 0.907 | 0.907 | +0.000 | 1.000 |
| abca | 0.906 | 0.906 | +0.000 | 1.000 |
| aba_noisy | 0.817 | 0.816 | -0.002 | 1.000 |
| ab_gradual | 0.876 | 0.869 | -0.007 | 0.438 |
| diff_x_same_y | 0.930 | 0.928 | -0.002 | 0.469 |
| same_marginal_diff_corr | 0.934 | 0.934 | +0.000 | 1.000 |
| varied_sizes | 0.910 | 0.910 | +0.000 | 1.000 |
| overlapping | 0.911 | 0.911 | +0.000 | 1.000 |
| imbalanced | 0.905 | 0.906 | +0.001 | 1.000 |
| feature_availability | 0.897 | 0.897 | +0.000 | 1.000 |

*post_drift_acc*

| scenario | MV | F | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 0.822 | 0.822 | +0.000 | 1.000 |
| noisy_recurring | 0.671 | 0.672 | +0.001 | 1.000 |
| gradual | 0.722 | 0.717 | -0.005 | 0.938 |
| incremental | 0.760 | 0.759 | -0.001 | 1.000 |
| similar_x_different_y | 0.797 | 0.797 | +0.000 | 1.000 |
| temporal_only | 0.940 | 0.937 | -0.003 | 1.000 |
| sea | 0.831 | 0.831 | +0.000 | 1.000 |
| aba | 0.803 | 0.803 | +0.000 | 1.000 |
| abca | 0.788 | 0.788 | +0.000 | 1.000 |
| aba_noisy | 0.738 | 0.737 | -0.001 | 1.000 |
| ab_gradual | 0.742 | 0.748 | +0.005 | 1.000 |
| diff_x_same_y | 0.934 | 0.937 | +0.003 | 0.199 |
| same_marginal_diff_corr | 0.941 | 0.941 | +0.000 | 1.000 |
| varied_sizes | 0.809 | 0.809 | -0.000 | 1.000 |
| overlapping | 0.810 | 0.810 | +0.000 | 1.000 |
| imbalanced | 0.782 | 0.784 | +0.002 | 1.000 |
| feature_availability | 0.809 | 0.809 | +0.000 | 1.000 |

*models_trained*

| scenario | MV | F | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 4.000 | 4.000 | +0.000 | 1.000 |
| noisy_recurring | 8.600 | 8.200 | -0.400 | 1.000 |
| gradual | 12.900 | 9.300 | -3.600 | 0.066 |
| incremental | 11.500 | 9.900 | -1.600 | 0.469 |
| similar_x_different_y | 3.000 | 3.000 | +0.000 | 1.000 |
| temporal_only | 1.400 | 2.800 | +1.400 | 0.125 |
| sea | 1.600 | 1.600 | +0.000 | 1.000 |
| hyperplane | 1.000 | 1.000 | +0.000 | 1.000 |
| aba | 2.000 | 2.000 | +0.000 | 1.000 |
| abca | 3.000 | 3.000 | +0.000 | 1.000 |
| aba_noisy | 2.100 | 2.100 | +0.000 | 1.000 |
| ab_gradual | 3.900 | 2.900 | -1.000 | 0.469 |
| diff_x_same_y | 2.700 | 1.000 | -1.700* | 0.035 |
| same_marginal_diff_corr | 1.000 | 1.000 | +0.000 | 1.000 |
| varied_sizes | 3.000 | 3.000 | +0.000 | 1.000 |
| overlapping | 4.100 | 4.100 | +0.000 | 1.000 |
| imbalanced | 3.300 | 3.200 | -0.100 | 1.000 |
| feature_availability | 3.000 | 3.000 | +0.000 | 1.000 |

*retrieval_precision*

| scenario | MV | F | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 1.000 | 1.000 | +0.000 | 1.000 |
| noisy_recurring | 1.000 | 1.000 | +0.000 | 1.000 |
| gradual | 0.944 | 0.914 | -0.030 | 1.000 |
| incremental | 0.699 | 0.669 | -0.030 | 1.000 |
| similar_x_different_y | 1.000 | 1.000 | +0.000 | 1.000 |
| aba | 1.000 | 1.000 | +0.000 | 1.000 |
| abca | 1.000 | 1.000 | +0.000 | 1.000 |
| aba_noisy | 1.000 | 1.000 | +0.000 | 1.000 |
| varied_sizes | 1.000 | 1.000 | +0.000 | 1.000 |
| overlapping | 0.967 | 0.967 | +0.000 | 1.000 |
| imbalanced | 1.000 | 1.000 | +0.000 | 1.000 |
| feature_availability | 1.000 | 1.000 | +0.000 | 1.000 |

*retrieval_recall*

| scenario | MV | F | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 1.000 | 1.000 | +0.000 | 1.000 |
| noisy_recurring | 0.560 | 0.594 | +0.035 | 1.000 |
| gradual | 0.400 | 0.428 | +0.029 | 1.000 |
| incremental | 0.495 | 0.429 | -0.066 | 1.000 |
| similar_x_different_y | 1.000 | 1.000 | +0.000 | 1.000 |
| aba | 1.000 | 1.000 | +0.000 | 1.000 |
| abca | 1.000 | 1.000 | +0.000 | 1.000 |
| aba_noisy | 0.900 | 0.900 | +0.000 | 1.000 |
| ab_gradual | 0.283 | 0.400 | +0.117 | 1.000 |
| varied_sizes | 1.000 | 1.000 | +0.000 | 1.000 |
| overlapping | 0.942 | 0.942 | +0.000 | 1.000 |
| imbalanced | 0.930 | 0.955 | +0.025 | 1.000 |
| feature_availability | 1.000 | 1.000 | +0.000 | 1.000 |

*false_match_rate*

| scenario | MV | F | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 0.000 | 0.000 | +0.000 | 1.000 |
| noisy_recurring | 0.000 | 0.000 | +0.000 | 1.000 |
| gradual | 0.023 | 0.046 | +0.023 | 1.000 |
| incremental | 0.151 | 0.140 | -0.011 | 1.000 |
| similar_x_different_y | 0.000 | 0.000 | +0.000 | 1.000 |
| sea | 0.330 | 0.330 | +0.000 | 1.000 |
| aba | 0.000 | 0.000 | +0.000 | 1.000 |
| abca | 0.000 | 0.000 | +0.000 | 1.000 |
| aba_noisy | 0.000 | 0.000 | +0.000 | 1.000 |
| ab_gradual | 0.000 | 0.000 | +0.000 | 1.000 |
| varied_sizes | 0.000 | 0.000 | +0.000 | 1.000 |
| overlapping | 0.017 | 0.017 | +0.000 | 1.000 |
| imbalanced | 0.000 | 0.000 | +0.000 | 1.000 |
| feature_availability | 0.000 | 0.000 | +0.000 | 1.000 |

*candidate_recall@1*

| scenario | MV | F | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 1.000 | 1.000 | +0.000 | 1.000 |
| noisy_recurring | 0.981 | 0.980 | -0.000 | 1.000 |
| gradual | 0.730 | 0.556 | -0.174 | 0.062 |
| incremental | 0.637 | 0.585 | -0.052 | 1.000 |
| similar_x_different_y | 0.439 | 0.645 | +0.205 | 0.117 |
| temporal_only | 0.722 | 0.728 | +0.006 | 1.000 |
| sea | 0.854 | 0.896 | +0.042 | 1.000 |
| hyperplane | 1.000 | 1.000 | +0.000 | 1.000 |
| aba | 1.000 | 1.000 | +0.000 | 1.000 |
| abca | 1.000 | 1.000 | +0.000 | 1.000 |
| aba_noisy | 1.000 | 1.000 | +0.000 | 1.000 |
| ab_gradual | 0.650 | 0.417 | -0.233 | 1.000 |
| varied_sizes | 1.000 | 1.000 | +0.000 | 1.000 |
| overlapping | 0.715 | 0.765 | +0.050 | 1.000 |
| imbalanced | 0.975 | 0.980 | +0.005 | 1.000 |
| feature_availability | 1.000 | 1.000 | +0.000 | 1.000 |


**MV schema 2.0 (baseline)  versus  v1 (legacy engine)** (per-scenario mean difference, exact paired sign-flip test over 10 seeds, Holm-adjusted p; * = adjusted p < 0.05)


*accuracy*

| scenario | v1 | MV | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 0.915 | 0.915 | -0.001 | 0.469 |
| noisy_recurring | 0.737 | 0.742 | +0.005* | 0.035 |
| gradual | 0.854 | 0.877 | +0.023* | 0.035 |
| incremental | 0.862 | 0.877 | +0.015* | 0.035 |
| similar_x_different_y | 0.911 | 0.910 | -0.001 | 0.469 |
| temporal_only | 0.935 | 0.934 | -0.001 | 0.834 |
| sea | 0.862 | 0.863 | +0.001* | 0.035 |
| hyperplane | 0.894 | 0.894 | +0.000 | 0.258 |
| aba | 0.907 | 0.907 | +0.000 | 1.000 |
| abca | 0.906 | 0.906 | -0.000 | 1.000 |
| aba_noisy | 0.813 | 0.817 | +0.004 | 0.178 |
| ab_gradual | 0.864 | 0.876 | +0.012* | 0.035 |
| diff_x_same_y | 0.928 | 0.930 | +0.002 | 0.258 |
| same_marginal_diff_corr | 0.934 | 0.934 | +0.000 | 1.000 |
| varied_sizes | 0.910 | 0.910 | -0.000 | 1.000 |
| overlapping | 0.912 | 0.911 | -0.001 | 1.000 |
| imbalanced | 0.905 | 0.905 | -0.000 | 1.000 |
| feature_availability | 0.895 | 0.897 | +0.002 | 0.531 |

*post_drift_acc*

| scenario | v1 | MV | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 0.826 | 0.822 | -0.004 | 0.711 |
| noisy_recurring | 0.652 | 0.671 | +0.019* | 0.033 |
| gradual | 0.719 | 0.722 | +0.003 | 1.000 |
| incremental | 0.752 | 0.760 | +0.008 | 0.711 |
| similar_x_different_y | 0.804 | 0.797 | -0.007 | 0.711 |
| temporal_only | 0.942 | 0.940 | -0.002 | 0.711 |
| sea | 0.827 | 0.831 | +0.004 | 0.250 |
| aba | 0.802 | 0.803 | +0.002 | 1.000 |
| abca | 0.791 | 0.788 | -0.003 | 1.000 |
| aba_noisy | 0.720 | 0.738 | +0.018 | 0.615 |
| ab_gradual | 0.744 | 0.742 | -0.001 | 1.000 |
| diff_x_same_y | 0.937 | 0.934 | -0.003 | 0.711 |
| same_marginal_diff_corr | 0.941 | 0.941 | +0.000 | 1.000 |
| varied_sizes | 0.812 | 0.809 | -0.004 | 1.000 |
| overlapping | 0.812 | 0.810 | -0.002 | 1.000 |
| imbalanced | 0.782 | 0.782 | +0.000 | 1.000 |
| feature_availability | 0.802 | 0.809 | +0.007 | 1.000 |

*models_trained*

| scenario | v1 | MV | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 4.000 | 4.000 | +0.000 | 1.000 |
| noisy_recurring | 12.400 | 8.600 | -3.800* | 0.035 |
| gradual | 9.600 | 12.900 | +3.300 | 0.088 |
| incremental | 9.900 | 11.500 | +1.600 | 0.109 |
| similar_x_different_y | 3.000 | 3.000 | +0.000 | 1.000 |
| temporal_only | 1.000 | 1.400 | +0.400 | 1.000 |
| sea | 2.100 | 1.600 | -0.500 | 0.750 |
| hyperplane | 1.000 | 1.000 | +0.000 | 1.000 |
| aba | 2.000 | 2.000 | +0.000 | 1.000 |
| abca | 3.000 | 3.000 | +0.000 | 1.000 |
| aba_noisy | 2.000 | 2.100 | +0.100 | 1.000 |
| ab_gradual | 2.900 | 3.900 | +1.000 | 0.203 |
| diff_x_same_y | 1.000 | 2.700 | +1.700* | 0.035 |
| same_marginal_diff_corr | 1.000 | 1.000 | +0.000 | 1.000 |
| varied_sizes | 3.100 | 3.000 | -0.100 | 1.000 |
| overlapping | 4.000 | 4.100 | +0.100 | 1.000 |
| imbalanced | 3.000 | 3.300 | +0.300 | 1.000 |
| feature_availability | 8.300 | 3.000 | -5.300* | 0.035 |

*retrieval_precision*

| scenario | v1 | MV | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 1.000 | 1.000 | +0.000 | 1.000 |
| noisy_recurring | 1.000 | 1.000 | +0.000 | 1.000 |
| gradual | 1.000 | 0.938 | -0.062 | 1.000 |
| incremental | 0.765 | 0.699 | -0.066 | 0.215 |
| similar_x_different_y | 1.000 | 1.000 | +0.000 | 1.000 |
| aba | 1.000 | 1.000 | +0.000 | 1.000 |
| abca | 1.000 | 1.000 | +0.000 | 1.000 |
| aba_noisy | 1.000 | 1.000 | +0.000 | 1.000 |
| varied_sizes | 1.000 | 1.000 | +0.000 | 1.000 |
| overlapping | 1.000 | 0.967 | -0.033 | 1.000 |
| imbalanced | 1.000 | 1.000 | +0.000 | 1.000 |

*retrieval_recall*

| scenario | v1 | MV | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 1.000 | 1.000 | +0.000 | 1.000 |
| noisy_recurring | 0.326 | 0.560 | +0.233 | 0.094 |
| gradual | 0.282 | 0.400 | +0.117 | 0.469 |
| incremental | 0.400 | 0.495 | +0.095 | 0.301 |
| similar_x_different_y | 1.000 | 1.000 | +0.000 | 1.000 |
| aba | 1.000 | 1.000 | +0.000 | 1.000 |
| abca | 1.000 | 1.000 | +0.000 | 1.000 |
| aba_noisy | 1.000 | 0.889 | -0.111 | 1.000 |
| ab_gradual | 0.300 | 0.283 | -0.017 | 1.000 |
| varied_sizes | 0.975 | 1.000 | +0.025 | 1.000 |
| overlapping | 1.000 | 0.942 | -0.058 | 1.000 |
| imbalanced | 1.000 | 0.930 | -0.070 | 1.000 |
| feature_availability | 0.000 | 1.000 | +1.000* | 0.025 |

*false_match_rate*

| scenario | v1 | MV | difference | adjusted p |
|---|---|---|---|---|
| abrupt_recurring | 0.000 | 0.000 | +0.000 | 1.000 |
| noisy_recurring | 0.000 | 0.000 | +0.000 | 1.000 |
| gradual | 0.000 | 0.023 | +0.023 | 1.000 |
| incremental | 0.085 | 0.151 | +0.066* | 0.027 |
| similar_x_different_y | 0.000 | 0.000 | +0.000 | 1.000 |
| sea | 0.400 | 0.330 | -0.070 | 1.000 |
| aba | 0.000 | 0.000 | +0.000 | 1.000 |
| abca | 0.000 | 0.000 | +0.000 | 1.000 |
| aba_noisy | 0.000 | 0.000 | +0.000 | 1.000 |
| ab_gradual | 0.000 | 0.000 | +0.000 | 1.000 |
| varied_sizes | 0.000 | 0.000 | +0.000 | 1.000 |
| overlapping | 0.000 | 0.017 | +0.017 | 1.000 |
| imbalanced | 0.000 | 0.000 | +0.000 | 1.000 |
| feature_availability | 0.000 | 0.000 | +0.000 | 1.000 |

*candidate_recall@1*

| scenario | v1 | MV | difference | adjusted p |
|---|---|---|---|---|


## A10. Scaling with repository size and fingerprint cost (E12)

### Matching latency vs repository size (d=4, query windows of 60 samples, 150 queries)

| concepts | configuration | Level 1 ms (mean / p95) | full match ms (mean / p95) | mean concepts scored | bytes per concept | build peak MB |
|---|---|---|---|---|---|---|
| 10 | basic, fixed k=5 (schema 2.0) | 0.072 / 0.079 | 3.45 / 3.70 | 5.0 | 1456 | 1.2 |
| 10 | basic, adaptive k | 0.072 / 0.077 | 2.12 / 2.17 | 3.0 | 1456 | 0.1 |
| 10 | shape_corr, adaptive k | 0.118 / 0.125 | 2.23 / 2.82 | 3.1 | 1456 | 0.1 |
| 10 | basic adaptive + full-scan fallback | 0.071 / 0.077 | 2.13 / 2.20 | 3.0 | 1456 | 0.1 |
| 10 | exhaustive level-2 scan | - | 6.6 / 6.7 | all | - | - |
| 50 | basic, fixed k=5 (schema 2.0) | 0.085 / 0.092 | 3.48 / 3.57 | 5.0 | 1456 | 0.4 |
| 50 | basic, adaptive k | 0.086 / 0.092 | 2.15 / 2.20 | 3.0 | 1456 | 0.4 |
| 50 | shape_corr, adaptive k | 0.132 / 0.140 | 2.54 / 4.48 | 3.6 | 1456 | 0.4 |
| 50 | basic adaptive + full-scan fallback | 0.084 / 0.090 | 2.13 / 2.21 | 3.0 | 1456 | 0.4 |
| 50 | exhaustive level-2 scan | - | 31.9 / 32.1 | all | - | - |
| 100 | basic, fixed k=5 (schema 2.0) | 0.100 / 0.106 | 3.45 / 3.62 | 5.0 | 1456 | 0.8 |
| 100 | basic, adaptive k | 0.101 / 0.112 | 2.14 / 2.20 | 3.0 | 1456 | 0.8 |
| 100 | shape_corr, adaptive k | 0.148 / 0.157 | 2.95 / 6.73 | 4.2 | 1456 | 0.8 |
| 100 | basic adaptive + full-scan fallback | 0.100 / 0.108 | 2.14 / 2.20 | 3.0 | 1456 | 0.8 |
| 100 | exhaustive level-2 scan | - | 64.1 / 65.1 | all | - | - |
| 500 | basic, fixed k=5 (schema 2.0) | 0.216 / 0.228 | 3.60 / 3.68 | 5.0 | 1456 | 4.0 |
| 500 | basic, adaptive k | 0.233 / 0.241 | 2.29 / 2.32 | 3.0 | 1456 | 4.0 |
| 500 | shape_corr, adaptive k | 0.288 / 0.300 | 4.53 / 6.94 | 6.4 | 1456 | 3.9 |
| 500 | basic adaptive + full-scan fallback | 0.218 / 0.226 | 2.28 / 2.32 | 3.0 | 1456 | 3.9 |
| 500 | exhaustive level-2 scan | - | 320.3 / 321.4 | all | - | - |
| 1000 | basic, fixed k=5 (schema 2.0) | 0.368 / 0.384 | 3.82 / 4.11 | 5.0 | 1456 | 7.9 |
| 1000 | basic, adaptive k | 0.367 / 0.379 | 2.45 / 2.49 | 3.0 | 1456 | 7.9 |
| 1000 | shape_corr, adaptive k | 0.458 / 0.477 | 5.77 / 7.19 | 8.0 | 1456 | 7.9 |
| 1000 | basic adaptive + full-scan fallback | 0.372 / 0.394 | 2.50 / 2.69 | 3.0 | 1456 | 7.9 |
| 1000 | exhaustive level-2 scan | - | 648.7 / 659.4 | all | - | - |

### Fingerprint extraction latency

| window | legacy v1 ms | multi-view ms (mean) | multi-view ms (p95) |
|---|---|---|---|
| n=40 d=4 | 0.044 | 1.090 | 1.131 |
| n=100 d=4 | 0.049 | 1.203 | 1.247 |
| n=300 d=4 | 0.068 | 1.404 | 1.455 |
| n=100 d=10 | 0.055 | 3.413 | 3.486 |
| n=100 d=30 | 0.062 | 14.838 | 15.063 |


## A11. Real datasets (E14): natural streams and constructed recurrence

**ELEC2 (45,312 rows, natural order)**: 45312 rows, positive rate 0.575 (majority-class accuracy 0.575). features ['day', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']; date dropped (a monotone counter)

| variant | accuracy | 1st half | 2nd half | worst 1000-window | models trained | drifts | recalls | false alarms | µs/sample |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.650 | 0.603 | 0.696 | 0.342 | 1 | 0 | 0 | 0 | 9 |
| retrain | 0.744 | 0.764 | 0.724 | 0.544 | 189 | 188 | 0 | 0 | 26 |
| v1 (legacy engine) | 0.742 | 0.772 | 0.711 | 0.557 | 82 | 191 | 105 | 5 | 23 |
| MV schema 2.0 (baseline) | 0.758 | 0.770 | 0.747 | 0.571 | 93 | 279 | 177 | 10 | 104 |
| D full multi-view (improved) | 0.765 | 0.775 | 0.755 | 0.601 | 67 | 193 | 111 | 16 | 141 |
| F = D without adaptive thresholds | 0.748 | 0.757 | 0.739 | 0.547 | 73 | 197 | 113 | 12 | 122 |

**Airlines (first 100,000 rows)**: 100000 rows, positive rate 0.359 (majority-class accuracy 0.641). features Airline(code), Flight, DayOfWeek, Time, Length; airports omitted; file order is not strictly chronological

| variant | accuracy | 1st half | 2nd half | worst 1000-window | models trained | drifts | recalls | false alarms | µs/sample |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.641 | 0.645 | 0.637 | 0.442 | 1 | 0 | 0 | 0 | 9 |
| retrain | 0.575 | 0.580 | 0.569 | 0.476 | 213 | 212 | 0 | 0 | 22 |
| v1 (legacy engine) | 0.572 | 0.583 | 0.562 | 0.466 | 212 | 231 | 15 | 5 | 27 |
| MV schema 2.0 (baseline) | 0.590 | 0.597 | 0.583 | 0.462 | 488 | 515 | 9 | 18 | 70 |
| D full multi-view (improved) | 0.574 | 0.580 | 0.567 | 0.478 | 178 | 247 | 41 | 29 | 107 |
| F = D without adaptive thresholds | 0.578 | 0.584 | 0.572 | 0.467 | 191 | 281 | 36 | 55 | 91 |

**Intel Berkeley mote 1 (first 60,000 readings)**: 43017 rows, positive rate 0.315 (majority-class accuracy 0.685). features ['temperature', 'humidity', 'light', 'voltage']; DERIVED label: temperature rises over the next 30 readings; raw sensor glitches kept

| variant | accuracy | 1st half | 2nd half | worst 1000-window | models trained | drifts | recalls | false alarms | µs/sample |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.685 | 0.719 | 0.652 | 0.079 | 1 | 0 | 0 | 0 | 9 |
| retrain | 0.753 | 0.739 | 0.767 | 0.424 | 128 | 127 | 0 | 0 | 24 |
| v1 (legacy engine) | 0.784 | 0.774 | 0.795 | 0.514 | 106 | 133 | 20 | 8 | 26 |
| MV schema 2.0 (baseline) | 0.750 | 0.732 | 0.769 | 0.452 | 229 | 251 | 12 | 11 | 69 |
| D full multi-view (improved) | 0.766 | 0.766 | 0.765 | 0.471 | 76 | 295 | 141 | 79 | 126 |
| F = D without adaptive thresholds | 0.755 | 0.738 | 0.772 | 0.448 | 171 | 269 | 65 | 34 | 183 |

**UCI load diagrams, 6 clients (first 70,000 intervals)**: 69999 rows, positive rate 0.237 (majority-class accuracy 0.763). features ['client_0', 'client_1', 'client_2', 'client_3', 'client_4', 'client_5', 'hour']; DERIVED label: total demand rises next interval

| variant | accuracy | 1st half | 2nd half | worst 1000-window | models trained | drifts | recalls | false alarms | µs/sample |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.763 | 1.000 | 0.525 | 0.496 | 1 | 0 | 0 | 0 | 9 |
| retrain | 0.800 | 1.000 | 0.600 | 0.480 | 34 | 33 | 0 | 0 | 20 |
| v1 (legacy engine) | 0.800 | 1.000 | 0.600 | 0.534 | 71 | 70 | 0 | 0 | 22 |
| MV schema 2.0 (baseline) | 0.799 | 1.000 | 0.599 | 0.517 | 46 | 56 | 8 | 3 | 58 |
| D full multi-view (improved) | 0.799 | 1.000 | 0.598 | 0.517 | 35 | 42 | 4 | 4 | 93 |
| F = D without adaptive thresholds | 0.799 | 1.000 | 0.598 | 0.517 | 35 | 42 | 4 | 4 | 72 |

**Constructed recurrence** (chronological blocks of a real stream replayed as A B A B A; ground truth = block; the repetition is an exact replay, not a natural recurrence)


*ELEC2 replay A B A B A*

| variant | accuracy | first 300 after drift | models trained | recalls | match precision | match recall | false-match | candidate R@1 |
|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 0.770 | 0.684 | 43 | 75 | 0.933 | 0.631 | 0.043 | n/a |
| MV schema 2.0 (baseline) | 0.770 | 0.671 | 50 | 122 | 0.984 | 0.714 | 0.012 | 0.983 |
| D full multi-view (improved) | 0.751 | 0.661 | 51 | 73 | 0.986 | 0.595 | 0.008 | 1.000 |
| F = D without adaptive thresholds | 0.754 | 0.630 | 51 | 83 | 0.988 | 0.626 | 0.008 | 0.986 |

*UCI replay A B A B A*

| variant | accuracy | first 300 after drift | models trained | recalls | match precision | match recall | false-match | candidate R@1 |
|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 0.839 | 0.770 | 23 | 0 | n/a | 0.000 | 0.000 | n/a |
| MV schema 2.0 (baseline) | 0.838 | 0.782 | 22 | 8 | 1.000 | 0.286 | 0.000 | 1.000 |
| D full multi-view (improved) | 0.837 | 0.775 | 16 | 3 | 1.000 | 0.176 | 0.000 | 1.000 |
| F = D without adaptive thresholds | 0.837 | 0.775 | 16 | 3 | 1.000 | 0.176 | 0.000 | 1.000 |

*Intel replay A B A B A*

| variant | accuracy | first 300 after drift | models trained | recalls | match precision | match recall | false-match | candidate R@1 |
|---|---|---|---|---|---|---|---|---|
| v1 (legacy engine) | 0.824 | 0.812 | 36 | 39 | 1.000 | 0.534 | 0.000 | n/a |
| MV schema 2.0 (baseline) | 0.805 | 0.777 | 144 | 14 | 0.929 | 0.084 | 0.006 | 0.822 |
| D full multi-view (improved) | 0.803 | 0.772 | 45 | 85 | 0.988 | 0.661 | 0.008 | 0.990 |
| F = D without adaptive thresholds | 0.792 | 0.818 | 90 | 61 | 0.984 | 0.405 | 0.007 | 0.983 |

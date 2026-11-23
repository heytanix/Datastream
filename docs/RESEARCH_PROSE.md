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

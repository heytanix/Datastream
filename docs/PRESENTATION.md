# Datastream: presentation outline

About 12 minutes. Every claim below comes from the code or from `python run.py bench`. Figures: `docs/architecture.pdf` page numbers in brackets.

## 1. The problem (1 min)
- A model learns a rule. The data changes the rule (concept drift) and accuracy collapses.
- Often the old rule returns. Retraining from scratch throws away what was already learned.
- Show: static model falls to 58.7% overall on the demo stream. [p5 plot, grey line]

## 2. The idea (1 min)
- Keep a **repository of concepts**: fingerprint + model.
- On drift, fingerprint the new data and look it up. Match: recall. No match: learn something new.

## 3. Architecture in one picture (2 min) [p1]
- Six layers, left to right: source, bus, processor, ML core, storage, dashboard.
- Green = built. Grey dashed = planned swap-ins (Kafka, Spark, Postgres).
- Key design rule: publishers, processor and dashboard only know the bus. Any part can restart without losing events.

## 4. Publish / subscribe (1 min) [p2]
- Three topics: `ds.raw`, `ds.events`, `ds.metrics`.
- Each consumer group has its own offset: the three strategies replay the same raw log, so the comparison is fair.
- The Sender drops (and counts) instead of blocking the source.

## 5. How a drift is handled (2 min) [p3]
- WARMUP, STABLE, RECOVERING.
- Detect: Page-Hinkley on the error stream (about 13 samples after the change) plus a label-free fingerprint-shift check.
- At 30 samples: early probe, does any stored model already explain the data? At 80: verify by fingerprint.
- Three outcomes: recall, false alarm, new concept.
- Quarantine queue: online learning is 30 samples late so an undetected change cannot corrupt a stored model. (Found during development: without it, a recalled model scored 60% on its own concept.)

## 6. The fingerprint (1.5 min) [p4]
- 17 numbers: means, stds, class gaps, feature/label correlation, label rate.
- Z-scored, RMS distance, threshold adapts to each concept's own noise (`d_in`) and a model-accuracy check guards against look-alike concepts.

## 7. Live demo (2 min)
- `python run.py serve`, open http://127.0.0.1:8000.
- Point at: the two strips (true concept vs concept Datastream believes), the accuracy dips and recoveries, the repository growing to 4 and staying at 4, the event log showing `probe` then `recall`.
- Bus panel: lag stays 0, dropped stays 0.

## 8. Results (1 min) [p5]
| | Overall | First 300 after drift | Models trained |
|---|---|---|---|
| static | 58.7% | 56.4% | 1 |
| retrain | 90.6% | 76.7% | 11 |
| Datastream | 91.8% | 83.2% | 4 |
- Honest framing: gain is in recovery speed and model count, not steady state.

## 9. Limits and next steps (0.5 min)
- Synthetic data, one seed, abrupt drifts only. Not yet tested on gradual drift or real streams.
- Not built: Kafka broker, Spark processor, real datasets, PostgreSQL.
- Next: real datasets, gradual drift, tune thresholds automatically, swap in Kafka.

## 10. Upgrade: multi-view fingerprinting (3 min) [PDF p5-7, p9]
- Problem with v1: one 17-number vector, nine of which use the label, so identity depended on labels and no "I am not sure" answer existed.
- v2: statistics + distributions + temporal (feature-only) and a separate optional predictive fingerprint. Missing evidence is NaN, never zero.
- Matching is hierarchical (fast global retrieval, feature-group ranking, predictive evidence last) and returns one of four decisions with a confidence score (explicitly not a probability).
- Safe recall: a matching fingerprint is only a hypothesis. Candidate models, including the active one, are shadow-validated on labelled samples; HIGH-confidence matches can be activated provisionally and rolled back.
- Honest results (5 seeds, synthetic): equal to v1 on clean recurrence, better on noisy streams (fewer models, 73% vs 45% retrieval recall), +2 points accuracy on gradual/incremental drift, but lower retrieval precision under incremental drift (61% vs 76%), about 3x per-sample cost, and blind to SEA-type concepts that share p(x).
- Two bugs the tests/benchmark found: autocorrelated data looked like constant drift (fixed with effective sample size), all-missing window looked like a perfect match (fixed).
- Demo: dashboard panels "latest concept match" (per-view distance bars, hatched = no evidence) and "recall decisions and validation".

## 11. Research pass: did the improvements help? (3 min) [PDF p8, p11; docs/RESEARCH.md]
- We audited the matcher first and found four real defects (merging, selection bias, worst-group aggregation, effective sample size).
- Then built and measured: noise models, richer distances, adaptive retrieval, conformal thresholds, calibrated confidence, paired validation, exact memory merging.
- Honest outcome on fresh seeds with exact paired tests: schema-2.0 MV beats v1 (accuracy better in 5 of 18 scenarios, never worse); the improved bundle does **not** beat schema 2.0 (no significant gain, worse in a few). Some parts earned their place (noise correction, predictive validation, exact merging), others did not (per-statistic ESS, score eviction, in-engine empirical thresholds).
- Lesson worth saying out loud: a better standalone matching operating point can lose recalls inside a pipeline whose validation already removes false matches.
- Not done: real datasets (needs permission to download).

## Likely questions
- *Why not just retrain faster?* A stored model is already fully trained; retraining needs fresh data first.
- *Is the bus really Kafka?* No. In-process stand-in with topics, offsets, groups, lag; the Kafka swap is listed as not built.
- *What if two concepts look alike?* The matcher also requires the stored model to score at least 75% on the new buffer.
- *Did the later improvements help?* Not on these synthetic streams; see the confirmation table (p11). They were tested on seeds that played no part in choosing them.
- *Is the new fingerprint better?* Not everywhere. See the table: equal on clean recurrence, better on noisy/gradual, worse precision under incremental drift, 3x slower.
- *Why is confidence not a probability?* It is a hand-built evidence score; it has not been calibrated against observed recall success.
- *Where do false alarms go?* Matched back to the active concept and logged as `false_alarm`; 2 occurred in the demo run.

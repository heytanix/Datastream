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
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
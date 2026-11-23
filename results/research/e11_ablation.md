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
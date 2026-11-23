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
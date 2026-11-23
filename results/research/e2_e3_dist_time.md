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
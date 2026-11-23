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
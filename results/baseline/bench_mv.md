Seeds: [1, 2, 3, 4, 5]. Values are mean±std over seeds. Synthetic data only.

### abrupt_recurring

| strategy | accuracy | post-drift acc | detect delay | models trained | recalls | precision | retrieval recall | false-match | missed-match | recall success | µs/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Static model | 58.9±0.8% | 55.8±0.6% | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | 5.4±0.4 |
| Retrain on drift | 90.3±0.3% | 76.5±0.7% | 17.2±2.1 | 11.4±0.5 | 0.0±0.0 | n/a | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | n/a | 12.1±0.1 |
| Datastream (v1) | 91.5±0.2% | 82.4±0.6% | 17.2±2.4 | 4.0±0.0 | 7.0±0.0 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | 11.6±0.1 |
| Datastream MV | 91.4±0.2% | 81.9±0.5% | 16.9±2.5 | 4.0±0.0 | 7.0±0.0 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | 38.0±0.2 |

### noisy_recurring

| strategy | accuracy | post-drift acc | detect delay | models trained | recalls | precision | retrieval recall | false-match | missed-match | recall success | µs/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Static model | 53.6±1.0% | 51.0±1.1% | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | 5.1±0.0 |
| Retrain on drift | 73.8±0.2% | 65.1±0.9% | 27.8±5.3 | 19.8±2.5 | 0.0±0.0 | n/a | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | n/a | 13.1±0.3 |
| Datastream (v1) | 73.9±0.4% | 65.0±1.2% | 33.4±13.3 | 10.4±3.4 | 4.6±1.4 | 96.0±8.0% | 44.7±18.3% | 1.7±3.3% | 55.3±18.3% | 66.7±18.0% | 12.3±0.5 |
| Datastream MV | 74.3±0.2% | 67.4±0.5% | 29.6±9.1 | 6.6±1.6 | 6.4±0.8 | 100.0±0.0% | 73.1±14.9% | 0.0±0.0% | 26.9±14.9% | 82.4±10.6% | 40.8±1.0 |

### gradual

| strategy | accuracy | post-drift acc | detect delay | models trained | recalls | precision | retrieval recall | false-match | missed-match | recall success | µs/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Static model | 58.6±1.0% | 55.2±1.6% | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | 5.1±0.0 |
| Retrain on drift | 85.2±0.4% | 71.3±1.0% | 135.4±5.7 | 11.8±0.4 | 0.0±0.0 | n/a | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | n/a | 12.1±0.1 |
| Datastream (v1) | 85.9±0.5% | 72.2±1.2% | 131.7±5.7 | 9.2±1.5 | 2.4±1.0 | 100.0±0.0% | 32.5±15.7% | 0.0±0.0% | 67.5±15.7% | 80.0±24.5% | 12.4±0.2 |
| Datastream MV | 88.4±0.6% | 72.6±1.2% | 110.4±4.0 | 13.0±0.6 | 6.2±0.7 | 100.0±0.0% | 40.7±2.8% | 0.0±0.0% | 59.3±2.8% | 88.1±10.9% | 41.8±0.9 |

### incremental

| strategy | accuracy | post-drift acc | detect delay | models trained | recalls | precision | retrieval recall | false-match | missed-match | recall success | µs/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Static model | 58.7±0.9% | 55.3±0.8% | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | 5.2±0.0 |
| Retrain on drift | 86.2±0.5% | 75.1±0.9% | 142.7±12.2 | 16.0±1.9 | 0.0±0.0 | n/a | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | n/a | 12.7±0.2 |
| Datastream (v1) | 85.6±0.6% | 74.7±1.3% | 146.1±10.9 | 10.2±2.4 | 3.8±1.0 | 76.0±24.8% | 33.5±12.1% | 7.3±6.7% | 66.5±12.1% | 50.7±14.4% | 12.4±0.4 |
| Datastream MV | 87.8±0.4% | 75.4±1.0% | 121.6±9.1 | 12.4±1.2 | 11.8±2.1 | 61.3±5.3% | 46.1±8.1% | 19.6±4.2% | 53.9±8.1% | 48.9±8.6% | 42.6±0.2 |

### similar_x_different_y  
_A and Ap are indistinguishable from X alone_

| strategy | accuracy | post-drift acc | detect delay | models trained | recalls | precision | retrieval recall | false-match | missed-match | recall success | µs/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Static model | 47.8±0.3% | 43.1±0.4% | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | 5.1±0.1 |
| Retrain on drift | 88.9±0.6% | 70.1±1.0% | 13.0±1.8 | 10.0±0.0 | 0.0±0.0 | n/a | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | n/a | 11.9±0.1 |
| Datastream (v1) | 91.1±0.2% | 80.2±0.7% | 12.9±1.6 | 3.0±0.0 | 7.0±0.0 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | 11.4±0.1 |
| Datastream MV | 90.9±0.2% | 79.1±0.3% | 12.7±1.8 | 3.0±0.0 | 7.0±0.0 | 100.0±0.0% | 100.0±0.0% | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | 38.2±0.1 |

### temporal_only  
_a static model is already optimal; this measures concept identification only_

| strategy | accuracy | post-drift acc | detect delay | models trained | recalls | precision | retrieval recall | false-match | missed-match | recall success | µs/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Static model | 93.4±1.1% | 93.4±1.6% | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | 5.2±0.0 |
| Retrain on drift | 92.9±0.3% | 93.0±0.5% | 259.2±191.8 | 2.2±0.7 | 0.0±0.0 | n/a | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | n/a | 11.1±0.2 |
| Datastream (v1) | 93.2±0.2% | 93.4±0.5% | 259.2±191.8 | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | 11.5±0.1 |
| Datastream MV | 93.1±0.4% | 93.3±0.6% | 322.6±118.2 | 1.6±0.5 | 0.8±0.7 | 16.7±23.6% | 100.0±0.0% | 44.4±7.9% | 0.0±0.0% | 100.0±0.0% | 37.2±0.4 |
| datastream_mv[time-weighted] | 93.2±0.3% | 93.3±0.5% | 128.6±26.1 | 1.4±0.5 | 0.0±0.0 | n/a | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | n/a | 38.1±0.2 |

### sea  
_feature-only fingerprints cannot separate SEA concepts_

| strategy | accuracy | post-drift acc | detect delay | models trained | recalls | precision | retrieval recall | false-match | missed-match | recall success | µs/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Static model | 82.4±0.7% | 82.3±0.7% | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | 5.2±0.0 |
| Retrain on drift | 85.8±0.3% | 83.1±1.1% | 166.3±84.3 | 7.8±1.7 | 0.0±0.0 | n/a | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | n/a | 11.9±0.3 |
| Datastream (v1) | 85.7±0.3% | 82.5±1.6% | 165.4±83.4 | 2.0±0.0 | 0.2±0.4 | 0.0±0.0% | n/a | 10.0±20.0% | n/a | 100.0±0.0% | 11.3±0.1 |
| Datastream MV | 85.8±0.3% | 82.8±1.4% | 165.1±83.3 | 1.6±0.5 | 0.0±0.0 | n/a | n/a | 0.0±0.0% | n/a | n/a | 33.2±0.2 |

### hyperplane  
_incremental drift, no recurring concepts_

| strategy | accuracy | post-drift acc | detect delay | models trained | recalls | precision | retrieval recall | false-match | missed-match | recall success | µs/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Static model | 71.6±2.5% | n/a | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | 5.2±0.0 |
| Retrain on drift | 89.6±0.3% | n/a | n/a | 4.0±0.9 | 0.0±0.0 | n/a | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | n/a | 11.3±0.1 |
| Datastream (v1) | 89.6±0.2% | n/a | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | 11.5±0.1 |
| Datastream MV | 89.7±0.2% | n/a | n/a | 1.0±0.0 | 0.0±0.0 | n/a | n/a | n/a | n/a | n/a | 63.6±0.1 |

### Memory per stored concept (fingerprint only, models excluded)

| strategy | bytes per concept |
|---|---|
| Datastream (v1) | 136 |
| Datastream MV | 1456 |

### Fingerprint cost

| window | legacy 17-d (ms) | multi-view (ms) |
|---|---|---|
| n100_d4 | 0.037 | 1.176 |
| n300_d4 | 0.052 | 1.368 |
| n100_d10 | 0.038 | 3.385 |

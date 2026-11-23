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
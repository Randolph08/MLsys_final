# E5.2 MT-Bench Category Analysis

本目录汇总全量 MT-Bench 80 records / 160 turns 在不同 category 上的结果。
这里把 MT-Bench 的 8 个 category 作为不同任务子集，用同一批全量实验日志做离线聚合；没有额外引入小样本手写 prompt。

## EAGLE-3 baseline vs naive

| category | turns | naive tok/s | EAGLE tok/s | speedup | turn match |
| --- | --- | --- | --- | --- | --- |
| writing | 20 | 17.49 | 66.58 | 3.806 | 80.0% |
| roleplay | 20 | 13.47 | 45.64 | 3.387 | 65.0% |
| reasoning | 20 | 20.62 | 79.16 | 3.839 | 85.0% |
| math | 20 | 20.97 | 89.37 | 4.262 | 95.0% |
| coding | 20 | 15.09 | 63.23 | 4.190 | 90.0% |
| extraction | 20 | 20.17 | 78.06 | 3.869 | 100.0% |
| stem | 20 | 14.73 | 57.02 | 3.870 | 80.0% |
| humanities | 20 | 16.51 | 61.84 | 3.745 | 80.0% |

## Fixed depth: depth 5 vs depth 11

| category | d5 tok/s | d5 accept | d11 tok/s | d11 accept | d11/d5 tok/s |
| --- | --- | --- | --- | --- | --- |
| writing | 77.35 | 5.150 | 74.13 | 5.777 | 0.958 |
| roleplay | 69.20 | 4.449 | 60.58 | 4.665 | 0.875 |
| reasoning | 65.20 | 5.237 | 75.88 | 6.027 | 1.164 |
| math | 56.73 | 5.816 | 86.66 | 6.833 | 1.528 |
| coding | 56.58 | 5.831 | 95.01 | 7.552 | 1.679 |
| extraction | 50.85 | 5.466 | 82.48 | 7.018 | 1.622 |
| stem | 61.68 | 5.389 | 77.58 | 5.981 | 1.258 |
| humanities | 77.22 | 5.141 | 52.60 | 5.818 | 0.681 |

## DDD threshold sweep by category

| category | fixed d11 tok/s | tau=-2 tok/s | tau=-2 depth | tau=-2 early | tau=-0.5 tok/s | tau=-0.5 depth | tau=-0.5 early |
| --- | --- | --- | --- | --- | --- | --- | --- |
| writing | 74.13 | 76.44 | 8.679 | 77.8% | 81.02 | 6.658 | 96.1% |
| roleplay | 60.58 | 66.56 | 7.882 | 91.4% | 52.80 | 6.238 | 99.3% |
| reasoning | 75.88 | 82.42 | 8.758 | 79.9% | 52.47 | 6.567 | 96.6% |
| math | 86.66 | 71.52 | 10.468 | 46.8% | 58.99 | 7.811 | 87.5% |
| coding | 95.01 | 62.59 | 10.485 | 44.3% | 65.21 | 8.063 | 82.7% |
| extraction | 82.48 | 55.29 | 10.068 | 53.6% | 81.77 | 7.923 | 83.0% |
| stem | 77.58 | 52.17 | 9.027 | 76.4% | 84.73 | 6.775 | 97.4% |
| humanities | 52.60 | 68.33 | 8.625 | 80.0% | 81.93 | 6.616 | 95.9% |

## Optimized DDD tau=-1.0 by category

| category | fixed d11 tok/s | tau=-1 tok/s | tau=-1 accept | tau=-1 depth | tau=-1 early | tau=-1/d11 tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| writing | 74.13 | 83.98 | 5.623 | 7.300 | 91.0% | 1.133 |
| roleplay | 60.58 | 71.59 | 4.597 | 6.600 | 97.7% | 1.182 |
| reasoning | 75.88 | 86.55 | 5.939 | 7.328 | 93.7% | 1.141 |
| math | 86.66 | 94.39 | 6.740 | 8.888 | 73.8% | 1.089 |
| coding | 95.01 | 103.01 | 7.464 | 9.041 | 67.2% | 1.084 |
| extraction | 82.48 | 87.69 | 6.745 | 8.662 | 72.5% | 1.063 |
| stem | 77.58 | 88.84 | 5.995 | 7.477 | 92.8% | 1.145 |
| humanities | 52.60 | 85.13 | 5.664 | 7.229 | 92.4% | 1.618 |

## OPT-Tree dynamic baseline by category

| category | EAGLE tok/s | OPT tok/s | OPT/EAGLE | EAGLE accept | OPT accept | accept delta |
| --- | --- | --- | --- | --- | --- | --- |
| writing | 51.23 | 42.58 | 0.831 | 5.150 | 5.150 | 0.000 |
| roleplay | 44.81 | 37.25 | 0.831 | 4.449 | 4.449 | 0.000 |
| reasoning | 51.84 | 43.13 | 0.832 | 5.237 | 5.237 | 0.000 |
| math | 53.11 | 47.98 | 0.903 | 5.816 | 5.816 | 0.000 |
| coding | 85.23 | 48.12 | 0.565 | 5.831 | 5.831 | 0.000 |
| extraction | 76.68 | 43.45 | 0.567 | 5.466 | 5.466 | 0.000 |
| stem | 78.44 | 47.44 | 0.605 | 5.389 | 5.389 | 0.000 |
| humanities | 50.46 | 66.26 | 1.313 | 5.141 | 5.141 | 0.000 |

## Main observations

- MT-Bench category 差异很明显：EAGLE baseline 的 category speedup 约在 3.39x 到 4.26x 之间，说明不同任务子集本身会显著改变复现结论。
- Full MT-Bench 上 depth 11 是比 depth 5 更强的固定深度点；按 category 看，大多数子集也呈现 depth 11 更优，但幅度不一致。
- 早期 DDD 阈值 tau=-2.0 / tau=-0.5 在 full MT-Bench 总体上没有超过 fixed depth 11；补充的 tau=-1.0 则在全量总体上成为当前最佳配置。
- tau=-1.0 的 category 表显示，不同任务子集仍有明显差异；在当前全量结果中，8 个 category 均高于 fixed depth 11，但提升幅度从 1.063x 到 1.618x 不等。
- OPT-Tree 的 accept delta 在所有 category 上都是 0，和 E4.1 的树集合 Jaccard=1.0 一致：在当前 EAGLE-3 框架里 OPT-Tree 没有改变最终候选树，也没有带来算法层面的接受长度提升。

详细机器可读结果见 `mt_bench_category_summary.json`。

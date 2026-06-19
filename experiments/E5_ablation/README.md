# E5.1 Final Ablation Matrix

本目录记录最终阶段补跑的 DDD+OPT 组合实验。该组合用于补齐主消融表，不作为新的算法改动。

## Saved Outputs

- `experiments/E5_ablation/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m2p0-m0p5_opt-1p5.json`
- `experiments/E5_ablation/ddd_sweep_mt_bench_limit-80_max-128_depth-11_tau-m1p0_opt-1p5.json`

## DDD+OPT Results

配置：MT-Bench 80 records / 160 turns，`max_new_tokens=128`，`ddd_max_depth=11`，`ddd_check_steps=5,7,9`，`opt_expand_factor=1.5`。

| Method | tok/s | accept/step | mean loop | avg depth | early stop |
|--------|-------|-------------|-----------|-----------|------------|
| DDD+OPT tau=-2.0 | 60.56 | 6.365 | 19.53 | 9.121 | 71.05% |
| DDD+OPT tau=-0.5 | 69.35 | 6.041 | 20.43 | 6.991 | 93.18% |
| DDD+OPT tau=-1.0 | 77.56 | 6.252 | 19.81 | 7.695 | 86.60% |

与 DDD-only 对比：

| Method | DDD-only tok/s | DDD+OPT tok/s | OPT ratio | accept delta | depth delta | early-stop delta |
|--------|----------------|---------------|-----------|--------------|-------------|------------------|
| tau=-2.0 | 68.51 | 60.56 | 0.884 | 0.000 | 0.000 | 0.00pp |
| tau=-0.5 | 70.74 | 69.35 | 0.980 | 0.000 | 0.000 | 0.00pp |
| tau=-1.0 | 88.27 | 77.56 | 0.879 | 0.000 | 0.000 | 0.00pp |

## Interpretation

DDD+OPT 的 `accept/step`、`mean loop count`、`avg depth`、`early stop rate` 与 DDD-only 完全对齐，说明在 DDD 动态深度路径下，OPT-Tree 仍然没有改变实际 acceptance 行为。

吞吐上，DDD+OPT 没有超过 DDD-only：`tau=-2.0` 明显下降，`tau=-0.5` 接近但仍低于 DDD-only；补跑优化后的 `tau=-1.0` 后，DDD+OPT 仍从 `88.27 tok/s` 降到 `77.56 tok/s`。结合 E4.1 的 tree-set diff，最终可以把 OPT-Tree 写成 strong EAGLE-3 dynamic baseline 上的 non-orthogonality negative result。

# Iterative uncertain benchmark repair (2026-04-12)

## iter0_20260412 uncertain averages
| mode | selection_score | min_recovery_ratio | cvr_eval | iar_eval | lipschitz_mean | wait_hold_usage_eval |
|---|---:|---:|---:|---:|---:|---:|
| baseline_rl | 0.257919 | 0.510443 | 0.000000 | 0.000000 | 76.511021 | 0.244444 |
| single_shot_llm | 0.247243 | 0.480743 | 0.022222 | 0.022222 | 145.510162 | 0.516667 |
| full_outer_loop | 0.270151 | 0.510000 | 0.016667 | 0.016667 | 161.043827 | 0.050000 |

Per-seed full_outer_loop:
| seed | selection_score | min_recovery_ratio | cvr_eval | iar_eval | lipschitz_mean | wait_hold_usage_eval | artifact |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.266395 | 0.510000 | 0.033333 | 0.033333 | 126.413383 | 0.033333 | outputs/iter0_20260412/full_outer_loop_benchmark_uncertain_presets_s42.json |
| 43 | 0.273907 | 0.510000 | 0.000000 | 0.000000 | 195.674271 | 0.066667 | outputs/iter0_20260412/full_outer_loop_benchmark_uncertain_presets_s43.json |

## iter1_20260412 uncertain averages
| mode | selection_score | min_recovery_ratio | cvr_eval | iar_eval | lipschitz_mean | wait_hold_usage_eval |
|---|---:|---:|---:|---:|---:|---:|
| baseline_rl | 0.257919 | 0.510443 | 0.000000 | 0.000000 | 76.511021 | 0.244444 |
| single_shot_llm | 0.266446 | 0.523016 | 0.027778 | 0.027778 | 156.672653 | 0.088889 |
| full_outer_loop | 0.266446 | 0.523016 | 0.027778 | 0.027778 | 156.672653 | 0.088889 |

Per-seed full_outer_loop:
| seed | selection_score | min_recovery_ratio | cvr_eval | iar_eval | lipschitz_mean | wait_hold_usage_eval | artifact |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.275896 | 0.536032 | 0.055556 | 0.055556 | 159.285492 | 0.033333 | outputs/iter1_20260412/full_outer_loop_benchmark_uncertain_presets_s42.json |
| 43 | 0.256996 | 0.510000 | 0.000000 | 0.000000 | 154.059814 | 0.144444 | outputs/iter1_20260412/full_outer_loop_benchmark_uncertain_presets_s43.json |

## iter2_20260412 uncertain averages
| mode | selection_score | min_recovery_ratio | cvr_eval | iar_eval | lipschitz_mean | wait_hold_usage_eval |
|---|---:|---:|---:|---:|---:|---:|
| baseline_rl | 0.257919 | 0.510443 | 0.000000 | 0.000000 | 76.511021 | 0.244444 |
| single_shot_llm | 0.277782 | 0.501271 | 0.000000 | 0.000000 | 160.885677 | 0.066667 |
| full_outer_loop | 0.283466 | 0.518771 | 0.000000 | 0.000000 | 158.136776 | 0.066667 |

Per-seed full_outer_loop:
| seed | selection_score | min_recovery_ratio | cvr_eval | iar_eval | lipschitz_mean | wait_hold_usage_eval | artifact |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.293025 | 0.527541 | 0.000000 | 0.000000 | 120.599281 | 0.066667 | outputs/iter2_20260412/full_outer_loop_benchmark_uncertain_presets_s42.json |
| 43 | 0.273907 | 0.510000 | 0.000000 | 0.000000 | 195.674271 | 0.066667 | outputs/iter2_20260412/full_outer_loop_benchmark_uncertain_presets_s43.json |

## Commands used
- `python3 run_benchmark_eval.py --mode <baseline_rl|single_shot_llm|full_outer_loop> --seed <42|43> --reward-mode engineered --split-name benchmark_uncertain_presets --severity moderate --out outputs/<iter>/...json`
- `python3 run_benchmark_eval.py --mode <baseline_rl|single_shot_llm|full_outer_loop> --seed 42 --reward-mode engineered --split-name benchmark_eval_presets --severity moderate --out outputs/iter2_20260412/...json`

## iter2_20260412 benchmark_eval_presets (seed 42)
| mode | selection_score | min_recovery_ratio | cvr_eval | iar_eval | lipschitz_mean | wait_hold_usage_eval | artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_rl | 0.283476 | 0.608333 | 0.000000 | 0.000000 | 109.382141 | 0.166667 | outputs/iter2_20260412/baseline_rl_benchmark_eval_presets_s42.json |
| single_shot_llm | 0.429467 | 0.683088 | 0.000000 | 0.000000 | 119.670509 | 0.000000 | outputs/iter2_20260412/single_shot_llm_benchmark_eval_presets_s42.json |
| full_outer_loop | 0.429467 | 0.683088 | 0.000000 | 0.000000 | 119.670509 | 0.000000 | outputs/iter2_20260412/full_outer_loop_benchmark_eval_presets_s42.json |

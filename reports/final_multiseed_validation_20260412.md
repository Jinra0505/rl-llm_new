# Final multi-seed validation (2026-04-12)

Reduced single_shot_llm seeds (uncertain: 3 seeds; eval: 1 seed) to keep runtime practical; baseline/full kept at full requested seed sets.

## benchmark_uncertain_presets per-run
| mode | seed | selection_score | min_recovery_ratio | cvr_eval | iar_eval | lipschitz_mean | wait_hold_usage_eval | completed | failed | artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline_rl | 42 | 0.242562 | 0.510000 | 0.000000 | 0.000000 | 86.671730 | 0.333333 | true | false | outputs/final_validation_20260412/baseline_rl_benchmark_uncertain_presets_s42.json |
| baseline_rl | 43 | 0.273277 | 0.510886 | 0.000000 | 0.000000 | 66.350311 | 0.155556 | true | false | outputs/final_validation_20260412/baseline_rl_benchmark_uncertain_presets_s43.json |
| baseline_rl | 44 | 0.264698 | 0.475000 | 0.033333 | 0.033333 | 81.602928 | 0.000000 | true | false | outputs/final_validation_20260412/baseline_rl_benchmark_uncertain_presets_s44.json |
| baseline_rl | 45 | 0.276417 | 0.510000 | 0.000000 | 0.000000 | 99.957893 | 0.066667 | true | false | outputs/final_validation_20260412/baseline_rl_benchmark_uncertain_presets_s45.json |
| baseline_rl | 46 | 0.176021 | 0.475000 | 0.200000 | 0.200000 | 155.208420 | 0.166667 | true | false | outputs/final_validation_20260412/baseline_rl_benchmark_uncertain_presets_s46.json |
| single_shot_llm | 42 | 0.293025 | 0.527541 | 0.000000 | 0.000000 | 120.599281 | 0.066667 | true | false | outputs/final_validation_20260412/single_shot_llm_benchmark_uncertain_presets_s42.json |
| single_shot_llm | 43 | 0.262540 | 0.475000 | 0.000000 | 0.000000 | 201.172073 | 0.066667 | true | false | outputs/final_validation_20260412/single_shot_llm_benchmark_uncertain_presets_s43.json |
| single_shot_llm | 44 | 0.242986 | 0.490273 | 0.000000 | 0.000000 | 140.806946 | 0.411111 | true | false | outputs/final_validation_20260412/single_shot_llm_benchmark_uncertain_presets_s44.json |
| full_outer_loop | 42 | 0.293025 | 0.527541 | 0.000000 | 0.000000 | 120.599281 | 0.066667 | true | false | outputs/final_validation_20260412/full_outer_loop_benchmark_uncertain_presets_s42.json |
| full_outer_loop | 43 | 0.273907 | 0.510000 | 0.000000 | 0.000000 | 195.674271 | 0.066667 | true | false | outputs/final_validation_20260412/full_outer_loop_benchmark_uncertain_presets_s43.json |
| full_outer_loop | 44 | 0.242986 | 0.490273 | 0.000000 | 0.000000 | 140.806946 | 0.411111 | true | false | outputs/final_validation_20260412/full_outer_loop_benchmark_uncertain_presets_s44.json |
| full_outer_loop | 45 | 0.265151 | 0.510000 | 0.000000 | 0.000000 | 160.069366 | 0.066667 | true | false | outputs/final_validation_20260412/full_outer_loop_benchmark_uncertain_presets_s45.json |
| full_outer_loop | 46 | 0.276421 | 0.510000 | 0.000000 | 0.000000 | 156.712967 | 0.066667 | true | false | outputs/final_validation_20260412/full_outer_loop_benchmark_uncertain_presets_s46.json |

## benchmark_uncertain_presets summary (mean ± sd [min, max])
| mode | selection_score | min_recovery_ratio | cvr_eval | iar_eval | lipschitz_mean | wait_hold_usage_eval | n |
|---|---|---|---|---|---|---|---:|
| baseline_rl | 0.246595 ± 0.037218 [0.176021, 0.276417] | 0.496177 ± 0.017294 [0.475000, 0.510886] | 0.046667 ± 0.077746 [0.000000, 0.200000] | 0.046667 ± 0.077746 [0.000000, 0.200000] | 97.958257 ± 30.579485 [66.350311, 155.208420] | 0.144444 ± 0.112437 [0.000000, 0.333333] | 5 |
| single_shot_llm | 0.266184 ± 0.020590 [0.242986, 0.293025] | 0.497605 ± 0.022068 [0.475000, 0.527541] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 154.192767 ± 34.228437 [120.599281, 201.172073] | 0.181481 ± 0.162373 [0.066667, 0.411111] | 3 |
| full_outer_loop | 0.270298 ± 0.016367 [0.242986, 0.293025] | 0.509563 ± 0.011797 [0.490273, 0.527541] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 154.772566 ± 24.769427 [120.599281, 195.674271] | 0.135556 ± 0.137778 [0.066667, 0.411111] | 5 |

## benchmark_eval_presets per-run
| mode | seed | selection_score | min_recovery_ratio | cvr_eval | iar_eval | lipschitz_mean | wait_hold_usage_eval | completed | failed | artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline_rl | 42 | 0.283476 | 0.608333 | 0.000000 | 0.000000 | 109.382141 | 0.166667 | true | false | outputs/final_validation_20260412/baseline_rl_benchmark_eval_presets_s42.json |
| baseline_rl | 43 | 0.528362 | 0.702198 | 0.000000 | 0.000000 | 215.913864 | 0.126984 | true | false | outputs/final_validation_20260412/baseline_rl_benchmark_eval_presets_s43.json |
| baseline_rl | 44 | 0.322735 | 0.630000 | 0.000000 | 0.000000 | 74.695290 | 0.000000 | true | false | outputs/final_validation_20260412/baseline_rl_benchmark_eval_presets_s44.json |
| single_shot_llm | 42 | 0.429467 | 0.683088 | 0.000000 | 0.000000 | 119.670509 | 0.000000 | true | false | outputs/final_validation_20260412/single_shot_llm_benchmark_eval_presets_s42.json |
| full_outer_loop | 42 | 0.429467 | 0.683088 | 0.000000 | 0.000000 | 119.670509 | 0.000000 | true | false | outputs/final_validation_20260412/full_outer_loop_benchmark_eval_presets_s42.json |
| full_outer_loop | 43 | 0.388726 | 0.630000 | 0.000000 | 0.000000 | 152.410049 | 0.000000 | true | false | outputs/final_validation_20260412/full_outer_loop_benchmark_eval_presets_s43.json |
| full_outer_loop | 44 | 0.544495 | 0.722166 | 0.000000 | 0.000000 | 133.094040 | 0.000000 | true | false | outputs/final_validation_20260412/full_outer_loop_benchmark_eval_presets_s44.json |

## benchmark_eval_presets summary (mean ± sd [min, max])
| mode | selection_score | min_recovery_ratio | cvr_eval | iar_eval | lipschitz_mean | wait_hold_usage_eval | n |
|---|---|---|---|---|---|---|---:|
| baseline_rl | 0.378191 ± 0.107390 [0.283476, 0.528362] | 0.646844 ± 0.040129 [0.608333, 0.702198] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 133.330432 ± 60.087780 [74.695290, 215.913864] | 0.097884 ± 0.071085 [0.000000, 0.166667] | 3 |
| single_shot_llm | 0.429467 ± 0.000000 [0.429467, 0.429467] | 0.683088 ± 0.000000 [0.683088, 0.683088] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 119.670509 ± 0.000000 [119.670509, 119.670509] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 1 |
| full_outer_loop | 0.454229 ± 0.065959 [0.388726, 0.544495] | 0.678418 ± 0.037771 [0.630000, 0.722166] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 135.058200 ± 13.437828 [119.670509, 152.410049] | 0.000000 ± 0.000000 [0.000000, 0.000000] | 3 |

## Safety / claim checks
- full_outer_loop strict uncertain safety across seeds: **True**
- full_outer_loop uncertain avg selection_score > baseline_rl: **True**
- full_outer_loop uncertain avg min_recovery_ratio competitive vs baseline_rl: **True**

# resource_moderate summary

## Notes
- single_shot_llm uses rerun files under outputs/final_paper_data/reruns/resource_moderate/.
- baseline/full_outer_loop use existing resource_constrained_validation_20260418_final files.
- ablation_fixed_global per-seed files were unavailable; topic_summary aggregate used as supplemental only.

## Method means (valid runs only)
### baseline_rl
- selection_score: 0.234888 ± 0.018043
- min_recovery_ratio: 0.431749 ± 0.027844
- critical_load_recovery_ratio: 0.655793 ± 0.047581
- communication_recovery_ratio: 0.488881 ± 0.025131
- power_recovery_ratio: 0.571201 ± 0.021354
- road_recovery_ratio: 0.433678 ± 0.030571
- constraint_violation_rate_eval: 0.044444 ± 0.062854
- invalid_action_rate_eval: 0.044444 ± 0.062854
- wait_hold_usage_eval: 0.259259 ± 0.151082
- mean_progress_delta_eval: 0.001817 ± 0.000438
- eval_success_rate: 0.000000 ± 0.000000
- safety_capacity_index: 0.667306 ± 0.014275
### single_shot_llm
- selection_score: 0.347684 ± 0.000993
- min_recovery_ratio: 0.589142 ± 0.000053
- critical_load_recovery_ratio: 0.917865 ± 0.003929
- communication_recovery_ratio: 0.636037 ± 0.000053
- power_recovery_ratio: 0.816493 ± 0.000708
- road_recovery_ratio: 0.589142 ± 0.000053
- constraint_violation_rate_eval: 0.000000 ± 0.000000
- invalid_action_rate_eval: 0.000000 ± 0.000000
- wait_hold_usage_eval: 0.350000 ± 0.000000
- mean_progress_delta_eval: 0.004871 ± 0.000006
- eval_success_rate: 0.000000 ± 0.000000
- safety_capacity_index: 0.827452 ± 0.001393
### full_outer_loop
- selection_score: 0.260865 ± 0.017147
- min_recovery_ratio: 0.456753 ± 0.018880
- critical_load_recovery_ratio: 0.677083 ± 0.054937
- communication_recovery_ratio: 0.506528 ± 0.025078
- power_recovery_ratio: 0.581562 ± 0.047044
- road_recovery_ratio: 0.464201 ± 0.026102
- constraint_violation_rate_eval: 0.000000 ± 0.000000
- invalid_action_rate_eval: 0.000000 ± 0.000000
- wait_hold_usage_eval: 0.359259 ± 0.180154
- mean_progress_delta_eval: 0.003044 ± 0.001243
- eval_success_rate: 0.000000 ± 0.000000
- safety_capacity_index: 0.696843 ± 0.025077
### ablation_fixed_global
- selection_score: 0.343494 ± 0.000000
- min_recovery_ratio: 0.589194 ± 0.000000
- critical_load_recovery_ratio: 0.901092 ± 0.000000
- communication_recovery_ratio: 0.000000 ± 0.000000
- power_recovery_ratio: 0.000000 ± 0.000000
- road_recovery_ratio: 0.000000 ± 0.000000
- constraint_violation_rate_eval: 0.000000 ± 0.000000
- invalid_action_rate_eval: 0.000000 ± 0.000000
- wait_hold_usage_eval: 0.350000 ± 0.000000
- mean_progress_delta_eval: 0.000000 ± 0.000000
- eval_success_rate: 0.000000 ± 0.000000
- safety_capacity_index: 0.821600 ± 0.000000
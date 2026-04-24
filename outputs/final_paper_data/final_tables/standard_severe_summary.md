# standard_severe summary

## Notes
- standard_severe preserved as supplemental robustness summary; no rerun performed.
- Do not claim full_outer_loop dominates all methods if single_shot_llm or ablation_fixed_global has higher selection_score or critical_load_recovery_ratio.
- Per-seed standard_severe JSONs were unavailable in current repository; summary derived from existing topic_summary.

## Method means (valid runs only)
### baseline_rl
- selection_score: 0.313295 ± 0.000000
- min_recovery_ratio: 0.586757 ± 0.000000
- critical_load_recovery_ratio: 0.849502 ± 0.000000
- communication_recovery_ratio: 0.000000 ± 0.000000
- power_recovery_ratio: 0.000000 ± 0.000000
- road_recovery_ratio: 0.000000 ± 0.000000
- constraint_violation_rate_eval: 0.080556 ± 0.000000
- invalid_action_rate_eval: 0.080556 ± 0.000000
- wait_hold_usage_eval: 0.255556 ± 0.000000
- mean_progress_delta_eval: 0.000000 ± 0.000000
- eval_success_rate: 0.000000 ± 0.000000
- safety_capacity_index: 0.778524 ± 0.000000
### single_shot_llm
- selection_score: 0.419033 ± 0.000000
- min_recovery_ratio: 0.619127 ± 0.000000
- critical_load_recovery_ratio: 0.868503 ± 0.000000
- communication_recovery_ratio: 0.000000 ± 0.000000
- power_recovery_ratio: 0.000000 ± 0.000000
- road_recovery_ratio: 0.000000 ± 0.000000
- constraint_violation_rate_eval: 0.000000 ± 0.000000
- invalid_action_rate_eval: 0.000000 ± 0.000000
- wait_hold_usage_eval: 0.263413 ± 0.000000
- mean_progress_delta_eval: 0.000000 ± 0.000000
- eval_success_rate: 0.000000 ± 0.000000
- safety_capacity_index: 0.820671 ± 0.000000
### full_outer_loop
- selection_score: 0.339774 ± 0.000000
- min_recovery_ratio: 0.617104 ± 0.000000
- critical_load_recovery_ratio: 0.864008 ± 0.000000
- communication_recovery_ratio: 0.000000 ± 0.000000
- power_recovery_ratio: 0.000000 ± 0.000000
- road_recovery_ratio: 0.000000 ± 0.000000
- constraint_violation_rate_eval: 0.000000 ± 0.000000
- invalid_action_rate_eval: 0.000000 ± 0.000000
- wait_hold_usage_eval: 0.255556 ± 0.000000
- mean_progress_delta_eval: 0.000000 ± 0.000000
- eval_success_rate: 0.000000 ± 0.000000
- safety_capacity_index: 0.818389 ± 0.000000
### ablation_fixed_global
- selection_score: 0.433083 ± 0.000000
- min_recovery_ratio: 0.622741 ± 0.000000
- critical_load_recovery_ratio: 0.877047 ± 0.000000
- communication_recovery_ratio: 0.000000 ± 0.000000
- power_recovery_ratio: 0.000000 ± 0.000000
- road_recovery_ratio: 0.000000 ± 0.000000
- constraint_violation_rate_eval: 0.000000 ± 0.000000
- invalid_action_rate_eval: 0.000000 ± 0.000000
- wait_hold_usage_eval: 0.262749 ± 0.000000
- mean_progress_delta_eval: 0.000000 ± 0.000000
- eval_success_rate: 0.000000 ± 0.000000
- safety_capacity_index: 0.824926 ± 0.000000
# resource_moderate summary

## Method means (with evidence metadata)
### baseline_rl
- evidence_level: per_seed_n3
- source_quality_note: Complete 3-seed per-seed evidence.
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
- evidence_level: valid_seed_n2
- source_quality_note: 3 seeds attempted; seed42 is sentinel-invalid and excluded from valid means.
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
- evidence_level: per_seed_n3
- source_quality_note: Complete 3-seed per-seed evidence.
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
- evidence_level: aggregate_only_n1
- source_quality_note: Aggregate-only source; per-seed files unavailable.
- selection_score: 0.343494 ± 0.000000
- min_recovery_ratio: 0.589194 ± 0.000000
- critical_load_recovery_ratio: 0.901092 ± 0.000000
- communication_recovery_ratio: N/A ± N/A
- power_recovery_ratio: N/A ± N/A
- road_recovery_ratio: N/A ± N/A
- constraint_violation_rate_eval: 0.000000 ± 0.000000
- invalid_action_rate_eval: 0.000000 ± 0.000000
- wait_hold_usage_eval: 0.350000 ± 0.000000
- mean_progress_delta_eval: N/A ± N/A
- eval_success_rate: 0.000000 ± 0.000000
- safety_capacity_index: 0.821600 ± 0.000000
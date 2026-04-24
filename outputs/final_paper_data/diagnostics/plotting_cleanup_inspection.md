# Plotting Cleanup Inspection

## Existing process CSV files
- action_usage_long.csv
- representative_eval_trace_long.csv
- candidate_selection_trace.csv
- outer_loop_round_summary.csv
- routing_trace.csv
- llm_call_summary.csv
- eval_trajectory_summary.csv
- stage_distribution_long.csv
- resource_end_summary.csv
- reward_curves_long.csv
- zone_layer_recovery_long.csv

## Missing action_category counts
- action_usage_long.csv: 255
- representative_eval_trace_long.csv: 132

## candidate_selection_trace vs outer_loop_round_summary
- mismatches: 2
- ('resource_moderate', 'single_shot_llm', '43', '1') -> r1_anchor
- ('resource_moderate', 'single_shot_llm', '44', '1') -> r1_backstop

## unavailable metrics encoded as 0
- count: 20
- standard_severe / baseline_rl / communication_recovery_ratio
- standard_severe / baseline_rl / power_recovery_ratio
- standard_severe / baseline_rl / road_recovery_ratio
- standard_severe / baseline_rl / mean_progress_delta_eval
- standard_severe / single_shot_llm / communication_recovery_ratio
- standard_severe / single_shot_llm / power_recovery_ratio
- standard_severe / single_shot_llm / road_recovery_ratio
- standard_severe / single_shot_llm / mean_progress_delta_eval
- standard_severe / full_outer_loop / communication_recovery_ratio
- standard_severe / full_outer_loop / power_recovery_ratio
- standard_severe / full_outer_loop / road_recovery_ratio
- standard_severe / full_outer_loop / mean_progress_delta_eval
- standard_severe / ablation_fixed_global / communication_recovery_ratio
- standard_severe / ablation_fixed_global / power_recovery_ratio
- standard_severe / ablation_fixed_global / road_recovery_ratio
- standard_severe / ablation_fixed_global / mean_progress_delta_eval
- resource_moderate / ablation_fixed_global / communication_recovery_ratio
- resource_moderate / ablation_fixed_global / power_recovery_ratio
- resource_moderate / ablation_fixed_global / road_recovery_ratio
- resource_moderate / ablation_fixed_global / mean_progress_delta_eval

## aggregate-only or partial-valid summaries
- resource_moderate / single_shot_llm: partial_valid
- resource_moderate / ablation_fixed_global: aggregate_only_or_single
- standard_severe / baseline_rl: aggregate_only_or_single
- standard_severe / single_shot_llm: aggregate_only_or_single
- standard_severe / full_outer_loop: aggregate_only_or_single
- standard_severe / ablation_fixed_global: aggregate_only_or_single
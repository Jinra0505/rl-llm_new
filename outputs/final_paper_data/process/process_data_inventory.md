# Process Data Inventory
## action_usage_long.csv
- row_count: 255
- columns: scenario, method, seed, action, action_name, action_label, action_category, usage_rate
- scenarios: resource_moderate, standard_moderate
- methods: baseline_rl, full_outer_loop, single_shot_llm
- seeds: 42, 43, 44
- suitable figure types: action composition, representative process timeline
- limitations: representative/process focus; not full statistical trajectory for all rounds

## representative_eval_trace_long.csv
- row_count: 222
- columns: scenario, method, seed, step, action, action_name, action_label, action_category, progress_delta, stage, invalid_action, invalid_reason, constraint_violation
- scenarios: resource_moderate, standard_moderate
- methods: baseline_rl, full_outer_loop, single_shot_llm
- seeds: 42, 43, 44
- suitable figure types: action composition, representative process timeline
- limitations: representative/process focus; not full statistical trajectory for all rounds

## eval_trajectory_summary.csv
- row_count: 17
- columns: scenario, method, seed, mean_steps, terminated_rate, truncated_rate, mean_invalid_action_rate, mean_constraint_violation_rate, mean_progress_delta
- scenarios: resource_moderate, standard_moderate
- methods: baseline_rl, full_outer_loop, single_shot_llm
- seeds: 42, 43, 44
- suitable figure types: summary/supporting process visualization
- limitations: 

## resource_end_summary.csv
- row_count: 17
- columns: scenario, method, seed, mes_soc_end_mean, material_stock_end_mean, switching_capability_end_mean, crew_power_status_end_mean, crew_comm_status_end_mean, crew_road_status_end_mean
- scenarios: resource_moderate, standard_moderate
- methods: baseline_rl, full_outer_loop, single_shot_llm
- seeds: 42, 43, 44
- suitable figure types: summary/supporting process visualization
- limitations: 

## reward_curves_long.csv
- row_count: 442
- columns: scenario, method, seed, phase, episode, reward
- scenarios: resource_moderate, standard_moderate
- methods: baseline_rl, full_outer_loop, single_shot_llm
- seeds: 42, 43, 44
- suitable figure types: summary/supporting process visualization
- limitations: 

## stage_distribution_long.csv
- row_count: 25
- columns: scenario, method, seed, stage, usage_rate
- scenarios: resource_moderate, standard_moderate
- methods: baseline_rl, full_outer_loop, single_shot_llm
- seeds: 42, 43, 44
- suitable figure types: summary/supporting process visualization
- limitations: 

## zone_layer_recovery_long.csv
- row_count: 204
- columns: scenario, method, seed, zone, layer, recovery_ratio
- scenarios: resource_moderate, standard_moderate
- methods: baseline_rl, full_outer_loop, single_shot_llm
- seeds: 42, 43, 44
- suitable figure types: summary/supporting process visualization
- limitations: 

## outer_loop_round_summary.csv
- row_count: 2
- columns: scenario, method, seed, round, selected_candidate_id, candidate_origin, task_mode, phase_mode, selection_score, min_recovery_ratio, critical_load_recovery_ratio, constraint_violation_rate_eval, invalid_action_rate_eval, wait_hold_usage_eval
- scenarios: resource_moderate
- methods: single_shot_llm
- seeds: 43, 44
- suitable figure types: case-level mechanism illustration
- limitations: limited rows; avoid strong statistical evolution claims

## candidate_selection_trace.csv
- row_count: 6
- columns: scenario, method, seed, round, candidate_id, candidate_origin, valid, selected, rejected, rejection_reasons, selection_score, min_recovery_ratio, critical_load_recovery_ratio, constraint_violation_rate_eval, invalid_action_rate_eval, wait_hold_usage_eval, selected_by_round_summary, rejection_stage
- scenarios: resource_moderate
- methods: single_shot_llm
- seeds: 43, 44
- suitable figure types: case-level mechanism illustration
- limitations: limited rows; avoid strong statistical evolution claims

## routing_trace.csv
- row_count: 2
- columns: scenario, method, seed, round, task_mode, confidence, dominant_signal, competing_signal, reason, source
- scenarios: resource_moderate
- methods: single_shot_llm
- seeds: 43, 44
- suitable figure types: case-level mechanism illustration
- limitations: limited rows; avoid strong statistical evolution claims

## llm_call_summary.csv
- row_count: 2
- columns: scenario, method, seed, response_kind, model, success, latency_sec, content_len, reasoning_content_len, error
- scenarios: resource_moderate
- methods: single_shot_llm
- seeds: 43, 44
- suitable figure types: LLM call diagnostics
- limitations: sparse model/latency/content fields in current data

- action_usage_long and representative_eval_trace_long are suitable for action-composition and representative-process figures after action mapping fix.
- candidate_selection_trace / outer_loop_round_summary / routing_trace are only suitable for case-level mechanism illustration because rows are limited.
- llm_call_summary is currently weak because model/latency/content fields are sparse.
- per-preset metrics are unavailable unless future reruns save per-eval-episode/per-preset metrics.
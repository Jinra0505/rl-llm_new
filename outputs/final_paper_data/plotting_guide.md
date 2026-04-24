# Plotting Guide

## Main-text candidate figures
### 1. Main performance comparison under standard_moderate
- data: outputs/final_paper_data/final_tables/figure_ready_metrics.csv, outputs/final_paper_data/final_tables/standard_moderate_per_seed.csv
- metrics: selection_score, min_recovery_ratio, critical_load_recovery_ratio, safety_capacity_index, invalid_action_rate_eval, wait_hold_usage_eval
- recommended plot: grouped dot plot with mean±std (or grouped bars with dots)
- note: Do not claim full_outer_loop best on every metric; single_shot has highest selection_score while full_outer_loop emphasizes recovery-floor robustness/lower wait.

### 2. Zone-layer recovery heatmap under standard_moderate
- data: outputs/final_paper_data/process/zone_layer_recovery_long.csv
- recommended plot: method × zone/layer heatmap
- note: Shows spatial-layer recovery structure.

### 3. Representative recovery/action process
- data: outputs/final_paper_data/process/representative_eval_trace_long.csv
- recommended plot: step-wise progress curve + action-category ribbon
- note: Representative process illustration, not full statistical trajectory.

### 4. Resource-end comparison
- data: outputs/final_paper_data/process/resource_end_summary.csv
- metrics: mes_soc_end_mean, material_stock_end_mean, switching_capability_end_mean, crew_power_status_end_mean, crew_comm_status_end_mean, crew_road_status_end_mean
- recommended plot: grouped dot plot or compact radar-like summary
- note: Use as main or supplement depending on space.

## Supplemental figures
### 5. resource_moderate summary
- data: resource_moderate_summary, figure_ready_metrics
- note: single_shot n_valid=2; ablation aggregate-only.

### 6. standard_severe robustness summary
- data: standard_severe_summary, figure_ready_metrics
- note: aggregate-only n=1; supplemental robustness only.

### 7. Reward curves
- data: reward_curves_long.csv
- note: supplement.

### 8. Stage distribution
- data: stage_distribution_long.csv
- note: supplement.

### 9. Candidate selection case study
- data: candidate_selection_trace.csv, outer_loop_round_summary.csv, routing_trace.csv
- note: case-level mechanism illustration only.

## Figures to avoid
- Per-preset performance figures (per-preset metrics unavailable).
- Strong statistical outer-loop evolution claims (outer-loop trace rows limited).
- Plots that treat unavailable aggregate-only layer metrics as zero.
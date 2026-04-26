# plotting_readiness_summary

- Can high-pressure cumulative process be plotted? yes
- Can high-pressure stepwise process be plotted? yes
- Can high-pressure action heatmap be plotted? yes

## Files to send to plotting assistant
- final_tables/figure_ready_metrics.csv
- process_exports/resource_moderate_mean_cumulative_progress.csv
- process_exports/standard_severe_mean_cumulative_progress.csv
- process_exports/resource_moderate_mean_stepwise_progress.csv
- process_exports/standard_severe_mean_stepwise_progress.csv
- mechanism_exports/resource_moderate_action_category_share.csv
- mechanism_exports/standard_severe_action_category_share.csv
- mechanism_exports/resource_moderate_stage_share.csv
- mechanism_exports/standard_severe_stage_share.csv

## Files not to use for final paper figures
- diagnostics/step_level_trace_sample.csv (diagnostic trace only)
- _tmp_raw_reruns/* (intermediate run artifacts)

- Should V8 replace V6 or remain diagnostic? replace V6
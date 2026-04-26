# data_integrity_check

- final_tables/standard_moderate_summary.csv: exists=yes, readable=yes, rows=3, cols=25
- final_tables/resource_moderate_summary.csv: exists=yes, readable=yes, rows=3, cols=25
- final_tables/standard_severe_summary.csv: exists=yes, readable=yes, rows=3, cols=25
- final_tables/robustness_stress_summary.csv: exists=yes, readable=yes, rows=3, cols=25
- final_tables/figure_ready_metrics.csv: exists=yes, readable=yes, rows=132, cols=6
- per_seed/standard_moderate_per_seed.csv: exists=yes, readable=yes, rows=9, cols=21
- per_seed/resource_moderate_per_seed.csv: exists=yes, readable=yes, rows=9, cols=21
- per_seed/standard_severe_per_seed.csv: exists=yes, readable=yes, rows=9, cols=21
- process_exports/standard_moderate_mean_cumulative_progress.csv: exists=yes, readable=yes, rows=120, cols=6
- process_exports/resource_moderate_mean_cumulative_progress.csv: exists=yes, readable=yes, rows=120, cols=6
- process_exports/standard_severe_mean_cumulative_progress.csv: exists=yes, readable=yes, rows=120, cols=6
- process_exports/standard_moderate_mean_stepwise_progress.csv: exists=yes, readable=yes, rows=120, cols=6
- process_exports/resource_moderate_mean_stepwise_progress.csv: exists=yes, readable=yes, rows=120, cols=6
- process_exports/standard_severe_mean_stepwise_progress.csv: exists=yes, readable=yes, rows=120, cols=6
- mechanism_exports/standard_moderate_action_category_share.csv: exists=yes, readable=yes, rows=17, cols=5
- mechanism_exports/resource_moderate_action_category_share.csv: exists=yes, readable=yes, rows=12, cols=5
- mechanism_exports/standard_severe_action_category_share.csv: exists=yes, readable=yes, rows=15, cols=5
- mechanism_exports/standard_moderate_stage_share.csv: exists=yes, readable=yes, rows=6, cols=5
- mechanism_exports/resource_moderate_stage_share.csv: exists=yes, readable=yes, rows=3, cols=5
- mechanism_exports/standard_severe_stage_share.csv: exists=yes, readable=yes, rows=6, cols=5
- diagnostics/v6_comparison_summary.csv: exists=yes, readable=yes, rows=54, cols=8

## Coverage checks
- standard_moderate: methods_present=['baseline_rl', 'full_outer_loop', 'single_shot_llm'], methods_ok=yes, scenario_ok=yes
- resource_moderate: methods_present=['baseline_rl', 'full_outer_loop', 'single_shot_llm'], methods_ok=yes, scenario_ok=yes
- standard_severe: methods_present=['baseline_rl', 'full_outer_loop', 'single_shot_llm'], methods_ok=yes, scenario_ok=yes
- resource_moderate_mean_cumulative_progress: methods_ok=yes
- resource_moderate_mean_stepwise_progress: methods_ok=yes
- resource_moderate_action_category_share: methods_ok=yes
- resource_moderate_stage_share: methods_ok=yes
- standard_severe_mean_cumulative_progress: methods_ok=yes
- standard_severe_mean_stepwise_progress: methods_ok=yes
- standard_severe_action_category_share: methods_ok=yes
- standard_severe_stage_share: methods_ok=yes

## Summary
- all required plotting CSVs exist: yes
- all required plotting CSVs readable: yes
- row counts nonzero for required plotting CSVs: yes
- required methods present: yes
- required scenarios present: yes
- high-pressure process files include all three main methods: yes
- high-pressure mechanism files include all three main methods: yes
- action-category shares sum approximately to 1: yes (max deviation=0.223611)
- stage shares sum approximately to 1: yes (max deviation=0.000000)
- no required plotting file is header-only: yes
- optional file exclusions: none (candidate diagnostics included as compact CSV)
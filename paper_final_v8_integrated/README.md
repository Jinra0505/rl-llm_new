# paper_final_v8_integrated

Final integrated rerun package for paper-ready tables and plotting exports from one coherent run.

## Scenarios
- standard_moderate
- resource_moderate
- standard_severe

## Methods
- baseline_rl
- single_shot_llm
- full_outer_loop
- ablation_fixed_global: not included in V8 rerun (optional, skipped for runtime/control)

## Seeds
- 42, 43, 44

## Main plotting files
- final_tables/figure_ready_metrics.csv
- process_exports/resource_moderate_mean_cumulative_progress.csv
- process_exports/standard_severe_mean_cumulative_progress.csv
- mechanism_exports/resource_moderate_action_category_share.csv
- mechanism_exports/standard_severe_action_category_share.csv

## Supplementary plotting files
- process_exports/*_mean_stepwise_progress.csv
- mechanism_exports/*_stage_share.csv
- per_seed/*_per_seed.csv

## Diagnostics
- diagnostics/final_v8_consistency_check.md
- diagnostics/v6_comparison_summary.md

## Recommendation
- V8 recommended as final source of truth: yes
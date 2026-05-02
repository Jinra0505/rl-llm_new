# paper_final_v9_rule_based_greedy_only_text

This package contains only newly generated `rule_based_greedy` baseline results.
Other methods (`baseline_rl`, `single_shot_llm`, `full_outer_loop`) should be taken from the existing V8 package.

Scenarios:
- standard_moderate
- resource_moderate
- standard_severe

Seeds:
- 42, 43, 44

Output files:
- diagnostics/rerun_status.csv
- diagnostics/action_mapping_note.md
- diagnostics/final_consistency_check.md
- per_seed/rule_based_greedy_per_seed.csv
- final_tables/rule_based_greedy_summary.csv
- final_tables/rule_based_greedy_figure_ready_metrics.csv
- process_exports/rule_based_greedy_mean_cumulative_progress.csv
- process_exports/rule_based_greedy_mean_stepwise_progress.csv
- mechanism_exports/rule_based_greedy_action_category_share.csv
- mechanism_exports/rule_based_greedy_stage_share.csv

Merge note:
- Append `per_seed/rule_based_greedy_per_seed.csv` rows to existing V8 per-seed table.
- Append `final_tables/rule_based_greedy_summary.csv` rows to scenario-method summary table.
- Append `final_tables/rule_based_greedy_figure_ready_metrics.csv` rows to V8 `figure_ready_metrics.csv`.

No binary files were intentionally generated.


Additional merge note:
- This package contains only the rule_based_greedy baseline. Other methods should be taken from the existing V8 package.
- Append `per_seed/rule_based_greedy_per_seed.csv` to the V8 per-seed table.
- Append `final_tables/rule_based_greedy_summary.csv` to the V8 scenario-method summary table.
- Append `final_tables/rule_based_greedy_figure_ready_metrics.csv` to the V8 `figure_ready_metrics.csv`.
- Append `mechanism_exports/rule_based_greedy_action_category_share.csv` to the corresponding V8 mechanism/action-category table after checking category definitions.

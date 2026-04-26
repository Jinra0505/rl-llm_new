# final_consistency_check

## Source of truth
- V5 per_seed CSV files were used as the sole authoritative source for regeneration.

## full_outer_loop vs single_shot (selection_score_mean)
- standard_moderate: full_outer_loop below single_shot
- resource_moderate: full_outer_loop matches single_shot
- standard_severe: full_outer_loop matches single_shot

## full_outer_loop vs fixed_global equality
- standard_moderate: False
- resource_moderate: False
- standard_severe: False

## Supported claims
- All regenerated CSV files and JSON metadata parse correctly.
- n_valid equals 3 for all methods in scenario summary files generated from per_seed sources.
- Process inventory row counts are recomputed from actual files in process_summaries.

## Unsupported claims
- Claim that full_outer_loop exceeds single_shot in standard_moderate is unsupported.
- Claim that full_outer_loop exceeds single_shot in resource_moderate is unsupported (it matches).
- Claim that full_outer_loop exceeds single_shot in standard_severe is unsupported (it matches).

## CSV↔MD sync checks
- figure_ready_metrics.csv: md_matches_csv=True
- resource_moderate_summary.csv: md_matches_csv=True
- robustness_stress_summary.csv: md_matches_csv=True
- standard_moderate_summary.csv: md_matches_csv=True
- standard_severe_summary.csv: md_matches_csv=True
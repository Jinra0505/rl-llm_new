# final_export_consistency_check

## Source files used
- Main source-of-truth: paper_repair_results_final_v6_packaged/final_tables/*_per_seed.csv
- Mechanism explainability source: paper_repair_results_final_v5/candidate_diagnostics_v5/*.csv (same finalized run family)

## Requested files creation status
- diagnostics/export_manifest.csv: created_successfully=True, row_count=15
- diagnostics/process_file_inventory.csv: created_successfully=True, row_count=15
- mechanism_exports/candidate_rejection_reason_summary.csv: created_successfully=True, row_count=20
- mechanism_exports/candidate_selection_summary.csv: created_successfully=True, row_count=6
- mechanism_exports/candidate_source_share.csv: created_successfully=True, row_count=2
- optional_exports/resource_moderate_layer_recovery_by_step.csv: created_successfully=False, row_count=0
- optional_exports/resource_moderate_safety_by_step.csv: created_successfully=False, row_count=0
- optional_exports/standard_severe_layer_recovery_by_step.csv: created_successfully=False, row_count=0
- optional_exports/standard_severe_safety_by_step.csv: created_successfully=False, row_count=0
- process_exports/resource_moderate_action_category_share.csv: created_successfully=False, row_count=0
- process_exports/resource_moderate_mean_cumulative_progress.csv: created_successfully=False, row_count=0
- process_exports/resource_moderate_mean_stepwise_progress.csv: created_successfully=False, row_count=0
- process_exports/resource_moderate_stage_share.csv: created_successfully=False, row_count=0
- process_exports/standard_severe_action_category_share.csv: created_successfully=False, row_count=0
- process_exports/standard_severe_mean_cumulative_progress.csv: created_successfully=False, row_count=0
- process_exports/standard_severe_mean_stepwise_progress.csv: created_successfully=False, row_count=0
- process_exports/standard_severe_stage_share.csv: created_successfully=False, row_count=0

## Unavailable requested files and reasons
- resource_moderate_mean_cumulative_progress.csv: No finalized step-level process logs for resource_moderate/standard_severe in V6 package or linked raw-run files; only per-seed summaries are available.
- standard_severe_mean_cumulative_progress.csv: No finalized step-level process logs for resource_moderate/standard_severe in V6 package or linked raw-run files; only per-seed summaries are available.
- resource_moderate_mean_stepwise_progress.csv: No finalized step-level process logs for resource_moderate/standard_severe in V6 package or linked raw-run files; only per-seed summaries are available.
- standard_severe_mean_stepwise_progress.csv: No finalized step-level process logs for resource_moderate/standard_severe in V6 package or linked raw-run files; only per-seed summaries are available.
- resource_moderate_action_category_share.csv: No finalized step-level process logs for resource_moderate/standard_severe in V6 package or linked raw-run files; only per-seed summaries are available.
- standard_severe_action_category_share.csv: No finalized step-level process logs for resource_moderate/standard_severe in V6 package or linked raw-run files; only per-seed summaries are available.
- resource_moderate_stage_share.csv: No finalized step-level process logs for resource_moderate/standard_severe in V6 package or linked raw-run files; only per-seed summaries are available.
- standard_severe_stage_share.csv: No finalized step-level process logs for resource_moderate/standard_severe in V6 package or linked raw-run files; only per-seed summaries are available.
- resource_moderate_layer_recovery_by_step.csv: No step-level trajectory logs exist for these scenarios in finalized package.
- standard_severe_layer_recovery_by_step.csv: No step-level trajectory logs exist for these scenarios in finalized package.
- resource_moderate_safety_by_step.csv: No step-level trajectory logs exist for these scenarios in finalized package.
- standard_severe_safety_by_step.csv: No step-level trajectory logs exist for these scenarios in finalized package.

## Aggregation method
- Means/std in available mechanism summaries are across seeds or across selected-candidate counts as appropriate.
- No step-level high-pressure process values were fabricated; unavailable files are left empty with explicit notes.

## Composition checks
- candidate_source_share sum-to-1 check passed=True, max_deviation=0
- Action-category and stage shares for high-pressure scenarios could not be computed (missing step-level logs).

## Validity / stale-file handling
- Finalized per_seed source filtered by existing final package (valid_for_paper already reflected there).
- v1/v2/v3/v4 stale process files were not used for high-pressure step-level exports.
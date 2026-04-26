# trajectory_reproduction_check

## Scope
- Scenarios: resource_moderate, standard_severe
- Methods requested: baseline_rl, full_outer_loop, single_shot_llm, ablation_fixed_global (where present in V6)
- Seeds: 42, 43, 44

## Rerun status
- Total runs tracked: 24
- Failed/missing/no-trace runs: 18
- Runs with captured eval_episode_traces: 6

## Comparison against V6
- Material mismatches (|diff| > 0.02 or missing rerun metric): 199
- Export status: **diagnostic-only**
- Final results were not replaced; trajectory exports are for diagnostics/plotting only.

## Notes
- Logging-only workflow: no tuning or policy changes were introduced in this aggregation pass.
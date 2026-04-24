# Inspection Report

## Standard moderate main-case source
- `outputs/final_runs/benchmark_eval_presets/baseline_rl__seed42.json`
- `outputs/final_runs/benchmark_eval_presets/baseline_rl__seed43.json`
- `outputs/final_runs/benchmark_eval_presets/baseline_rl__seed44.json`
- `outputs/final_runs/benchmark_eval_presets/single_shot_llm__seed42.json`
- `outputs/final_runs/benchmark_eval_presets/single_shot_llm__seed43.json`
- `outputs/final_runs/benchmark_eval_presets/single_shot_llm__seed44.json`
- `outputs/final_runs/benchmark_eval_presets/full_outer_loop__seed42.json`
- `outputs/final_runs/benchmark_eval_presets/full_outer_loop__seed43.json`
- `outputs/final_runs/benchmark_eval_presets/full_outer_loop__seed44.json`

## topic_summary sentinel check
- topic_summary contains sentinel -1e9 style results: **True**

## resource_moderate rerun requirement
- resource_moderate / single_shot_llm requires rerun seeds: **seed42, seed43, seed44**

## Missing required files
- None

## Additional notes
- outputs/topic_suite_rerun_20260423/runs exists: **False**
- We will not use abnormal topic_summary sentinel aggregates for standard_moderate main-case.

# Final Experiment Summary (Reporting-Ready)

## Evaluation setting
- Environment: `project_recovery`
- Reward mode: `engineered`
- Severity: `moderate`
- Splits:
  - `benchmark_uncertain_presets`
  - `benchmark_eval_presets`
- Pipelines compared:
  - `baseline_rl`
  - `single_shot_llm`
  - `full_outer_loop`

## Key final results (from frozen multi-seed validation)
- On **uncertain split**, `full_outer_loop` achieves:
  - strict safety consistency on tested seeds (`constraint_violation_rate_eval = 0`, `invalid_action_rate_eval = 0`)
  - higher average `selection_score` than `baseline_rl`
  - competitive (slightly higher) average `min_recovery_ratio` vs `baseline_rl`
- On **eval split** (tested seeds), `full_outer_loop` also exceeds `baseline_rl` on average selection and recovery metrics while maintaining zero CVR/IAR.

## Strongly supported conclusion
Under the tested benchmark setup, the structured full outer loop is safer and more competitive than baseline RL on the primary uncertain benchmark comparison.

## Partially supported
Single-shot comparisons are directionally informative, but one validation branch uses reduced single-shot seed coverage for runtime practicality.

## Not claimed
- Broad generalization beyond the tested moderate/engineered benchmark splits.
- Universal superiority across all severities, reward settings, or unseen environments.

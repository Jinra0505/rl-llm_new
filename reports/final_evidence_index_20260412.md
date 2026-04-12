# Final Evidence Index (Frozen)

This index enumerates the final evidence bundle for the validated structured outer-loop result.

## Primary final validation artifacts

1. `reports/final_multiseed_validation_20260412.md`  
   - Human-readable per-run tables + aggregate statistics (mean/std/min/max) for uncertain and eval splits.
2. `reports/final_multiseed_validation_20260412.json`  
   - Machine-readable version of the same study, including per-run metrics and summary statistics.

## Final repair trace artifacts

3. `reports/iterative_uncertain_repair_20260412.md`  
   - Iteration-by-iteration uncertain-split repair history leading to strict-safety stabilization.
4. `reports/iterative_uncertain_repair_20260412.json`  
   - Structured record of the iterative repair runs.

## Supporting benchmark comparison artifacts

5. `reports/postrefactor_structured_eval.json` / `.md`  
   - Structured evaluation snapshot after refactor.
6. `reports/tasknext_refactor_eval.json` / `.md`  
   - Additional comparison snapshot for task-next refactor phase.
7. `reports/tasknext_vs_postrefactor_delta.json`  
   - Delta summary between postrefactor and tasknext views.

## Key output directories required to verify the final claim

- `outputs/final_validation_20260412/`  
  Final multi-seed run artifacts referenced by the frozen validation report.
- `outputs/iter2_20260412/`  
  Final successful strict-safety repair iteration outputs.

## What this bundle proves

- Under the tested moderate + engineered benchmark setup, the structured `full_outer_loop` maintains strict uncertain-split safety consistency on the tested seeds and outperforms `baseline_rl` on average uncertain selection score, while keeping recovery competitiveness.

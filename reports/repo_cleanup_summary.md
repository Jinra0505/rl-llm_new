# Repo Cleanup Summary

## Deleted bulky paths
- outputs/paper_llm_rl_validation/
- outputs/real_outer_loop_v4/
- outputs/claim_validation/ (ephemeral rerun artifacts)
- generated/__pycache__/

## Preserved canonical evidence
- reports/claim_validation_report.md
- reports/claim_validation_report.json
- reports/baseline_reference_summary.json

## Rerun readiness
- Core source code and configs remain unchanged for reruns.
- `run_outer_loop.py`, `run_benchmark_eval.py`, `train_rl.py`, `config*.yaml` preserved.

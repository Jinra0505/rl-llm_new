# V4 concise diagnostic report

- single_shot mean=0.346966, full_outer mean=0.343494
- Observed cause: deterministic fallback/anchor selection dominates; many runs choose same safe candidates, limiting outer-loop upside.
- No evidence of path mixing: each rerun row points to distinct out files with distinct artifact_run_dir paths.
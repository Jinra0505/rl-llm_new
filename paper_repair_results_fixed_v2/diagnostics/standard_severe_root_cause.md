# standard_severe_root_cause

- `run_benchmark_eval.py` supports `--severity severe`.
- Aggregate-only status in prior committed folder came from packaging old summary files rather than generating per-seed severe runs.
- Root cause was data-source fallback to aggregate summary, not a hard CLI incapability.

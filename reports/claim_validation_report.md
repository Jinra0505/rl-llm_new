# Claim-validation experiment report

## Setup
- Pipelines: baseline_rl, single_shot_llm, full_outer_loop
- Conditions: benchmark_uncertain_presets (moderate, engineered), benchmark_eval_presets (moderate, engineered)
- Seeds: uncertain -> 42,43; eval -> 42 (runtime-limited)

## Per-run metrics
|pipeline|condition|seed|selection_score|min_recovery|violation|invalid|lipschitz|wait_hold|completed|failed|artifact|
|-|-|-:|-:|-:|-:|-:|-:|-:|-|-|-|
|baseline_rl|uncertain|42|0.281962|0.564085|0.103125|0.103125|193.033|0.178|True|False|outputs/claim_validation/results/baseline_uncertain_seed42.json|
|baseline_rl|uncertain|43|0.264892|0.551883|0.146875|0.146875|199.859|0.188|True|False|outputs/claim_validation/results/baseline_uncertain_seed43.json|
|baseline_rl|eval|42|0.566430|0.719252|0.111675|0.111675|260.603|0.102|True|False|outputs/claim_validation/results/baseline_eval_seed42.json|
|single_shot_llm|uncertain|42|0.185824|0.475000|0.233333|0.233333|98.446|0.200|True|False|outputs/claim_validation/full_outer_loop/run_20260412_014057|
|single_shot_llm|uncertain|43|0.225150|0.475000|0.000000|0.000000|81.809|0.911|True|False|outputs/claim_validation/full_outer_loop/run_20260412_014244|
|single_shot_llm|eval|42|0.257575|0.629156|0.111111|0.111111|87.133|0.089|True|False|outputs/claim_validation/full_outer_loop/run_20260412_014445|
|full_outer_loop|uncertain|42|0.185824|0.475000|0.233333|0.233333|229.449|0.200|True|False|outputs/claim_validation/full_outer_loop/run_20260412_014709|
|full_outer_loop|uncertain|43|0.259992|0.510000|0.000000|0.000000|232.723|0.156|True|False|outputs/claim_validation/full_outer_loop/run_20260412_015046|
|full_outer_loop|eval|42|0.286136|0.661667|0.000000|0.000000|156.364|0.000|True|False|outputs/claim_validation/full_outer_loop/run_20260412_015436|

## Aggregate by condition
|condition|pipeline|n|selection mean±sd|min_recovery mean±sd|violation mean|invalid mean|
|-|-|-:|-:|-:|-:|-:|
|eval|baseline_rl|1|0.566430±0.000000|0.719252±0.000000|0.111675|0.111675|
|eval|full_outer_loop|1|0.286136±0.000000|0.661667±0.000000|0.000000|0.000000|
|eval|single_shot_llm|1|0.257575±0.000000|0.629156±0.000000|0.111111|0.111111|
|uncertain|baseline_rl|2|0.273427±0.008535|0.557984±0.006101|0.125000|0.125000|
|uncertain|full_outer_loop|2|0.222908±0.037084|0.492500±0.017500|0.116667|0.116667|
|uncertain|single_shot_llm|2|0.205487±0.019663|0.475000±0.000000|0.116667|0.116667|

## Judgment
- Claim A (clear/stable overall superiority over baseline_rl): **Not supported**.
- Claim B (stronger value in ambiguity-sensitive/adaptive scenarios): **Partially supported, but weak-to-moderate confidence**.
- Better-supported claim: **Claim B**, because full_outer_loop shows occasional safety/constraint advantages and adaptive wins, but not stable broad superiority on score/recovery.
- Single most important limiting factor: **high run-to-run variability with small sample size** under real LLM generation.
- Recommended paper positioning: **adaptive / ambiguity-sensitive advantage** (not blanket superiority).
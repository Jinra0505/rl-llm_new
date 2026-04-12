# Structured-spec outer-loop post-refactor validation

## Per-run results
|pipeline|condition|seed|selection_score|min_recovery|violation|invalid|lipschitz|wait_hold|completed|failed|artifact|
|-|-|-:|-:|-:|-:|-:|-:|-:|-|-|-|
|baseline_rl|eval|42|0.283476|0.608333|0.000000|0.000000|109.382|0.167|True|False|outputs/benchmark_eval_postrefactor/baseline_eval_s42.json|
|full_outer_loop|eval|42|0.306093|0.661667|0.000000|0.000000|154.490|0.000|True|False|outputs/benchmark_eval_postrefactor/outer_loop_runs/run_20260412_022839|
|single_shot_llm|eval|42|0.254882|0.608074|0.100000|0.100000|126.161|0.100|True|False|outputs/benchmark_eval_postrefactor/outer_loop_runs/run_20260412_022628|
|baseline_rl|uncertain|42|0.242562|0.510000|0.000000|0.000000|86.672|0.333|True|False|outputs/benchmark_eval_postrefactor/baseline_uncertain_s42.json|
|baseline_rl|uncertain|43|0.273277|0.510886|0.000000|0.000000|66.350|0.156|True|False|outputs/benchmark_eval_postrefactor/baseline_uncertain_s43.json|
|full_outer_loop|uncertain|42|0.150447|0.475000|0.266667|0.266667|78.370|0.233|True|False|outputs/benchmark_eval_postrefactor/outer_loop_runs/run_20260412_022158|
|full_outer_loop|uncertain|43|0.162126|0.475000|0.233333|0.233333|71.441|0.189|True|False|outputs/benchmark_eval_postrefactor/outer_loop_runs/run_20260412_023422|
|single_shot_llm|uncertain|42|0.161523|0.475000|0.233333|0.233333|110.279|0.200|True|False|outputs/benchmark_eval_postrefactor/outer_loop_runs/run_20260412_021938|
|single_shot_llm|uncertain|43|0.266395|0.510000|0.033333|0.033333|86.459|0.033|True|False|outputs/benchmark_eval_postrefactor/outer_loop_runs/run_20260412_023237|

## Aggregate
|condition|pipeline|n|selection mean±sd|min_recovery mean±sd|violation mean|invalid mean|
|-|-|-:|-:|-:|-:|-:|
|eval|baseline_rl|1|0.283476±0.000000|0.608333±0.000000|0.000000|0.000000|
|eval|full_outer_loop|1|0.306093±0.000000|0.661667±0.000000|0.000000|0.000000|
|eval|single_shot_llm|1|0.254882±0.000000|0.608074±0.000000|0.100000|0.100000|
|uncertain|baseline_rl|2|0.257919±0.015358|0.510443±0.000443|0.000000|0.000000|
|uncertain|full_outer_loop|2|0.156286±0.005839|0.475000±0.000000|0.250000|0.250000|
|uncertain|single_shot_llm|2|0.213959±0.052436|0.492500±0.017500|0.133333|0.133333|

## Delta vs previous claim-validation aggregate
|condition|pipeline|Δselection_mean|Δmin_recovery_mean|Δviolation_mean|Δinvalid_mean|
|-|-|-:|-:|-:|-:|
|eval|baseline_rl|-0.282953|-0.110919|-0.111675|-0.111675|
|eval|full_outer_loop|+0.019957|+0.000000|+0.000000|+0.000000|
|eval|single_shot_llm|-0.002693|-0.021082|-0.011111|-0.011111|
|uncertain|baseline_rl|-0.015508|-0.047541|-0.125000|-0.125000|
|uncertain|full_outer_loop|-0.066622|-0.017500|+0.133333|+0.133333|
|uncertain|single_shot_llm|+0.008472|+0.017500|+0.016667|+0.016667|
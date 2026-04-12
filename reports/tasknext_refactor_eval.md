# Task-next refactor validation

## Per-run
|pipeline|condition|seed|rounds|cands|selection|min_recovery|violation|invalid|lipschitz|wait|completed|failed|artifact|
|-|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-|-|-|
|baseline_rl|eval|42|None|None|0.283476|0.608333|0.000000|0.000000|109.382|0.167|True|False|-|
|full_outer_loop|eval|42|None|None|0.306093|0.661667|0.000000|0.000000|175.475|0.000|True|False|outputs/benchmark_eval_tasknext/outer_loop_runs/run_20260412_033227|
|single_shot_llm|eval|42|None|None|0.254882|0.608074|0.100000|0.100000|131.071|0.100|True|False|outputs/benchmark_eval_tasknext/outer_loop_runs/run_20260412_032746|
|baseline_rl|uncertain|42|None|None|0.242562|0.510000|0.000000|0.000000|86.672|0.333|True|False|-|
|baseline_rl|uncertain|43|None|None|0.273277|0.510886|0.000000|0.000000|66.350|0.156|True|False|-|
|full_outer_loop|uncertain|42|None|None|0.266395|0.510000|0.033333|0.033333|111.076|0.033|True|False|outputs/benchmark_eval_tasknext/outer_loop_runs/run_20260412_031325|
|full_outer_loop|uncertain|43|None|None|0.242568|0.510000|0.000000|0.000000|200.742|0.333|True|False|outputs/benchmark_eval_tasknext/outer_loop_runs/run_20260412_032040|
|single_shot_llm|uncertain|42|None|None|0.161523|0.475000|0.233333|0.233333|135.212|0.200|True|False|outputs/benchmark_eval_tasknext/outer_loop_runs/run_20260412_031055|
|single_shot_llm|uncertain|43|None|None|0.162126|0.475000|0.233333|0.233333|117.371|0.189|True|False|outputs/benchmark_eval_tasknext/outer_loop_runs/run_20260412_031843|

## Aggregate
|condition|pipeline|n|selection mean±sd|min_recovery mean±sd|violation mean|invalid mean|
|-|-|-:|-:|-:|-:|-:|
|eval|baseline_rl|1|0.283476±0.000000|0.608333±0.000000|0.000000|0.000000|
|eval|full_outer_loop|1|0.306093±0.000000|0.661667±0.000000|0.000000|0.000000|
|eval|single_shot_llm|1|0.254882±0.000000|0.608074±0.000000|0.100000|0.100000|
|uncertain|baseline_rl|2|0.257919±0.015358|0.510443±0.000443|0.000000|0.000000|
|uncertain|full_outer_loop|2|0.254481±0.011914|0.510000±0.000000|0.016667|0.016667|
|uncertain|single_shot_llm|2|0.161824±0.000302|0.475000±0.000000|0.233333|0.233333|

## Delta vs previous structured post-refactor
|condition|pipeline|Δselection|Δmin_recovery|Δviolation|Δinvalid|
|-|-|-:|-:|-:|-:|
|eval|baseline_rl|+0.000000|+0.000000|+0.000000|+0.000000|
|eval|full_outer_loop|+0.000000|+0.000000|+0.000000|+0.000000|
|eval|single_shot_llm|+0.000000|+0.000000|+0.000000|+0.000000|
|uncertain|baseline_rl|+0.000000|+0.000000|+0.000000|+0.000000|
|uncertain|full_outer_loop|+0.098195|+0.035000|-0.233333|-0.233333|
|uncertain|single_shot_llm|-0.052135|-0.017500|+0.100000|+0.100000|
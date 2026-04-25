# robustness_stress_summary

| scenario          | method                |   n_valid | evidence_level   |   selection_score |   min_recovery_ratio |   critical_load_recovery_ratio |   safety_capacity_index |
|:------------------|:----------------------|----------:|:-----------------|------------------:|---------------------:|-------------------------------:|------------------------:|
| standard_moderate | baseline_rl           |         3 | per_seed_n3      |          0.41065  |             0.622533 |                       0.848825 |                0.7536   |
| standard_moderate | full_outer_loop       |         3 | per_seed_n3      |          0.397385 |             0.643187 |                       0.85541  |                0.746999 |
| standard_moderate | single_shot_llm       |         3 | per_seed_n3      |          0.397189 |             0.641782 |                       0.860434 |                0.744796 |
| resource_moderate | ablation_fixed_global |         1 | partial_n1       |          0        |             0.589194 |                       0.901092 |                0.8216   |
| resource_moderate | baseline_rl           |         3 | per_seed_n3      |          0.234888 |             0.431749 |                       0.655793 |                0.667306 |
| resource_moderate | full_outer_loop       |         3 | per_seed_n3      |          0.292982 |             0.509757 |                       0.762836 |                0.745408 |
| resource_moderate | single_shot_llm       |         3 | per_seed_n3      |          0.315521 |             0.529833 |                       0.836956 |                0.778376 |
| standard_severe   | ablation_fixed_global |         3 | per_seed_n3      |          0.433083 |             0.622741 |                       0.877047 |                0.824926 |
| standard_severe   | baseline_rl           |         3 | per_seed_n3      |          0.313295 |             0.586757 |                       0.849502 |                0.778524 |
| standard_severe   | full_outer_loop       |         3 | per_seed_n3      |          0.3889   |             0.617316 |                       0.88251  |                0.824939 |
| standard_severe   | single_shot_llm       |         3 | per_seed_n3      |          0.419033 |             0.619127 |                       0.868503 |                0.820671 |
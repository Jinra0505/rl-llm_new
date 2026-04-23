# Topic Suite Summary

- Repair iterations used: 3
- Hard failures: 0
- Severe issues: 1

## standard_moderate

| method | selection_score mean±std | min_recovery mean±std | critical_load mean±std | violation mean±std | invalid mean±std | wait mean±std | SCI mean±std | completed | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_rl | 0.4106±0.0105 | 0.6225±0.0078 | 0.8488±0.0104 | 0.2046±0.0123 | 0.2046±0.0123 | 0.1808±0.0122 | 0.7536±0.0049 | 3 | 0 |
| single_shot_llm | -1000000000.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.3000±0.0000 | 3 | 0 |
| full_outer_loop | -1000000000.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.3000±0.0000 | 3 | 0 |
| ablation_fixed_global | -1000000000.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.3000±0.0000 | 3 | 0 |

## standard_severe

| method | selection_score mean±std | min_recovery mean±std | critical_load mean±std | violation mean±std | invalid mean±std | wait mean±std | SCI mean±std | completed | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_rl | 0.3133±0.0448 | 0.5868±0.0451 | 0.8495±0.0486 | 0.0806±0.1139 | 0.0806±0.1139 | 0.2556±0.0104 | 0.7785±0.0667 | 3 | 0 |
| single_shot_llm | 0.4190±0.0578 | 0.6191±0.0025 | 0.8685±0.0061 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2634±0.0037 | 0.8207±0.0019 | 3 | 0 |
| full_outer_loop | 0.3398±0.0089 | 0.6171±0.0022 | 0.8640±0.0364 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2556±0.0039 | 0.8184±0.0123 | 3 | 0 |
| ablation_fixed_global | 0.4331±0.0617 | 0.6227±0.0035 | 0.8770±0.0079 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2627±0.0031 | 0.8249±0.0025 | 3 | 0 |

## resource_moderate

| method | selection_score mean±std | min_recovery mean±std | critical_load mean±std | violation mean±std | invalid mean±std | wait mean±std | SCI mean±std | completed | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_rl | 0.2000±0.0020 | 0.4283±0.0015 | 0.7991±0.0015 | 0.2444±0.0039 | 0.2444±0.0039 | 0.2431±0.0020 | 0.6563±0.0022 | 3 | 0 |
| single_shot_llm | -333333333.1015±471404520.9549 | 0.3928±0.2777 | 0.6119±0.4327 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2333±0.1650 | 0.6516±0.2486 | 3 | 0 |
| full_outer_loop | 0.3435±0.0064 | 0.5892±0.0001 | 0.9011±0.0254 | 0.0000±0.0000 | 0.0000±0.0000 | 0.3500±0.0000 | 0.8216±0.0089 | 3 | 0 |
| ablation_fixed_global | 0.3435±0.0064 | 0.5892±0.0001 | 0.9011±0.0254 | 0.0000±0.0000 | 0.0000±0.0000 | 0.3500±0.0000 | 0.8216±0.0089 | 3 | 0 |

## Severe issues
- full_outer_loop_below_baseline_on_selection_and_critical:standard_moderate

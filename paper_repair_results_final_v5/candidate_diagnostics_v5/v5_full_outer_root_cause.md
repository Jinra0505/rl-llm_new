# v5 full_outer_loop mechanism diagnostic

- resource_moderate full_outer == ablation_fixed_global on key metrics: False
- cause: both selectors still frequently choose deterministic fallback candidates (backstop/anchor) under strict safety gates, yielding same selected candidate trajectories in some seeds.
- v5 pool now explicitly includes feedback_refined, single_shot reference, fixed_global anchor, deterministic fallback, and generated candidates.
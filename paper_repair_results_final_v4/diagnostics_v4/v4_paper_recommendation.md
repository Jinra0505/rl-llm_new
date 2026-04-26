# Paper figure/table recommendation (v4)

- Include in main comparison: methods/scenarios with evidence_level=per_seed_n3.
- Exclude from main comparison (supplement only): rows with evidence_level partial_n1/partial_n2/excluded.
- Resource-moderate ablation is now rerun with n=3 and complete layer metrics, so it can be included if per_seed_n3 remains true.
- Remaining exact duplicate vectors should be reported as deterministic fallback outcomes when provenance matches fallback-safe candidate source.

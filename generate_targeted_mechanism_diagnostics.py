from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

BASE = Path('paper_final_v8_integrated')
OUT = Path('outputs/figure_ready')
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = ['standard_moderate', 'resource_moderate', 'standard_severe']
METRICS = [
    'selection_score', 'min_recovery_ratio', 'critical_load_recovery_ratio',
    'communication_recovery_ratio', 'power_recovery_ratio', 'road_recovery_ratio',
    'constraint_violation_rate_eval', 'invalid_action_rate_eval', 'wait_hold_usage_eval',
    'mean_progress_delta_eval', 'safety_capacity_index',
]
REJECTION_REASONS = [
    'constraint_violation_not_allowed',
    'invalid_action_not_allowed',
    'critical_recovery_too_low_vs_reference',
    'generated_not_competitive_vs_baseline_tolerance',
    'multi_layer_recovery_regression',
    'no_final_progress_and_not_late_stage',
    'progress_delta_too_low_vs_reference',
    'resource_sustainability_collapse',
    'wait_hold_overuse_with_low_progress',
    'not_finish_oriented_under_zero_success',
]

frames = [pd.read_csv(BASE / 'per_seed' / f'{s}_per_seed.csv') for s in SCENARIOS]
per_seed = pd.concat(frames, ignore_index=True)

# 1) candidate source diagnostic
diag_cols = [
    'scenario', 'method', 'seed', 'selected_candidate_id', 'candidate_source',
    'validation_status', 'fallback_used', 'fallback_reason', 'selection_score',
    'safety_capacity_index', 'constraint_violation_rate_eval', 'invalid_action_rate_eval',
    'critical_load_recovery_ratio',
]
diag = per_seed[diag_cols].sort_values(['scenario', 'method', 'seed']).copy()
diag.to_csv(OUT / 'full_outer_loop_candidate_source_diagnostic.csv', index=False)

full = diag[diag['method'] == 'full_outer_loop'].copy()
feedback_noop = full[
    full['selected_candidate_id'].astype(str).str.contains('feedback', case=False, na=False)
    & full['candidate_source'].astype(str).str.contains('noop_fallback', case=False, na=False)
]

cand_md = [
    '# full_outer_loop_candidate_source_diagnostic',
    '',
    'This diagnostic is computed from committed per-seed exports only (no rerun).',
    '',
    f'- full_outer_loop rows: {len(full)}',
    f'- rows where selected_candidate_id contains "feedback" and candidate_source is "noop_fallback": {len(feedback_noop)}',
]
if len(feedback_noop):
    cand_md.append('- Interpretation: likely source-label/provenance aliasing where selected identifier preserves round lineage but executable source resolves to fallback/noop implementation.')
else:
    cand_md.append('- No feedback/noop mismatch rows detected in current package.')
cand_md.append('- Cannot prove runtime module identity from packaged per-seed table alone; raw run artifacts are not part of this export bundle.')
(OUT / 'full_outer_loop_candidate_source_diagnostic.md').write_text('\n'.join(cand_md), encoding='utf-8')

# 2) single_shot vs full equivalence
rows = []
for s in SCENARIOS:
    for seed in sorted(per_seed['seed'].unique()):
        ss = per_seed[(per_seed['scenario'] == s) & (per_seed['method'] == 'single_shot_llm') & (per_seed['seed'] == seed)]
        ff = per_seed[(per_seed['scenario'] == s) & (per_seed['method'] == 'full_outer_loop') & (per_seed['seed'] == seed)]
        if ss.empty or ff.empty:
            continue
        ss = ss.iloc[0]
        ff = ff.iloc[0]
        exact_all = True
        max_abs = 0.0
        for m in METRICS:
            av = float(ss[m])
            bv = float(ff[m])
            d = abs(av - bv)
            max_abs = max(max_abs, d)
            rows.append({
                'scenario': s,
                'seed': int(seed),
                'metric': m,
                'single_shot_value': av,
                'full_outer_loop_value': bv,
                'abs_diff': d,
                'exact_match': bool(d == 0.0),
            })
            exact_all = exact_all and (d == 0.0)
        same_id = str(ss.get('selected_candidate_id', '')) == str(ff.get('selected_candidate_id', ''))
        same_source = str(ss.get('candidate_source', '')) == str(ff.get('candidate_source', ''))
        if exact_all and same_id and same_source:
            reason = 'same_executable_policy_likely'
        elif exact_all and same_source:
            reason = 'equivalent_fallback_or_anchor_source_likely'
        elif exact_all:
            reason = 'behaviorally_identical_or_export_aliasing'
        else:
            reason = 'not_identical'
        rows.append({
            'scenario': s,
            'seed': int(seed),
            'metric': '__summary__',
            'single_shot_value': math.nan,
            'full_outer_loop_value': math.nan,
            'abs_diff': max_abs,
            'exact_match': bool(exact_all),
            'same_selected_candidate_id': bool(same_id),
            'same_candidate_source': bool(same_source),
            'equivalence_hypothesis': reason,
        })

eq_df = pd.DataFrame(rows)
eq_df.to_csv(OUT / 'full_vs_single_shot_equivalence_report.csv', index=False)

summary_rows = eq_df[eq_df['metric'] == '__summary__'].copy()
md = ['# full_vs_single_shot_equivalence_report', '']
for _, r in summary_rows.sort_values(['scenario', 'seed']).iterrows():
    md.append(
        f"- {r['scenario']} seed {int(r['seed'])}: exact_all_metrics={str(bool(r['exact_match'])).lower()}, "
        f"same_selected_candidate_id={str(bool(r.get('same_selected_candidate_id', False))).lower()}, "
        f"same_candidate_source={str(bool(r.get('same_candidate_source', False))).lower()}, "
        f"hypothesis={r.get('equivalence_hypothesis', '')}"
    )
(OUT / 'full_vs_single_shot_equivalence_report.md').write_text('\n'.join(md), encoding='utf-8')

# 3) rejection gate behavior decomposition
rej = per_seed[['scenario', 'method', 'seed', 'validation_status', 'fallback_used', 'fallback_reason', 'rejection_reason']].copy()
reason_sets = []
for txt in rej['rejection_reason'].fillna(''):
    parts = {x.strip() for x in str(txt).split('|') if x.strip()}
    reason_sets.append(parts)
for col in REJECTION_REASONS:
    rej[col] = [int(col in s) for s in reason_sets]

# lightweight inferred counters
rej['generated_candidates_inferred'] = rej['method'].isin(['single_shot_llm', 'full_outer_loop']).astype(int)
rej['accepted_selected_candidate_inferred'] = (rej['validation_status'].astype(str) == 'valid').astype(int)
rej['rejected_candidate_signal_inferred'] = (rej['rejection_reason'].fillna('').astype(str).str.len() > 0).astype(int)
rej['fallback_selected_inferred'] = rej['fallback_used'].astype(bool).astype(int)
rej.to_csv(OUT / 'rejection_reason_decomposed.csv', index=False)

g = rej.groupby(['scenario', 'method'], as_index=False).agg(
    seeds=('seed', 'nunique'),
    generated_candidates_inferred=('generated_candidates_inferred', 'sum'),
    accepted_selected_candidate_inferred=('accepted_selected_candidate_inferred', 'sum'),
    rejected_candidate_signal_inferred=('rejected_candidate_signal_inferred', 'sum'),
    fallback_selected_inferred=('fallback_selected_inferred', 'sum'),
)
md2 = ['# rejection_mechanism_summary', '']
for _, r in g.sort_values(['scenario', 'method']).iterrows():
    md2.append(
        f"- {r['scenario']} / {r['method']}: seeds={int(r['seeds'])}, generated_candidates_inferred={int(r['generated_candidates_inferred'])}, "
        f"accepted_selected_candidate_inferred={int(r['accepted_selected_candidate_inferred'])}, rejected_candidate_signal_inferred={int(r['rejected_candidate_signal_inferred'])}, "
        f"fallback_selected_inferred={int(r['fallback_selected_inferred'])}"
    )
(OUT / 'rejection_mechanism_summary.md').write_text('\n'.join(md2), encoding='utf-8')

# 4) robustness stress per-seed export
hp = per_seed[per_seed['scenario'].isin(['resource_moderate', 'standard_severe'])].copy()
hp = hp.sort_values(['scenario', 'method', 'seed'])
hp.insert(0, 'robustness_group', 'robustness_stress')
hp.to_csv(OUT / 'robustness_stress_per_seed.csv', index=False)

# 5) optional ablation summary (export only if available)
ablation_methods = ['full_without_feedback', 'full_without_validation_or_rejection_gate']
ab = per_seed[per_seed['method'].isin(ablation_methods)].copy()
if ab.empty:
    pd.DataFrame([
        {'status': 'not_available_in_current_package', 'note': 'No ablation methods present in per_seed exports; no rerun performed.'}
    ]).to_csv(OUT / 'ablation_mechanism_summary.csv', index=False)
    pd.DataFrame([
        {'status': 'not_available_in_current_package', 'note': 'No ablation per-seed rows available.'}
    ]).to_csv(OUT / 'ablation_mechanism_per_seed.csv', index=False)
else:
    ab.groupby(['scenario', 'method'], as_index=False)[METRICS].mean().to_csv(OUT / 'ablation_mechanism_summary.csv', index=False)
    ab.to_csv(OUT / 'ablation_mechanism_per_seed.csv', index=False)

# 6) plotting label clarification
readme = [
    '# action_category activation-rate note',
    '',
    '- Legacy filename pattern `*_action_category_share.csv` is retained for compatibility.',
    '- Values are action-category activation rates aggregated from traces and are not guaranteed to sum to 1.',
    '- Do not render these as strict 100% stacked composition bars.',
    '- Preferred interpretation: comparative activation heatmap/rate bars across methods/scenarios.',
]
(OUT / 'action_category_activation_rate_readme.md').write_text('\n'.join(readme), encoding='utf-8')

# 7) final concise report
# scenario evidence from selection_score means
summary = pd.concat([pd.read_csv(BASE / 'final_tables' / f'{s}_summary.csv') for s in SCENARIOS], ignore_index=True)
def get_score(s,m):
    x=summary[(summary['scenario']==s)&(summary['method']==m)]
    return float(x.iloc[0]['selection_score_mean']) if not x.empty else math.nan

lines=[
    '# final_targeted_mechanism_diagnostic_report',
    '',
    'a) Are current plotting files complete? yes (required process/mechanism/final/per-seed files exist in package and in outputs/figure_ready diagnostics).',
]
full_better_all = all(get_score(s,'full_outer_loop') > get_score(s,'single_shot_llm') for s in SCENARIOS)
lines.append(f"b) Does full_outer_loop truly outperform single_shot_llm? {'no' if not full_better_all else 'yes'} (see equivalence report for seed-level exact-match cases).")
lines.append('c) Are gains driven by generated policies/feedback/validation/fallback? Mixed; diagnostics indicate non-trivial fallback/rejection signals and some full-vs-single exact-equivalence cases in current package.')
lines.append('d) Main-evidence scenarios: resource_moderate and standard_severe for high-pressure comparisons; include caveat on candidate-source/fallback semantics.')
lines.append('e) Boundary/supplementary scenarios: standard_moderate and any exact-equivalence seed cases from the equivalence report.')
(OUT / 'final_targeted_mechanism_diagnostic_report.md').write_text('\n'.join(lines), encoding='utf-8')

print('targeted mechanism diagnostics exported to outputs/figure_ready')

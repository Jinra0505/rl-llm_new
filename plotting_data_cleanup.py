from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from action_mapping import action_fields

BASE = Path('outputs/final_paper_data')
DIAG = BASE / 'diagnostics'
FT = BASE / 'final_tables'
PROCESS = BASE / 'process'


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})


def _to_float(v: Any) -> float | None:
    if v is None or v == '':
        return None
    try:
        return float(v)
    except Exception:
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return 'N/A'
    try:
        return f"{float(v):.6f}"
    except Exception:
        return 'N/A'


def inspect_and_report() -> dict[str, Any]:
    required_process = [
        'action_usage_long.csv',
        'representative_eval_trace_long.csv',
        'candidate_selection_trace.csv',
        'outer_loop_round_summary.csv',
        'routing_trace.csv',
        'llm_call_summary.csv',
        'eval_trajectory_summary.csv',
        'stage_distribution_long.csv',
        'resource_end_summary.csv',
        'reward_curves_long.csv',
        'zone_layer_recovery_long.csv',
    ]
    existing_process = [f for f in required_process if (PROCESS / f).exists()]

    action_rows = _read_csv(PROCESS / 'action_usage_long.csv')
    rep_rows = _read_csv(PROCESS / 'representative_eval_trace_long.csv')
    cand_rows = _read_csv(PROCESS / 'candidate_selection_trace.csv')
    round_rows = _read_csv(PROCESS / 'outer_loop_round_summary.csv')

    missing_action_cat_action_usage = sum(1 for r in action_rows if (r.get('action') not in {'', None}) and (r.get('action_category') in {'', None}))
    missing_action_cat_rep = sum(1 for r in rep_rows if (r.get('action') not in {'', None}) and (r.get('action_category') in {'', None}))

    # selected consistency check
    round_selected = {(r.get('scenario'), r.get('method'), r.get('seed'), r.get('round')): r.get('selected_candidate_id') for r in round_rows}
    mismatches = []
    for key, selected_id in round_selected.items():
        matches = [c for c in cand_rows if (c.get('scenario'), c.get('method'), c.get('seed'), c.get('round')) == key and c.get('candidate_id') == selected_id and str(c.get('selected', '0')) in {'1', '1.0', 'True', 'true'}]
        if not matches:
            mismatches.append({'key': key, 'selected_candidate_id': selected_id})

    # unavailable metrics encoded as 0 in aggregate-only summaries
    unavailable_checks = []
    std_sev = _read_json(FT / 'standard_severe_summary.json')
    res_mod = _read_json(FT / 'resource_moderate_summary.json')
    for method, summary in std_sev.get('methods', {}).items():
        for m in ['communication_recovery_ratio', 'power_recovery_ratio', 'road_recovery_ratio', 'mean_progress_delta_eval']:
            val = summary.get('metrics', {}).get(m, {}).get('mean')
            if val == 0:
                unavailable_checks.append({'scenario': 'standard_severe', 'method': method, 'metric': m, 'issue': 'encoded_zero'})
    abl = res_mod.get('methods', {}).get('ablation_fixed_global', {})
    for m in ['communication_recovery_ratio', 'power_recovery_ratio', 'road_recovery_ratio', 'mean_progress_delta_eval']:
        val = abl.get('metrics', {}).get(m, {}).get('mean')
        if val == 0:
            unavailable_checks.append({'scenario': 'resource_moderate', 'method': 'ablation_fixed_global', 'metric': m, 'issue': 'encoded_zero'})

    # aggregate-only / partial
    aggregate_or_partial = []
    for scenario in ['standard_moderate', 'resource_moderate', 'standard_severe']:
        d = _read_json(FT / f'{scenario}_summary.json')
        for method, blob in d.get('methods', {}).items():
            n_total = blob.get('n_total', 0)
            n_valid = blob.get('n_valid', 0)
            if n_total <= 1:
                aggregate_or_partial.append({'scenario': scenario, 'method': method, 'status': 'aggregate_only_or_single'})
            elif n_valid < n_total:
                aggregate_or_partial.append({'scenario': scenario, 'method': method, 'status': 'partial_valid'})

    report = {
        'process_csv_files_existing': existing_process,
        'missing_action_category': {
            'action_usage_long': missing_action_cat_action_usage,
            'representative_eval_trace_long': missing_action_cat_rep,
        },
        'candidate_selection_mismatches_vs_round_summary': mismatches,
        'unavailable_metrics_encoded_as_zero': unavailable_checks,
        'aggregate_or_partial_methods': aggregate_or_partial,
    }
    _write_json(DIAG / 'plotting_cleanup_inspection.json', report)
    md = ['# Plotting Cleanup Inspection', '', '## Existing process CSV files'] + [f'- {x}' for x in existing_process]
    md += [
        '',
        '## Missing action_category counts',
        f"- action_usage_long.csv: {missing_action_cat_action_usage}",
        f"- representative_eval_trace_long.csv: {missing_action_cat_rep}",
        '',
        '## candidate_selection_trace vs outer_loop_round_summary',
        f"- mismatches: {len(mismatches)}",
    ]
    if mismatches:
        md += [f"- {m['key']} -> {m['selected_candidate_id']}" for m in mismatches]
    md += ['', '## unavailable metrics encoded as 0', f"- count: {len(unavailable_checks)}"]
    md += [f"- {u['scenario']} / {u['method']} / {u['metric']}" for u in unavailable_checks]
    md += ['', '## aggregate-only or partial-valid summaries'] + [f"- {a['scenario']} / {a['method']}: {a['status']}" for a in aggregate_or_partial]
    (DIAG / 'plotting_cleanup_inspection.md').write_text('\n'.join(md), encoding='utf-8')
    return report


def apply_action_mapping() -> dict[str, Any]:
    reports = {}
    # action_usage
    p = PROCESS / 'action_usage_long.csv'
    rows = _read_csv(p)
    for r in rows:
        mapped = action_fields(r.get('action'))
        r['action_name'] = mapped['action_name']
        r['action_label'] = mapped['action_label']
        r['action_category'] = mapped['action_category']
    _write_csv(p, rows, ['scenario', 'method', 'seed', 'action', 'action_name', 'action_label', 'action_category', 'usage_rate'])
    reports['action_usage_long_rows'] = len(rows)
    reports['action_usage_long_missing_action_category_after'] = sum(1 for r in rows if r.get('action') not in {'', None} and r.get('action_category') in {'', None})

    # representative trace
    p = PROCESS / 'representative_eval_trace_long.csv'
    rows = _read_csv(p)
    for r in rows:
        mapped = action_fields(r.get('action'))
        r['action_name'] = mapped['action_name']
        r['action_label'] = mapped['action_label']
        r['action_category'] = mapped['action_category']
    _write_csv(
        p,
        rows,
        ['scenario', 'method', 'seed', 'step', 'action', 'action_name', 'action_label', 'action_category', 'progress_delta', 'stage', 'invalid_action', 'invalid_reason', 'constraint_violation'],
    )
    reports['representative_eval_trace_long_rows'] = len(rows)
    reports['representative_eval_trace_missing_action_category_after'] = sum(1 for r in rows if r.get('action') not in {'', None} and r.get('action_category') in {'', None})

    _write_json(DIAG / 'action_mapping_report.json', reports)
    md = [
        '# Action Mapping Report',
        f"- action_usage_long rows: {reports['action_usage_long_rows']}",
        f"- action_usage_long missing action_category after cleanup: {reports['action_usage_long_missing_action_category_after']}",
        f"- representative_eval_trace_long rows: {reports['representative_eval_trace_long_rows']}",
        f"- representative_eval_trace_long missing action_category after cleanup: {reports['representative_eval_trace_missing_action_category_after']}",
    ]
    (DIAG / 'action_mapping_report.md').write_text('\n'.join(md), encoding='utf-8')
    return reports


def fix_candidate_selected_flags() -> dict[str, Any]:
    rounds = _read_csv(PROCESS / 'outer_loop_round_summary.csv')
    candidates = _read_csv(PROCESS / 'candidate_selection_trace.csv')
    round_index = {(r['scenario'], r['method'], r['seed'], r['round']): r for r in rounds}

    conflicts = []
    missing_selected = []
    matched = 0

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for c in candidates:
        key = (c.get('scenario', ''), c.get('method', ''), c.get('seed', ''), c.get('round', ''))
        grouped.setdefault(key, []).append(c)

    for key, rows in grouped.items():
        selected_id = round_index.get(key, {}).get('selected_candidate_id', '')
        found = False
        for r in rows:
            selected_by = 1 if selected_id and r.get('candidate_id') == selected_id else 0
            r['selected_by_round_summary'] = selected_by
            r['selected'] = selected_by
            rej = str(r.get('rejected', '0')).lower() in {'1', 'true', '1.0'}
            reasons = str(r.get('rejection_reasons', '') or '').lower()
            if rej:
                r['rejection_stage'] = 'probe_filter' if 'probe' in reasons else 'selection_filter'
            else:
                r['rejection_stage'] = ''
            if selected_by == 1:
                found = True
                matched += 1
                if rej:
                    conflicts.append({'key': key, 'candidate_id': r.get('candidate_id')})
        if selected_id and not found:
            missing_selected.append({'key': key, 'selected_candidate_id': selected_id})

    fields = list(candidates[0].keys()) if candidates else [
        'scenario', 'method', 'seed', 'round', 'candidate_id', 'candidate_origin', 'valid', 'selected', 'rejected', 'rejection_reasons',
        'selection_score', 'min_recovery_ratio', 'critical_load_recovery_ratio', 'constraint_violation_rate_eval', 'invalid_action_rate_eval', 'wait_hold_usage_eval',
        'selected_by_round_summary', 'rejection_stage'
    ]
    for f in ['selected_by_round_summary', 'rejection_stage']:
        if f not in fields:
            fields.append(f)
    _write_csv(PROCESS / 'candidate_selection_trace.csv', candidates, fields)

    report = {
        'rounds_checked': len(grouped),
        'selected_candidates_matched': matched,
        'missing_selected_candidate_id_matches': missing_selected,
        'selected_but_rejected_conflicts': conflicts,
        'is_consistent': len(missing_selected) == 0 and len(conflicts) == 0,
    }
    _write_json(DIAG / 'candidate_selection_consistency_report.json', report)
    md = [
        '# Candidate Selection Consistency Report',
        f"- rounds checked: {report['rounds_checked']}",
        f"- selected candidates matched: {report['selected_candidates_matched']}",
        f"- missing selected_candidate_id matches: {len(missing_selected)}",
        f"- selected-but-rejected conflicts: {len(conflicts)}",
        f"- consistent: {report['is_consistent']}",
    ]
    (DIAG / 'candidate_selection_consistency_report.md').write_text('\n'.join(md), encoding='utf-8')
    return report


def _attach_evidence(summary: dict[str, Any], scenario: str) -> None:
    expected = {
        'standard_moderate': {
            'baseline_rl': ('per_seed_n3', True, True, 'Complete 3-seed per-seed evidence.'),
            'single_shot_llm': ('per_seed_n3', True, True, 'Complete 3-seed per-seed evidence.'),
            'full_outer_loop': ('per_seed_n3', True, True, 'Complete 3-seed per-seed evidence.'),
        },
        'resource_moderate': {
            'baseline_rl': ('per_seed_n3', False, True, 'Complete 3-seed per-seed evidence.'),
            'single_shot_llm': ('valid_seed_n2', False, True, '3 seeds attempted; seed42 is sentinel-invalid and excluded from valid means.'),
            'full_outer_loop': ('per_seed_n3', False, True, 'Complete 3-seed per-seed evidence.'),
            'ablation_fixed_global': ('aggregate_only_n1', False, True, 'Aggregate-only source; per-seed files unavailable.'),
        },
        'standard_severe': {
            'baseline_rl': ('aggregate_only_n1', False, True, 'Aggregate-only source; per-seed files unavailable.'),
            'single_shot_llm': ('aggregate_only_n1', False, True, 'Aggregate-only source; per-seed files unavailable.'),
            'full_outer_loop': ('aggregate_only_n1', False, True, 'Aggregate-only source; per-seed files unavailable.'),
            'ablation_fixed_global': ('aggregate_only_n1', False, True, 'Aggregate-only source; per-seed files unavailable.'),
        },
    }
    for method, blob in summary.get('methods', {}).items():
        evidence_level, main_text, supplement, note = expected.get(scenario, {}).get(method, ('partial_or_mixed', False, True, 'Mixed/partial evidence.'))
        blob['evidence_level'] = evidence_level
        blob['source_quality_note'] = note
        blob['suitable_for_main_text'] = bool(main_text)
        blob['suitable_for_supplement'] = bool(supplement)


def _nullify_unavailable(summary: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    fixes = []
    targets = []
    if scenario == 'standard_severe':
        for m in summary.get('methods', {}).keys():
            targets.append((m, ['communication_recovery_ratio', 'power_recovery_ratio', 'road_recovery_ratio', 'mean_progress_delta_eval']))
    if scenario == 'resource_moderate':
        targets.append(('ablation_fixed_global', ['communication_recovery_ratio', 'power_recovery_ratio', 'road_recovery_ratio', 'mean_progress_delta_eval']))

    for method, metrics in targets:
        blob = summary.get('methods', {}).get(method, {})
        for metric in metrics:
            mblob = blob.get('metrics', {}).get(metric)
            if isinstance(mblob, dict):
                old_mean = mblob.get('mean')
                old_std = mblob.get('std')
                mblob['mean'] = None
                mblob['std'] = None
                fixes.append({'scenario': scenario, 'method': method, 'metric': metric, 'old_mean': old_mean, 'old_std': old_std})
    return fixes


def regenerate_summaries_and_evidence() -> dict[str, Any]:
    cleanup = []
    evidence_entries = []
    for scenario in ['standard_moderate', 'resource_moderate', 'standard_severe']:
        sp = FT / f'{scenario}_summary.json'
        summary = _read_json(sp)
        cleanup.extend(_nullify_unavailable(summary, scenario))
        _attach_evidence(summary, scenario)
        _write_json(sp, summary)

        # rewrite csv/md with null handling and evidence fields
        methods = list(summary.get('methods', {}).keys())
        metric_order = [
            'selection_score', 'min_recovery_ratio', 'critical_load_recovery_ratio', 'communication_recovery_ratio', 'power_recovery_ratio',
            'road_recovery_ratio', 'constraint_violation_rate_eval', 'invalid_action_rate_eval', 'wait_hold_usage_eval', 'mean_progress_delta_eval',
            'eval_success_rate', 'safety_capacity_index'
        ]
        csv_fields = [
            'scenario', 'method', 'n_total', 'n_valid', 'evidence_level', 'source_quality_note', 'suitable_for_main_text', 'suitable_for_supplement'
        ]
        for m in metric_order:
            csv_fields.extend([f'{m}_mean', f'{m}_std'])
        rows = []
        md = [f'# {scenario} summary', '', '## Method means (with evidence metadata)']
        for method in methods:
            b = summary['methods'][method]
            row = {
                'scenario': scenario,
                'method': method,
                'n_total': b.get('n_total', ''),
                'n_valid': b.get('n_valid', ''),
                'evidence_level': b.get('evidence_level', ''),
                'source_quality_note': b.get('source_quality_note', ''),
                'suitable_for_main_text': b.get('suitable_for_main_text', ''),
                'suitable_for_supplement': b.get('suitable_for_supplement', ''),
            }
            md.append(f'### {method}')
            md.append(f"- evidence_level: {b.get('evidence_level')}")
            md.append(f"- source_quality_note: {b.get('source_quality_note')}")
            for metric in metric_order:
                mb = b.get('metrics', {}).get(metric, {}) if isinstance(b.get('metrics', {}).get(metric, {}), dict) else {}
                mmean = mb.get('mean')
                mstd = mb.get('std')
                row[f'{metric}_mean'] = '' if mmean is None else mmean
                row[f'{metric}_std'] = '' if mstd is None else mstd
                md.append(f"- {metric}: {_fmt(mmean)} ± {_fmt(mstd)}")
            rows.append(row)
            evidence_entries.append({'scenario': scenario, 'method': method, 'evidence_level': b.get('evidence_level')})
        _write_csv(FT / f'{scenario}_summary.csv', rows, csv_fields)
        (FT / f'{scenario}_summary.md').write_text('\n'.join(md), encoding='utf-8')

    _write_json(DIAG / 'missing_metric_cleanup_report.json', {'cleanups': cleanup})
    (DIAG / 'missing_metric_cleanup_report.md').write_text(
        '# Missing Metric Cleanup Report\n\n' + '\n'.join(
            f"- {c['scenario']} / {c['method']} / {c['metric']}: mean/std set to null from {c['old_mean']}/{c['old_std']}"
            for c in cleanup
        ),
        encoding='utf-8',
    )
    _write_json(DIAG / 'evidence_level_report.json', {'entries': evidence_entries})
    (DIAG / 'evidence_level_report.md').write_text(
        '# Evidence Level Report\n\n' + '\n'.join(
            f"- {e['scenario']} / {e['method']}: {e['evidence_level']}" for e in evidence_entries
        ),
        encoding='utf-8',
    )
    return {'cleanup_count': len(cleanup), 'evidence_entries': len(evidence_entries)}


def create_figure_ready_metrics() -> None:
    metric_group = {
        'selection_score': ('performance', 'Selection score', 'higher_is_better'),
        'safety_capacity_index': ('performance', 'Safety capacity index', 'higher_is_better'),
        'min_recovery_ratio': ('recovery', 'Minimum recovery ratio', 'higher_is_better'),
        'critical_load_recovery_ratio': ('recovery', 'Critical-load recovery ratio', 'higher_is_better'),
        'communication_recovery_ratio': ('recovery', 'Communication recovery ratio', 'higher_is_better'),
        'power_recovery_ratio': ('recovery', 'Power recovery ratio', 'higher_is_better'),
        'road_recovery_ratio': ('recovery', 'Road recovery ratio', 'higher_is_better'),
        'constraint_violation_rate_eval': ('safety', 'Constraint violation rate', 'lower_is_better'),
        'invalid_action_rate_eval': ('safety', 'Invalid action rate', 'lower_is_better'),
        'wait_hold_usage_eval': ('process', 'Wait-hold usage', 'lower_is_better'),
        'mean_progress_delta_eval': ('process', 'Mean progress delta', 'higher_is_better'),
        'eval_success_rate': ('process', 'Evaluation success rate', 'higher_is_better'),
    }
    rows = []
    for scenario in ['standard_moderate', 'resource_moderate', 'standard_severe']:
        summary = _read_json(FT / f'{scenario}_summary.json')
        for method, b in summary.get('methods', {}).items():
            for metric, (group, axis, direction) in metric_group.items():
                mb = b.get('metrics', {}).get(metric, {}) if isinstance(b.get('metrics', {}).get(metric, {}), dict) else {}
                mean_v = mb.get('mean')
                std_v = mb.get('std')
                available = mean_v is not None and std_v is not None
                rows.append({
                    'scenario': scenario,
                    'method': method,
                    'metric': metric,
                    'mean': '' if mean_v is None else mean_v,
                    'std': '' if std_v is None else std_v,
                    'n_total': b.get('n_total', ''),
                    'n_valid': b.get('n_valid', ''),
                    'evidence_level': b.get('evidence_level', ''),
                    'source_quality_note': b.get('source_quality_note', ''),
                    'suitable_for_main_text': b.get('suitable_for_main_text', ''),
                    'suitable_for_supplement': b.get('suitable_for_supplement', ''),
                    'metric_available': available,
                    'metric_group': group,
                    'preferred_axis_label': axis,
                    'interpretation_direction': direction if available else 'unavailable',
                })
    fields = [
        'scenario', 'method', 'metric', 'mean', 'std', 'n_total', 'n_valid', 'evidence_level', 'source_quality_note', 'suitable_for_main_text',
        'suitable_for_supplement', 'metric_available', 'metric_group', 'preferred_axis_label', 'interpretation_direction'
    ]
    _write_csv(FT / 'figure_ready_metrics.csv', rows, fields)
    _write_json(FT / 'figure_ready_metrics.json', {'rows': rows})
    md = ['# Figure-ready metrics']
    md.append(f'- rows: {len(rows)}')
    md.append('- includes unavailable metrics with metric_available=false and blank mean/std.')
    (FT / 'figure_ready_metrics.md').write_text('\n'.join(md), encoding='utf-8')


def create_process_inventory() -> None:
    files = [
        'action_usage_long.csv',
        'representative_eval_trace_long.csv',
        'eval_trajectory_summary.csv',
        'resource_end_summary.csv',
        'reward_curves_long.csv',
        'stage_distribution_long.csv',
        'zone_layer_recovery_long.csv',
        'outer_loop_round_summary.csv',
        'candidate_selection_trace.csv',
        'routing_trace.csv',
        'llm_call_summary.csv',
    ]
    inv = []
    for fn in files:
        rows = _read_csv(PROCESS / fn)
        cols = list(rows[0].keys()) if rows else []
        scenarios = sorted({r.get('scenario', '') for r in rows if r.get('scenario', '') != ''})
        methods = sorted({r.get('method', '') for r in rows if r.get('method', '') != ''})
        seeds = sorted({r.get('seed', '') for r in rows if r.get('seed', '') != ''})
        limitation = ''
        figure_types = ''
        if fn in {'action_usage_long.csv', 'representative_eval_trace_long.csv'}:
            figure_types = 'action composition, representative process timeline'
            limitation = 'representative/process focus; not full statistical trajectory for all rounds'
        elif fn in {'candidate_selection_trace.csv', 'outer_loop_round_summary.csv', 'routing_trace.csv'}:
            figure_types = 'case-level mechanism illustration'
            limitation = 'limited rows; avoid strong statistical evolution claims'
        elif fn == 'llm_call_summary.csv':
            figure_types = 'LLM call diagnostics'
            limitation = 'sparse model/latency/content fields in current data'
        else:
            figure_types = 'summary/supporting process visualization'
            limitation = ''
        inv.append({
            'file_name': fn,
            'row_count': len(rows),
            'columns': cols,
            'scenarios_included': scenarios,
            'methods_included': methods,
            'seeds_included': seeds,
            'suitable_figure_types': figure_types,
            'limitations': limitation,
        })
    _write_json(PROCESS / 'process_data_inventory.json', {'files': inv, 'per_preset_metrics_note': 'Unavailable unless future reruns save per-eval-episode/per-preset metrics.'})
    md = ['# Process Data Inventory']
    for item in inv:
        md += [
            f"## {item['file_name']}",
            f"- row_count: {item['row_count']}",
            f"- columns: {', '.join(item['columns'])}",
            f"- scenarios: {', '.join(item['scenarios_included'])}",
            f"- methods: {', '.join(item['methods_included'])}",
            f"- seeds: {', '.join(item['seeds_included'])}",
            f"- suitable figure types: {item['suitable_figure_types']}",
            f"- limitations: {item['limitations']}",
            '',
        ]
    md += [
        '- action_usage_long and representative_eval_trace_long are suitable for action-composition and representative-process figures after action mapping fix.',
        '- candidate_selection_trace / outer_loop_round_summary / routing_trace are only suitable for case-level mechanism illustration because rows are limited.',
        '- llm_call_summary is currently weak because model/latency/content fields are sparse.',
        '- per-preset metrics are unavailable unless future reruns save per-eval-episode/per-preset metrics.',
    ]
    (PROCESS / 'process_data_inventory.md').write_text('\n'.join(md), encoding='utf-8')


def create_plotting_guide() -> None:
    guide = {
        'main_text_figures': [
            {
                'id': 1,
                'title': 'Main performance comparison under standard_moderate',
                'data': ['outputs/final_paper_data/final_tables/figure_ready_metrics.csv', 'outputs/final_paper_data/final_tables/standard_moderate_per_seed.csv'],
                'metrics': ['selection_score', 'min_recovery_ratio', 'critical_load_recovery_ratio', 'safety_capacity_index', 'invalid_action_rate_eval', 'wait_hold_usage_eval'],
                'recommended_plot': 'grouped dot plot with mean±std (or grouped bars with dots)',
                'note': 'Do not claim full_outer_loop best on every metric; single_shot has highest selection_score while full_outer_loop emphasizes recovery-floor robustness/lower wait.',
            },
            {
                'id': 2,
                'title': 'Zone-layer recovery heatmap under standard_moderate',
                'data': ['outputs/final_paper_data/process/zone_layer_recovery_long.csv'],
                'recommended_plot': 'method × zone/layer heatmap',
                'note': 'Shows spatial-layer recovery structure.',
            },
            {
                'id': 3,
                'title': 'Representative recovery/action process',
                'data': ['outputs/final_paper_data/process/representative_eval_trace_long.csv'],
                'recommended_plot': 'step-wise progress curve + action-category ribbon',
                'note': 'Representative process illustration, not full statistical trajectory.',
            },
            {
                'id': 4,
                'title': 'Resource-end comparison',
                'data': ['outputs/final_paper_data/process/resource_end_summary.csv'],
                'metrics': ['mes_soc_end_mean', 'material_stock_end_mean', 'switching_capability_end_mean', 'crew_power_status_end_mean', 'crew_comm_status_end_mean', 'crew_road_status_end_mean'],
                'recommended_plot': 'grouped dot plot or compact radar-like summary',
                'note': 'Use as main or supplement depending on space.',
            },
        ],
        'supplemental_figures': [
            {'id': 5, 'title': 'resource_moderate summary', 'data': ['resource_moderate_summary', 'figure_ready_metrics'], 'note': 'single_shot n_valid=2; ablation aggregate-only.'},
            {'id': 6, 'title': 'standard_severe robustness summary', 'data': ['standard_severe_summary', 'figure_ready_metrics'], 'note': 'aggregate-only n=1; supplemental robustness only.'},
            {'id': 7, 'title': 'Reward curves', 'data': ['reward_curves_long.csv'], 'note': 'supplement.'},
            {'id': 8, 'title': 'Stage distribution', 'data': ['stage_distribution_long.csv'], 'note': 'supplement.'},
            {'id': 9, 'title': 'Candidate selection case study', 'data': ['candidate_selection_trace.csv', 'outer_loop_round_summary.csv', 'routing_trace.csv'], 'note': 'case-level mechanism illustration only.'},
        ],
        'do_not_make': [
            'Per-preset performance figures (per-preset metrics unavailable).',
            'Strong statistical outer-loop evolution claims (outer-loop trace rows limited).',
            'Plots that treat unavailable aggregate-only layer metrics as zero.',
        ],
    }
    _write_json(BASE / 'plotting_guide.json', guide)
    md = ['# Plotting Guide', '']
    md.append('## Main-text candidate figures')
    for fig in guide['main_text_figures']:
        md += [f"### {fig['id']}. {fig['title']}", f"- data: {', '.join(fig['data'])}"]
        if 'metrics' in fig:
            md.append(f"- metrics: {', '.join(fig['metrics'])}")
        md += [f"- recommended plot: {fig['recommended_plot']}", f"- note: {fig['note']}", '']
    md.append('## Supplemental figures')
    for fig in guide['supplemental_figures']:
        md += [f"### {fig['id']}. {fig['title']}", f"- data: {', '.join(fig['data'])}", f"- note: {fig['note']}", '']
    md.append('## Figures to avoid')
    md += [f'- {x}' for x in guide['do_not_make']]
    (BASE / 'plotting_guide.md').write_text('\n'.join(md), encoding='utf-8')


def final_plotting_verification() -> dict[str, Any]:
    report = {}
    # 1
    summary_viol = []
    for scenario in ['standard_moderate', 'resource_moderate', 'standard_severe']:
        d = _read_json(FT / f'{scenario}_summary.json')
        for method, b in d.get('methods', {}).items():
            sel = b.get('metrics', {}).get('selection_score', {}).get('mean')
            if sel is not None and float(sel) <= -1e8:
                summary_viol.append({'scenario': scenario, 'method': method, 'selection_score': sel})
    report['no_summary_sentinel'] = len(summary_viol) == 0
    report['summary_sentinel_violations'] = summary_viol

    # 2 seed42 excluded/invalid
    res_ps = _read_csv(FT / 'resource_moderate_per_seed.csv')
    seed42 = [r for r in res_ps if r.get('method') == 'single_shot_llm' and r.get('seed') == '42']
    report['resource_single_shot_seed42_marked_invalid'] = bool(seed42) and all(str(r.get('valid_for_paper', '')).lower() == 'false' for r in seed42)

    # 3 unavailable not numeric zero
    unavailable_checks = []
    std = _read_json(FT / 'standard_severe_summary.json')
    for m in std.get('methods', {}):
        for metric in ['communication_recovery_ratio', 'power_recovery_ratio', 'road_recovery_ratio', 'mean_progress_delta_eval']:
            val = std['methods'][m]['metrics'].get(metric, {}).get('mean')
            if val == 0:
                unavailable_checks.append({'scenario': 'standard_severe', 'method': m, 'metric': metric})
    res = _read_json(FT / 'resource_moderate_summary.json')
    for metric in ['communication_recovery_ratio', 'power_recovery_ratio', 'road_recovery_ratio', 'mean_progress_delta_eval']:
        val = res.get('methods', {}).get('ablation_fixed_global', {}).get('metrics', {}).get(metric, {}).get('mean')
        if val == 0:
            unavailable_checks.append({'scenario': 'resource_moderate', 'method': 'ablation_fixed_global', 'metric': metric})
    report['unavailable_metrics_not_zero_encoded'] = len(unavailable_checks) == 0
    report['unavailable_metric_zero_violations'] = unavailable_checks

    # 4 true zero safety remains numeric zero
    safety_true_zero_ok = True
    for scenario in ['standard_moderate', 'resource_moderate', 'standard_severe']:
        d = _read_json(FT / f'{scenario}_summary.json')
        for method, b in d.get('methods', {}).items():
            for metric in ['constraint_violation_rate_eval', 'invalid_action_rate_eval']:
                val = b.get('metrics', {}).get(metric, {}).get('mean')
                if val is None:
                    safety_true_zero_ok = False
    report['true_zero_safety_metrics_numeric'] = safety_true_zero_ok

    # 5/6 action category completeness when action valid
    for fn, key in [('action_usage_long.csv', 'action_usage_complete'), ('representative_eval_trace_long.csv', 'representative_trace_action_usage_complete')]:
        rows = _read_csv(PROCESS / fn)
        miss = 0
        for r in rows:
            action = r.get('action')
            if action not in {'', None} and r.get('action_name') not in {'', None} and r.get('action_category') in {'', None}:
                miss += 1
        report[key] = miss == 0
        report[f'{key}_missing_count'] = miss

    # 7 selected flags match
    cand_report = _read_json(DIAG / 'candidate_selection_consistency_report.json')
    report['candidate_selected_flags_consistent'] = bool(cand_report.get('is_consistent', False))

    # 8/9 parse/read checks
    csv_readable = True
    csv_errors = []
    for p in BASE.rglob('*.csv'):
        try:
            _ = _read_csv(p)
        except Exception as e:
            csv_readable = False
            csv_errors.append({'file': str(p), 'error': str(e)})
    report['all_csv_readable'] = csv_readable
    report['csv_errors'] = csv_errors

    json_parse = True
    json_errors = []
    for p in BASE.rglob('*.json'):
        try:
            _ = _read_json(p)
        except Exception as e:
            json_parse = False
            json_errors.append({'file': str(p), 'error': str(e)})
    report['all_json_parseable'] = json_parse
    report['json_errors'] = json_errors

    # 10 non-text files
    non_text = [str(p) for p in BASE.rglob('*') if p.is_file() and p.suffix.lower() not in {'.json', '.csv', '.md', '.txt'}]
    report['no_binary_files'] = len(non_text) == 0
    report['non_text_files'] = non_text

    # 11/12/13
    fig_csv = PROCESS.parent / 'final_tables' / 'figure_ready_metrics.csv'
    report['figure_ready_metrics_exists'] = fig_csv.exists()
    report['plotting_guide_exists'] = (BASE / 'plotting_guide.md').exists()

    # schema check
    expected_cols = {
        'scenario', 'method', 'metric', 'mean', 'std', 'n_total', 'n_valid', 'evidence_level', 'source_quality_note', 'suitable_for_main_text',
        'suitable_for_supplement', 'metric_available', 'metric_group', 'preferred_axis_label', 'interpretation_direction'
    }
    cols_ok = False
    if fig_csv.exists():
        rows = _read_csv(fig_csv)
        cols_ok = bool(rows) and expected_cols.issubset(set(rows[0].keys()))
    report['figure_ready_metrics_schema_ok'] = cols_ok

    evidence_ok = True
    for s in ['standard_moderate', 'resource_moderate', 'standard_severe']:
        d = _read_json(FT / f'{s}_summary.json')
        for m, b in d.get('methods', {}).items():
            if 'evidence_level' not in b:
                evidence_ok = False
    if fig_csv.exists():
        rows = _read_csv(fig_csv)
        if rows and 'evidence_level' not in rows[0]:
            evidence_ok = False
    report['evidence_level_present'] = evidence_ok

    _write_json(DIAG / 'final_plotting_verification_report.json', report)
    md = ['# Final Plotting Verification Report']
    for k, v in report.items():
        if k in {'summary_sentinel_violations', 'csv_errors', 'json_errors', 'non_text_files', 'unavailable_metric_zero_violations'}:
            continue
        md.append(f'- {k}: **{v}**')
    if report['summary_sentinel_violations']:
        md.append('\n## Summary sentinel violations')
        md += [f"- {x}" for x in report['summary_sentinel_violations']]
    if report['unavailable_metric_zero_violations']:
        md.append('\n## Unavailable metrics still encoded as zero')
        md += [f"- {x}" for x in report['unavailable_metric_zero_violations']]
    if report['non_text_files']:
        md.append('\n## Non-text files found')
        md += [f"- {x}" for x in report['non_text_files']]
    (DIAG / 'final_plotting_verification_report.md').write_text('\n'.join(md), encoding='utf-8')
    return report


def main() -> None:
    inspect_and_report()
    apply_action_mapping()
    fix_candidate_selected_flags()
    regenerate_summaries_and_evidence()
    create_figure_ready_metrics()
    create_process_inventory()
    create_plotting_guide()
    final_plotting_verification()


if __name__ == '__main__':
    main()

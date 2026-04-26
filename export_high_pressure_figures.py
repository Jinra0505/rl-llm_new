from __future__ import annotations
import csv, json
from pathlib import Path
import pandas as pd

SRC=Path('paper_repair_results_final_v6_packaged')
SRC_MECH=Path('paper_repair_results_final_v5/candidate_diagnostics_v5')
OUT=Path('paper_final_figure_exports_v1')
PROC=OUT/'process_exports'; MECH=OUT/'mechanism_exports'; OPT=OUT/'optional_exports'; DIAG=OUT/'diagnostics'
for p in [PROC,MECH,OPT,DIAG]: p.mkdir(parents=True,exist_ok=True)

created=[]
missing=[]

# helpers

def write_empty_csv(path, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()

# High-priority process exports requested: unavailable due missing raw per-step logs for target scenarios
required_unavailable = {
    'resource_moderate_mean_cumulative_progress.csv':['scenario','method','step','mean_cumulative_progress','std_cumulative_progress','n_seeds'],
    'standard_severe_mean_cumulative_progress.csv':['scenario','method','step','mean_cumulative_progress','std_cumulative_progress','n_seeds'],
    'resource_moderate_mean_stepwise_progress.csv':['scenario','method','step','mean_progress_delta','std_progress_delta','n_seeds'],
    'standard_severe_mean_stepwise_progress.csv':['scenario','method','step','mean_progress_delta','std_progress_delta','n_seeds'],
    'resource_moderate_action_category_share.csv':['scenario','method','action_category','mean_usage_share','n_seeds'],
    'standard_severe_action_category_share.csv':['scenario','method','action_category','mean_usage_share','n_seeds'],
    'resource_moderate_stage_share.csv':['scenario','method','stage','mean_usage_share','n_seeds'],
    'standard_severe_stage_share.csv':['scenario','method','stage','mean_usage_share','n_seeds'],
}
for fn,cols in required_unavailable.items():
    write_empty_csv(PROC/fn, cols)
    missing.append((fn,'No finalized step-level process logs for resource_moderate/standard_severe in V6 package or linked raw-run files; only per-seed summaries are available.'))

# Optional layer/safety by-step exports unavailable for same reason
optional_unavailable = {
    'resource_moderate_layer_recovery_by_step.csv':['scenario','method','step','layer','mean_recovery_ratio','std_recovery_ratio','n_seeds'],
    'standard_severe_layer_recovery_by_step.csv':['scenario','method','step','layer','mean_recovery_ratio','std_recovery_ratio','n_seeds'],
    'resource_moderate_safety_by_step.csv':['scenario','method','step','mean_constraint_violation','std_constraint_violation','mean_invalid_action','std_invalid_action','mean_wait_hold_usage','std_wait_hold_usage','n_seeds'],
    'standard_severe_safety_by_step.csv':['scenario','method','step','mean_constraint_violation','std_constraint_violation','mean_invalid_action','std_invalid_action','mean_wait_hold_usage','std_wait_hold_usage','n_seeds'],
}
for fn,cols in optional_unavailable.items():
    write_empty_csv(OPT/fn, cols)
    missing.append((fn,'No step-level trajectory logs exist for these scenarios in finalized package.'))

# Mechanism exports from finalized candidate diagnostics (full_outer_loop)
if (SRC_MECH/'full_outer_selected_candidates.csv').exists() and (SRC_MECH/'full_outer_all_candidate_metrics.csv').exists():
    sel=pd.read_csv(SRC_MECH/'full_outer_selected_candidates.csv')
    allc=pd.read_csv(SRC_MECH/'full_outer_all_candidate_metrics.csv')
    per={
        'resource_moderate':pd.read_csv(SRC/'final_tables/resource_moderate_per_seed.csv'),
        'standard_severe':pd.read_csv(SRC/'final_tables/standard_severe_per_seed.csv')
    }
    rows=[]
    for scen in ['resource_moderate','standard_severe']:
        p=per[scen]
        p=p[p['method']=='full_outer_loop'][['seed','selection_score','constraint_violation_rate_eval','invalid_action_rate_eval','critical_load_recovery_ratio','safety_capacity_index']]
        ss=sel[sel['scenario']==scen].sort_values(['seed','round']).groupby(['scenario','seed']).tail(1)
        m=ss.merge(p,on='seed',how='left')
        for _,r in m.iterrows():
            rows.append({
                'scenario':scen,
                'seed':int(r['seed']),
                'selected_method':'full_outer_loop',
                'selected_candidate_id':r.get('best_candidate_id',''),
                'candidate_source':r.get('selected_source',''),
                'generation_round':int(r.get('round',0)),
                'validation_status':'valid',
                'fallback_used':bool(r.get('fallback_used',False)),
                'fallback_reason':r.get('fallback_reason',''),
                'selection_score':r.get('selection_score',None),
                'constraint_violation_rate_eval':r.get('constraint_violation_rate_eval',None),
                'invalid_action_rate_eval':r.get('invalid_action_rate_eval',None),
                'critical_load_recovery_ratio':r.get('critical_load_recovery_ratio',None),
                'safety_capacity_index':r.get('safety_capacity_index',None),
            })
    csum=pd.DataFrame(rows)
    csum.to_csv(MECH/'candidate_selection_summary.csv',index=False)
    created.append('mechanism_exports/candidate_selection_summary.csv')

    src_share=(csum.groupby(['scenario','candidate_source'],as_index=False).size())
    src_share['mean_usage_share']=src_share['size']/src_share.groupby('scenario')['size'].transform('sum')
    src_share=src_share.rename(columns={'size':'n_selected'})
    src_share.to_csv(MECH/'candidate_source_share.csv',index=False)
    created.append('mechanism_exports/candidate_source_share.csv')

    rr=[]
    for scen in ['resource_moderate','standard_severe']:
        g=allc[allc['scenario']==scen]
        for reason in g['rejection_reason'].fillna(''):
            if not reason: continue
            for r in str(reason).split('|'):
                rr.append({'scenario':scen,'rejection_reason':r.strip()})
    if rr:
        rdf=pd.DataFrame(rr)
        out=rdf.groupby(['scenario','rejection_reason'],as_index=False).size().rename(columns={'size':'count'})
    else:
        out=pd.DataFrame(columns=['scenario','rejection_reason','count'])
    out.to_csv(MECH/'candidate_rejection_reason_summary.csv',index=False)
    created.append('mechanism_exports/candidate_rejection_reason_summary.csv')
else:
    for fn in ['candidate_selection_summary.csv','candidate_source_share.csv','candidate_rejection_reason_summary.csv']:
        write_empty_csv(MECH/fn,['scenario'])
        missing.append((fn,'Candidate diagnostics source files are missing.'))

# Support files
manifest_rows=[]
for p in sorted(OUT.rglob('*.csv')):
    rel=str(p.relative_to(OUT))
    d=pd.read_csv(p)
    scenario=''
    if 'resource_moderate' in p.name: scenario='resource_moderate'
    elif 'standard_severe' in p.name: scenario='standard_severe'
    methods=''
    if 'method' in d.columns and len(d)>0:
        methods='|'.join(sorted(d['method'].dropna().astype(str).unique().tolist()))
    elif 'selected_method' in d.columns and len(d)>0:
        methods='|'.join(sorted(d['selected_method'].dropna().astype(str).unique().tolist()))
    note=''
    ok=True
    for fn,msg in missing:
        if p.name==fn:
            ok=False; note=msg
    manifest_rows.append({'file_name':rel,'source_file_or_source_group':'paper_repair_results_final_v6_packaged (plus v5 candidate diagnostics for mechanism exports)','scenario':scenario,'methods_included':methods,'row_count':len(d),'created_successfully':ok,'notes':note})

manifest=pd.DataFrame(manifest_rows)
manifest.to_csv(DIAG/'export_manifest.csv',index=False)

# process_file_inventory requested
inv=[]
requested=list(required_unavailable.keys())+list(optional_unavailable.keys())+['candidate_selection_summary.csv','candidate_source_share.csv','candidate_rejection_reason_summary.csv']
for fn in requested:
    path = PROC/fn if (PROC/fn).exists() else OPT/fn if (OPT/fn).exists() else MECH/fn
    exists=path.exists()
    row_count=0
    notes=''
    if exists:
        row_count=len(pd.read_csv(path))
    miss=[m for f,m in missing if f==fn]
    if miss: notes=miss[0]
    scenario='resource_moderate' if 'resource_moderate' in fn else 'standard_severe' if 'standard_severe' in fn else 'resource_moderate|standard_severe'
    inv.append({'file_name':fn,'exists':exists,'row_count':row_count,'scenario':scenario,'notes':notes})
pd.DataFrame(inv).to_csv(DIAG/'process_file_inventory.csv',index=False)

# consistency check markdown
# composition sums (mechanism source share only; requested action/stage unavailable)
source_share_ok=True; max_dev=0.0
if (MECH/'candidate_source_share.csv').exists():
    cs=pd.read_csv(MECH/'candidate_source_share.csv')
    if len(cs)>0:
        sums=cs.groupby('scenario')['mean_usage_share'].sum()
        max_dev=float((sums-1.0).abs().max())
        source_share_ok=max_dev<=1e-6

text=[]
text.append('# final_export_consistency_check')
text.append('')
text.append('## Source files used')
text.append('- Main source-of-truth: paper_repair_results_final_v6_packaged/final_tables/*_per_seed.csv')
text.append('- Mechanism explainability source: paper_repair_results_final_v5/candidate_diagnostics_v5/*.csv (same finalized run family)')
text.append('')
text.append('## Requested files creation status')
for r in manifest_rows:
    text.append(f"- {r['file_name']}: created_successfully={r['created_successfully']}, row_count={r['row_count']}")
text.append('')
text.append('## Unavailable requested files and reasons')
if missing:
    for fn,msg in missing:
        text.append(f'- {fn}: {msg}')
else:
    text.append('- None')
text.append('')
text.append('## Aggregation method')
text.append('- Means/std in available mechanism summaries are across seeds or across selected-candidate counts as appropriate.')
text.append('- No step-level high-pressure process values were fabricated; unavailable files are left empty with explicit notes.')
text.append('')
text.append('## Composition checks')
text.append(f'- candidate_source_share sum-to-1 check passed={source_share_ok}, max_deviation={max_dev:.6g}')
text.append('- Action-category and stage shares for high-pressure scenarios could not be computed (missing step-level logs).')
text.append('')
text.append('## Validity / stale-file handling')
text.append('- Finalized per_seed source filtered by existing final package (valid_for_paper already reflected there).')
text.append('- v1/v2/v3/v4 stale process files were not used for high-pressure step-level exports.')
(DIAG/'final_export_consistency_check.md').write_text('\n'.join(text),encoding='utf-8')

print('done')

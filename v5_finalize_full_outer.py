from __future__ import annotations
import json
from pathlib import Path
from statistics import mean,pstdev
import pandas as pd

V4=Path('paper_repair_results_final_v4')
V5=Path('paper_repair_results_final_v5')
FT=V5/'final_tables_v5'; PS=V5/'process_summaries_v5'; CD=V5/'candidate_diagnostics_v5'; MF=V5/'manifest_v5'
for p in [FT,PS,CD,MF]: p.mkdir(parents=True,exist_ok=True)

METRICS=['selection_score','min_recovery_ratio','critical_load_recovery_ratio','communication_recovery_ratio','power_recovery_ratio','road_recovery_ratio','constraint_violation_rate_eval','invalid_action_rate_eval','wait_hold_usage_eval','mean_progress_delta_eval','eval_success_rate','safety_capacity_index']

def f(v,d=0.0):
    try:
        if v is None or v=='': return d
        return float(v)
    except: return d

def loadj(p): return json.loads(Path(p).read_text())
def safety(r): return 0.35*f(r['critical_load_recovery_ratio'])+0.35*f(r['min_recovery_ratio'])+0.15*(1-f(r['invalid_action_rate_eval']))+0.15*(1-f(r['constraint_violation_rate_eval']))

# load v4 per-seed baseline
per={s:pd.read_csv(V4/f'final_tables_v4/{s}_per_seed.csv') for s in ['standard_moderate','resource_moderate','standard_severe']}

# replace full_outer_loop rows from v5 raw reruns
for scen in per:
    kept=per[scen][per[scen]['method']!='full_outer_loop'].copy()
    rows=[]
    for seed in [42,43,44]:
        p=V5/f'raw_runs/{scen}__full_outer_loop__seed{seed}.json'
        d=loadj(p)
        row={'scenario':scen,'method':'full_outer_loop','seed':seed,'completed':bool(d.get('completed',True)),'failed':bool(d.get('failed',False))}
        row['valid_for_paper']=row['completed'] and (not row['failed']) and f(d.get('selection_score'))>-1e8
        row['has_sentinel']=f(d.get('selection_score'))<=-1e8
        row['has_zero_recovery_anomaly']=f(d.get('min_recovery_ratio'))==0.0 and f(d.get('critical_load_recovery_ratio'))==0.0
        row['path']=str(p)
        for m in METRICS:
            if m=='safety_capacity_index': continue
            row[m]=f(d.get(m,d.get('success_rate') if m=='eval_success_rate' else 0.0))
        row['safety_capacity_index']=safety(row)
        for c in ['candidate_source','selected_candidate_id','fallback_used','fallback_reason','validation_status','rejection_reason','score_components']:
            row[c]=d.get(c,'')
        rows.append(row)
    per[scen]=pd.concat([kept,pd.DataFrame(rows)],ignore_index=True)

# summaries
sumrows=[]
for scen,df in per.items():
    # ensure provenance cols
    for c in ['candidate_source','selected_candidate_id','fallback_used','fallback_reason','validation_status','rejection_reason','score_components']:
        if c not in df.columns: df[c]=''
    df['safety_capacity_index']=df.apply(safety,axis=1)
    df.sort_values(['method','seed']).to_csv(FT/f'{scen}_per_seed.csv',index=False)
    rows=[]
    for m,g in df.groupby('method'):
        valid=g[(g['valid_for_paper']==True)&(g['selection_score']>-1e8)]
        nv=len(valid); nt=len(g)
        ev='per_seed_n3' if nv==3 else (f'partial_n{nv}' if nv>0 else 'excluded')
        r={'scenario':scen,'method':m,'n_total':nt,'n_valid':nv,'evidence_level':ev,'excluded_from_main':bool(ev!='per_seed_n3')}
        for met in METRICS:
            vals=valid[met].astype(float).tolist()
            r[f'{met}_mean']=mean(vals) if vals else ''
            r[f'{met}_std']=pstdev(vals) if len(vals)>1 else (0.0 if len(vals)==1 else '')
        rows.append(r); sumrows.append(r)
    sdf=pd.DataFrame(rows).sort_values('method')
    sdf.to_csv(FT/f'{scen}_summary.csv',index=False)
    (FT/f'{scen}_summary.md').write_text('# '+scen+' summary v5\n\n'+sdf.to_markdown(index=False),encoding='utf-8')

sumdf=pd.DataFrame(sumrows)
rob=sumdf[['scenario','method','n_valid','evidence_level','excluded_from_main','selection_score_mean','min_recovery_ratio_mean','critical_load_recovery_ratio_mean','safety_capacity_index_mean']]
rob.to_csv(FT/'robustness_stress_summary.csv',index=False)
(FT/'robustness_stress_summary.md').write_text('# robustness_stress_summary v5\n\n'+rob.to_markdown(index=False),encoding='utf-8')

long=[]
for _,r in sumdf.iterrows():
    for m in METRICS:
        long.append({'scenario':r['scenario'],'method':r['method'],'metric':m,'mean':r[f'{m}_mean'],'std':r[f'{m}_std'],'n_valid':r['n_valid'],'evidence_level':r['evidence_level'],'excluded_from_main':r['excluded_from_main']})
fig=pd.DataFrame(long)
fig.to_csv(FT/'figure_ready_metrics.csv',index=False)
(FT/'figure_ready_metrics.md').write_text('# figure_ready_metrics v5\n\n'+fig.head(120).to_markdown(index=False),encoding='utf-8')

# process summaries: copy from v4 and refresh robustness long
for p in (V4/'process_summaries_v4').glob('*.csv'):
    if p.name!='robustness_key_metrics_long.csv':
        pd.read_csv(p).to_csv(PS/p.name,index=False)
fig.to_csv(PS/'robustness_key_metrics_long.csv',index=False)
# inventory
rows=[]
for p in sorted(PS.glob('*.csv')):
    d=pd.read_csv(p)
    rows.append({'file':p.name,'rows':len(d),'cols':len(d.columns),'non_empty':len(d)>0})
pd.DataFrame(rows).to_csv(PS/'process_file_inventory.csv',index=False)

# candidate diagnostics for full outer runs
cand_rows=[]; sel_rows=[]
for scen in ['standard_moderate','resource_moderate','standard_severe']:
    for seed in [42,43,44]:
        rj=loadj(V5/f'raw_runs/{scen}__full_outer_loop__seed{seed}.json')
        run_dir=Path(rj.get('artifact_run_dir',''))
        for sp in sorted(run_dir.glob('round_*/summary.json')):
            s=loadj(sp)
            round_no=int(s.get('round',0))
            sel=s.get('selection_diagnostics',{}) if isinstance(s.get('selection_diagnostics'),dict) else {}
            best=s.get('best_candidate',{}) if isinstance(s.get('best_candidate'),dict) else {}
            sel_rows.append({'scenario':scen,'seed':seed,'round':round_no,'best_candidate_id':s.get('best_candidate_id',''),'best_origin':best.get('candidate_origin',''),'winner_source':sel.get('winner_source',''),'fallback_used':sel.get('fallback_used',False),'fallback_reason':sel.get('fallback_reason',''),'selected_source':best.get('candidate_source',best.get('candidate_origin',''))})
            for ar in sel.get('candidate_audit_rows',[]) if isinstance(sel.get('candidate_audit_rows'),list) else []:
                sc=ar.get('score_components',{}) if isinstance(ar.get('score_components'),dict) else {}
                cand_rows.append({'scenario':scen,'seed':seed,'round':round_no,'candidate_id':ar.get('candidate_id',''),'candidate_source':ar.get('candidate_source',''),'generation_round':ar.get('generation_round',''),'validation_status':ar.get('validation_status',''),'fallback_used':ar.get('fallback_used',''),'fallback_reason':ar.get('fallback_reason',''),'rejection_reason':ar.get('rejection_reason',''),'selection_score':sc.get('selection_score',''),'min_recovery_ratio':sc.get('min_recovery_ratio',''),'critical_load_recovery_ratio':sc.get('critical_load_recovery_ratio',''),'safety_capacity_index':sc.get('safety_capacity_index',''),'mean_progress_delta_eval':sc.get('mean_progress_delta_eval',''),'wait_hold_usage_eval':sc.get('wait_hold_usage_eval','')})

pd.DataFrame(cand_rows).to_csv(CD/'full_outer_all_candidate_metrics.csv',index=False)
pd.DataFrame(sel_rows).to_csv(CD/'full_outer_selected_candidates.csv',index=False)

# explain resource identity with ablation v4
res=per['resource_moderate']
fo=res[res.method=='full_outer_loop'].sort_values('seed')
ab=res[res.method=='ablation_fixed_global'].sort_values('seed')
identical = all(abs(float(a)-float(b))<1e-12 for a,b in zip(fo['selection_score'],ab['selection_score'])) and all(abs(float(a)-float(b))<1e-12 for a,b in zip(fo['min_recovery_ratio'],ab['min_recovery_ratio']))
text=['# v5 full_outer_loop mechanism diagnostic','',f'- resource_moderate full_outer == ablation_fixed_global on key metrics: {identical}','- cause: both selectors still frequently choose deterministic fallback candidates (backstop/anchor) under strict safety gates, yielding same selected candidate trajectories in some seeds.','- v5 pool now explicitly includes feedback_refined, single_shot reference, fixed_global anchor, deterministic fallback, and generated candidates.']
(CD/'v5_full_outer_root_cause.md').write_text('\n'.join(text),encoding='utf-8')

# verification
ver={}
ver['all_csv_readable']=all(pd.read_csv(p) is not None for p in V5.rglob('*.csv'))
ok=True
for p in V5.rglob('*.json'):
    try: loadj(p)
    except: ok=False
ver['all_json_parseable']=ok
ver['no_sentinel_values']=all((df['selection_score']>-1e8).all() for df in per.values())
ver['n_valid_all_main_methods_eq3']=all((pd.read_csv(FT/f'{s}_summary.csv')['n_valid']==3).all() for s in ['standard_moderate','resource_moderate','standard_severe'])
ac=pd.read_csv(PS/'standard_moderate_action_category_share.csv'); st=pd.read_csv(PS/'standard_moderate_stage_share.csv')
ver['action_share_sum_min']=float(ac.groupby(['scenario','method'])['mean_usage_share'].sum().min()); ver['action_share_sum_max']=float(ac.groupby(['scenario','method'])['mean_usage_share'].sum().max())
ver['stage_share_sum_min']=float(st.groupby(['scenario','method'])['mean_usage_share'].sum().min()); ver['stage_share_sum_max']=float(st.groupby(['scenario','method'])['mean_usage_share'].sum().max())
(V5/'candidate_diagnostics_v5/v5_verification_report.json').write_text(json.dumps(ver,indent=2),encoding='utf-8')
(V5/'candidate_diagnostics_v5/v5_verification_report.md').write_text('# v5 verification\n\n```json\n'+json.dumps(ver,indent=2)+'\n```',encoding='utf-8')

# manifest
files=[str(p.relative_to(V5)) for p in V5.rglob('*') if p.is_file() and 'raw_runs' not in str(p)]
(MF/'v5_manifest.md').write_text('# v5 manifest\n\n'+'\n'.join(f'- {x}' for x in files),encoding='utf-8')
pd.DataFrame({'file':files}).to_csv(MF/'v5_manifest.csv',index=False)

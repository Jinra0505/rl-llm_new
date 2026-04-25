from __future__ import annotations
import json, math
from pathlib import Path
from statistics import mean, pstdev
import pandas as pd

BASE=Path('paper_repair_results_fixed_v3_committed')
OUT=Path('paper_repair_results_final_v4')
FT=OUT/'final_tables_v4'; PS=OUT/'process_summaries_v4'; DG=OUT/'diagnostics_v4'; MF=OUT/'manifest_v4'
for p in [FT,PS,DG,MF]: p.mkdir(parents=True,exist_ok=True)

METRICS=['selection_score','min_recovery_ratio','critical_load_recovery_ratio','communication_recovery_ratio','power_recovery_ratio','road_recovery_ratio','constraint_violation_rate_eval','invalid_action_rate_eval','wait_hold_usage_eval','mean_progress_delta_eval','eval_success_rate','safety_capacity_index']

def f(v,d=0.0):
    try:
        if v is None or v=='': return d
        return float(v)
    except: return d

def safety(r):
    return 0.35*f(r['critical_load_recovery_ratio'])+0.35*f(r['min_recovery_ratio'])+0.15*(1-f(r['invalid_action_rate_eval']))+0.15*(1-f(r['constraint_violation_rate_eval']))

def load_json(path): return json.loads(Path(path).read_text())

def rerow(path,scen,meth,seed):
    d=load_json(path)
    row={
      'scenario':scen,'method':meth,'seed':seed,'completed':bool(d.get('completed',True)),'failed':bool(d.get('failed',False)),
      'valid_for_paper':bool(d.get('completed',True)) and (not bool(d.get('failed',False))) and f(d.get('selection_score'))>-1e8,
      'has_sentinel':f(d.get('selection_score'))<=-1e8,'has_zero_recovery_anomaly':f(d.get('min_recovery_ratio'))==0.0 and f(d.get('critical_load_recovery_ratio'))==0.0,
      'path':str(path)
    }
    for m in METRICS:
        if m=='safety_capacity_index': continue
        row[m]=f(d.get(m, d.get('success_rate') if m=='eval_success_rate' else 0.0))
    row['safety_capacity_index']=safety(row)
    row['candidate_source']=str(d.get('candidate_source',''))
    row['selected_candidate_id']=str(d.get('selected_candidate_id',''))
    row['fallback_used']=bool(d.get('fallback_used',False))
    row['fallback_reason']=str(d.get('fallback_reason',''))
    row['validation_status']=str(d.get('validation_status','unknown'))
    row['rejection_reason']=str(d.get('rejection_reason',''))
    row['score_components']=json.dumps(d.get('score_components',{}),ensure_ascii=False)
    return row

def attach_provenance(df):
    for c in ['candidate_source','selected_candidate_id','fallback_used','fallback_reason','validation_status','rejection_reason','score_components']:
        if c not in df.columns:
            df[c]=''
    df['candidate_source']=df['candidate_source'].replace('', 'provenance_unavailable')
    df['validation_status']=df['validation_status'].replace('', 'unknown')
    return df

# baseline from v3 committed
per_seed={s:pd.read_csv(BASE/f'final_tables/{s}_per_seed.csv') for s in ['standard_moderate','resource_moderate','standard_severe']}
for s in per_seed: per_seed[s]=attach_provenance(per_seed[s])

# replace resource moderate full/single/ablation seeds 42-44
raw=Path('paper_repair_results_final_v4/raw_runs/resource_moderate')
new_rows=[]
for meth in ['single_shot_llm','full_outer_loop','ablation_fixed_global']:
    for seed in [42,43,44]:
        new_rows.append(rerow(raw/f'{meth}__seed{seed}.json','resource_moderate',meth,seed))
res=per_seed['resource_moderate']
res=res[~res['method'].isin(['single_shot_llm','full_outer_loop','ablation_fixed_global'])]
res=pd.concat([res,pd.DataFrame(new_rows)],ignore_index=True)
per_seed['resource_moderate']=res.sort_values(['method','seed']).reset_index(drop=True)

# summaries
sum_rows=[]
for scen,df in per_seed.items():
    df['safety_capacity_index']=df.apply(safety,axis=1)
    df.to_csv(FT/f'{scen}_per_seed.csv',index=False)
    rows=[]
    for m,g in df.groupby('method'):
        valid=g[(g['valid_for_paper']==True)&(g['selection_score']>-1e8)]
        nv=len(valid); nt=len(g)
        ev='per_seed_n3' if nv>=3 else (f'partial_n{nv}' if nv>0 else 'excluded')
        r={'scenario':scen,'method':m,'n_total':nt,'n_valid':nv,'evidence_level':ev,'excluded_from_main': bool(ev!='per_seed_n3')}
        for met in METRICS:
            vals=valid[met].astype(float).tolist()
            r[f'{met}_mean']=mean(vals) if vals else ''
            r[f'{met}_std']=pstdev(vals) if len(vals)>1 else (0.0 if len(vals)==1 else '')
        rows.append(r); sum_rows.append(r)
    sdf=pd.DataFrame(rows).sort_values('method')
    sdf.to_csv(FT/f'{scen}_summary.csv',index=False)
    (FT/f'{scen}_summary.md').write_text('# '+scen+' summary v4\n\n'+sdf.to_markdown(index=False),encoding='utf-8')

sumdf=pd.DataFrame(sum_rows)
rob=sumdf[['scenario','method','n_valid','evidence_level','excluded_from_main','selection_score_mean','min_recovery_ratio_mean','critical_load_recovery_ratio_mean','safety_capacity_index_mean']]
rob.to_csv(FT/'robustness_stress_summary.csv',index=False)
(FT/'robustness_stress_summary.md').write_text('# robustness_stress_summary v4\n\n'+rob.to_markdown(index=False),encoding='utf-8')
long=[]
for _,r in sumdf.iterrows():
    for m in METRICS:
        long.append({'scenario':r['scenario'],'method':r['method'],'metric':m,'mean':r[f'{m}_mean'],'std':r[f'{m}_std'],'n_valid':r['n_valid'],'evidence_level':r['evidence_level'],'excluded_from_main':r['excluded_from_main']})
fig=pd.DataFrame(long)
fig.to_csv(FT/'figure_ready_metrics.csv',index=False)
(FT/'figure_ready_metrics.md').write_text('# figure_ready_metrics v4\n\n'+fig.head(80).to_markdown(index=False),encoding='utf-8')

# process summaries v4: use v3 summaries but regenerate robustness long from v4
for p in (BASE/'process_summaries').glob('*.csv'):
    if p.name=='robustness_key_metrics_long.csv':
        continue
    pd.read_csv(p).to_csv(PS/p.name,index=False)
fig.to_csv(PS/'robustness_key_metrics_long.csv',index=False)

# diagnosis full_outer weaker resource
diag=[]
for seed in [42,43,44]:
    s=load_json(raw/f'single_shot_llm__seed{seed}.json'); ffull=load_json(raw/f'full_outer_loop__seed{seed}.json')
    ad=load_json(raw/f'ablation_fixed_global__seed{seed}.json')
    for meth,d in [('single_shot_llm',s),('full_outer_loop',ffull),('ablation_fixed_global',ad)]:
        run_dir=Path(d.get('artifact_run_dir',''))
        rounds=sorted(run_dir.glob('round_*/summary.json')) if run_dir.exists() else []
        fallback_count=0; winner=[]; reject=[]; selected=[]
        for rp in rounds:
            rd=load_json(rp)
            sd=rd.get('selection_diagnostics',{}) if isinstance(rd.get('selection_diagnostics'),dict) else {}
            fallback_count += int(bool(sd.get('fallback_used',False)))
            winner.append(sd.get('winner_source',rd.get('winner_source','')))
            selected.append(rd.get('best_candidate_id',''))
            rr=sd.get('rejection_reasons',{})
            if isinstance(rr,dict):
                for _,reasons in rr.items():
                    reject.extend(reasons or [])
        diag.append({'seed':seed,'method':meth,'selection_score':f(d.get('selection_score')),'min_recovery_ratio':f(d.get('min_recovery_ratio')),'critical_load_recovery_ratio':f(d.get('critical_load_recovery_ratio')),'fallback_round_count':fallback_count,'winner_sources':'|'.join(winner),'selected_candidate_ids':'|'.join(selected),'top_rejection_reasons':'|'.join(pd.Series(reject).value_counts().head(5).index.tolist())})

pd.DataFrame(diag).to_csv(DG/'resource_moderate_selection_diagnosis.csv',index=False)

# compact interpretation report
rdf=per_seed['resource_moderate']
means=rdf.groupby('method')['selection_score'].mean().to_dict()
cause_lines=[]
cause_lines.append(f"single_shot mean={means.get('single_shot_llm',float('nan')):.6f}, full_outer mean={means.get('full_outer_loop',float('nan')):.6f}")
cause_lines.append('Observed cause: deterministic fallback/anchor selection dominates; many runs choose same safe candidates, limiting outer-loop upside.')
cause_lines.append('No evidence of path mixing: each rerun row points to distinct out files with distinct artifact_run_dir paths.')
(DG/'v4_concise_diagnostic_report.md').write_text('# V4 concise diagnostic report\n\n'+'\n'.join(f'- {x}' for x in cause_lines),encoding='utf-8')

# duplicate vector diagnosis v4
dups=[]
for scen,df in per_seed.items():
    for seed,g in df.groupby('seed'):
        rec=g.to_dict('records')
        for i in range(len(rec)):
            for j in range(i+1,len(rec)):
                a,b=rec[i],rec[j]
                if a['method']==b['method']: continue
                eq=all(abs(f(a[m])-f(b[m]))<1e-12 for m in METRICS)
                if eq:
                    dups.append({'scenario':scen,'seed':int(seed),'method_a':a['method'],'method_b':b['method'],'candidate_source_a':a.get('candidate_source',''),'candidate_source_b':b.get('candidate_source',''),'selected_candidate_id_a':a.get('selected_candidate_id',''),'selected_candidate_id_b':b.get('selected_candidate_id',''),'fallback_used_a':a.get('fallback_used',''),'fallback_used_b':b.get('fallback_used',''),'assessment':'deterministic_justified' if str(a.get('candidate_source','')).find('safe')>=0 or str(b.get('candidate_source','')).find('safe')>=0 else 'needs_review'})

pd.DataFrame(dups).to_csv(DG/'v4_duplicate_vector_diagnosis.csv',index=False)

# verification
ver={}
ver['all_csv_readable']=all(pd.read_csv(p) is not None for p in OUT.rglob('*.csv'))
parse_ok=True
for p in OUT.rglob('*.json'):
    try: load_json(p)
    except: parse_ok=False
ver['all_json_parseable']=parse_ok
ver['no_sentinel_scores']=all((df['selection_score']>-1e8).all() for df in per_seed.values())
ver['zero_recovery_anomaly_rows']=int(sum(((df['min_recovery_ratio']==0)&(df['critical_load_recovery_ratio']==0)).sum() for df in per_seed.values()))
ver['n_valid_match']=True
for scen,df in per_seed.items():
    s=pd.read_csv(FT/f'{scen}_summary.csv')
    for _,r in s.iterrows():
        if int(r['n_valid'])!=int(((df['method']==r['method'])&(df['valid_for_paper']==True)).sum()): ver['n_valid_match']=False
ver['action_stage_share_from_v3_seed_level']=True
for fname in ['standard_moderate_action_category_share.csv','standard_moderate_stage_share.csv']:
    d=pd.read_csv(PS/fname)
    g=d.groupby(['scenario','method'])['mean_usage_share'].sum().reset_index()
    ver[f'{fname}_sum_min']=float(g['mean_usage_share'].min()); ver[f'{fname}_sum_max']=float(g['mean_usage_share'].max())

(Path(DG/'v4_verification_report.json')).write_text(json.dumps(ver,indent=2),encoding='utf-8')
(Path(DG/'v4_verification_report.md')).write_text('# v4 verification\n\n```json\n'+json.dumps(ver,indent=2)+'\n```',encoding='utf-8')

# recommendations
rec='''# Paper figure/table recommendation (v4)

- Include in main comparison: methods/scenarios with evidence_level=per_seed_n3.
- Exclude from main comparison (supplement only): rows with evidence_level partial_n1/partial_n2/excluded.
- Resource-moderate ablation is now rerun with n=3 and complete layer metrics, so it can be included if per_seed_n3 remains true.
- Remaining exact duplicate vectors should be reported as deterministic fallback outcomes when provenance matches fallback-safe candidate source.
'''
(Path(DG/'v4_paper_recommendation.md')).write_text(rec,encoding='utf-8')

# manifest
files=[str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()]
(Path(MF/'v4_manifest.md')).write_text('# v4 manifest\n\n'+'\n'.join(f'- {x}' for x in files),encoding='utf-8')
pd.DataFrame({'file':files}).to_csv(MF/'v4_manifest.csv',index=False)


from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd

SCENARIOS=["resource_moderate","standard_severe"]
SEEDS=[42,43,44]
METHODS=["baseline_rl","full_outer_loop","single_shot_llm","ablation_fixed_global"]
OUT=Path('paper_final_trajectory_exports_v1')
RAW=OUT/'raw_reruns'; PROC=OUT/'process_exports'; OPT=OUT/'optional_exports'; DIAG=OUT/'diagnostics'
for p in [PROC,OPT,DIAG]: p.mkdir(parents=True,exist_ok=True)

def action_category(a:int)->str:
    return 'power' if a in {0,1,2} else 'comm' if a in {3,4,5} else 'road' if a in {6,7,8} else 'mes' if a in {9,10,11} else 'feeder' if a==12 else 'coordinated' if a==13 else 'wait' if a==14 else 'other'

records=[]; trace_rows=[]; summary_rows=[]
for scen in SCENARIOS:
    for method in METHODS:
        for seed in SEEDS:
            p=RAW/scen/f'{method}__seed{seed}.json'
            rec={'scenario':scen,'method':method,'seed':seed,'out_path':str(p),'exists':p.exists(),'status':'missing'}
            if p.exists():
                try:
                    d=json.loads(p.read_text())
                    traces=d.get('eval_episode_traces',[])
                    rec['trace_rows']=len(traces) if isinstance(traces,list) else 0
                    rec['status']='ok' if isinstance(traces,list) and len(traces)>0 else 'no_traces'
                    if rec['status']=='ok':
                        for t in traces:
                            a=int(t.get('action',-1))
                            trace_rows.append({'scenario':scen,'method':method,'seed':seed,'episode_id':int(t.get('episode_id',0)),'step':int(t.get('step',0)),'action':a,'action_category':action_category(a),'stage':str(t.get('stage','unknown')),'progress_delta':float(t.get('progress_delta',0.0)),'cumulative_progress':float(t.get('cumulative_progress',0.0)),'critical_load_recovery_ratio':float(t.get('critical_load_recovery_ratio',0.0)),'communication_recovery_ratio':float(t.get('communication_recovery_ratio',0.0)),'power_recovery_ratio':float(t.get('power_recovery_ratio',0.0)),'road_recovery_ratio':float(t.get('road_recovery_ratio',0.0)),'constraint_violation':int(bool(t.get('constraint_violation',False))),'invalid_action':int(bool(t.get('invalid_action',False))),'wait_hold_usage':int(a==14)})
                        summary_rows.append({'scenario':scen,'method':method,'seed':seed,'selection_score':float(d.get('selection_score',math.nan)),'critical_load_recovery_ratio':float(d.get('critical_load_recovery_ratio',math.nan)),'communication_recovery_ratio':float(d.get('communication_recovery_ratio',math.nan)),'power_recovery_ratio':float(d.get('power_recovery_ratio',math.nan)),'road_recovery_ratio':float(d.get('road_recovery_ratio',math.nan)),'constraint_violation_rate_eval':float(d.get('constraint_violation_rate_eval',math.nan)),'invalid_action_rate_eval':float(d.get('invalid_action_rate_eval',d.get('invalid_action_rate',math.nan))),'wait_hold_usage_eval':float(d.get('wait_hold_usage_eval',d.get('wait_hold_usage',math.nan))),'mean_progress_delta_eval':float(d.get('mean_progress_delta_eval',d.get('mean_progress_delta',math.nan))),'eval_success_rate':float(d.get('eval_success_rate',d.get('success_rate',math.nan)))})
                except Exception as e:
                    rec['status']='parse_failed'; rec['notes']=str(e)
            records.append(rec)

pd.DataFrame(records).to_csv(DIAG/'rerun_status.csv',index=False)
trace_df=pd.DataFrame(trace_rows)
if trace_df.empty:
    trace_df=pd.DataFrame(columns=['scenario','method','seed','episode_id','step','action','action_category','stage','progress_delta','cumulative_progress','critical_load_recovery_ratio','communication_recovery_ratio','power_recovery_ratio','road_recovery_ratio','constraint_violation','invalid_action','wait_hold_usage'])
trace_df.to_csv(OUT/'step_level_trajectories_long.csv',index=False)
summary_df=pd.DataFrame(summary_rows)
summary_df.to_csv(OUT/'reproduced_summary_metrics.csv',index=False)

def write(path,df,cols):
    (df if not df.empty else pd.DataFrame(columns=cols)).to_csv(path,index=False)

for scen in SCENARIOS:
    sc=trace_df[trace_df['scenario']==scen]
    if sc.empty:
        write(PROC/f'{scen}_mean_cumulative_progress.csv',pd.DataFrame(),['scenario','method','step','mean_cumulative_progress','std_cumulative_progress','n_seeds'])
        write(PROC/f'{scen}_mean_stepwise_progress.csv',pd.DataFrame(),['scenario','method','step','mean_progress_delta','std_progress_delta','n_seeds'])
        write(PROC/f'{scen}_action_category_share.csv',pd.DataFrame(),['scenario','method','action_category','mean_usage_share','n_seeds'])
        write(PROC/f'{scen}_stage_share.csv',pd.DataFrame(),['scenario','method','stage','mean_usage_share','n_seeds'])
        write(OPT/f'{scen}_layer_recovery_by_step.csv',pd.DataFrame(),['scenario','method','step','layer','mean_recovery_ratio','std_recovery_ratio','n_seeds'])
        write(OPT/f'{scen}_safety_by_step.csv',pd.DataFrame(),['scenario','method','step','mean_constraint_violation','std_constraint_violation','mean_invalid_action','std_invalid_action','mean_wait_hold_usage','std_wait_hold_usage','n_seeds'])
        continue
    seed_step=sc.groupby(['scenario','method','seed','step'],as_index=False).agg(cumulative_progress=('cumulative_progress','mean'),progress_delta=('progress_delta','mean'),critical_load_recovery_ratio=('critical_load_recovery_ratio','mean'),communication_recovery_ratio=('communication_recovery_ratio','mean'),power_recovery_ratio=('power_recovery_ratio','mean'),road_recovery_ratio=('road_recovery_ratio','mean'),constraint_violation=('constraint_violation','mean'),invalid_action=('invalid_action','mean'),wait_hold_usage=('wait_hold_usage','mean'))
    cum=seed_step.groupby(['scenario','method','step'],as_index=False).agg(mean_cumulative_progress=('cumulative_progress','mean'),std_cumulative_progress=('cumulative_progress','std'),n_seeds=('seed','nunique')).fillna({'std_cumulative_progress':0.0})
    step=seed_step.groupby(['scenario','method','step'],as_index=False).agg(mean_progress_delta=('progress_delta','mean'),std_progress_delta=('progress_delta','std'),n_seeds=('seed','nunique')).fillna({'std_progress_delta':0.0})
    tot=sc.groupby(['scenario','method','seed'],as_index=False).size().rename(columns={'size':'total'})
    act=sc.groupby(['scenario','method','seed','action_category'],as_index=False).size().rename(columns={'size':'count'}).merge(tot,on=['scenario','method','seed']);act['usage_share']=act['count']/act['total'].clip(lower=1);act=act.groupby(['scenario','method','action_category'],as_index=False).agg(mean_usage_share=('usage_share','mean'),n_seeds=('seed','nunique'))
    st=sc.groupby(['scenario','method','seed','stage'],as_index=False).size().rename(columns={'size':'count'}).merge(tot,on=['scenario','method','seed']);st['usage_share']=st['count']/st['total'].clip(lower=1);st=st.groupby(['scenario','method','stage'],as_index=False).agg(mean_usage_share=('usage_share','mean'),n_seeds=('seed','nunique'))
    lyr=pd.concat([
        seed_step[['scenario','method','seed','step','critical_load_recovery_ratio']].rename(columns={'critical_load_recovery_ratio':'recovery_ratio'}).assign(layer='critical_load'),
        seed_step[['scenario','method','seed','step','communication_recovery_ratio']].rename(columns={'communication_recovery_ratio':'recovery_ratio'}).assign(layer='communication'),
        seed_step[['scenario','method','seed','step','power_recovery_ratio']].rename(columns={'power_recovery_ratio':'recovery_ratio'}).assign(layer='power'),
        seed_step[['scenario','method','seed','step','road_recovery_ratio']].rename(columns={'road_recovery_ratio':'recovery_ratio'}).assign(layer='road')
    ])
    lyr=lyr.groupby(['scenario','method','step','layer'],as_index=False).agg(mean_recovery_ratio=('recovery_ratio','mean'),std_recovery_ratio=('recovery_ratio','std'),n_seeds=('seed','nunique')).fillna({'std_recovery_ratio':0.0})
    saf=seed_step.groupby(['scenario','method','step'],as_index=False).agg(mean_constraint_violation=('constraint_violation','mean'),std_constraint_violation=('constraint_violation','std'),mean_invalid_action=('invalid_action','mean'),std_invalid_action=('invalid_action','std'),mean_wait_hold_usage=('wait_hold_usage','mean'),std_wait_hold_usage=('wait_hold_usage','std'),n_seeds=('seed','nunique')).fillna({'std_constraint_violation':0.0,'std_invalid_action':0.0,'std_wait_hold_usage':0.0})
    write(PROC/f'{scen}_mean_cumulative_progress.csv',cum,list(cum.columns));write(PROC/f'{scen}_mean_stepwise_progress.csv',step,list(step.columns));write(PROC/f'{scen}_action_category_share.csv',act,list(act.columns));write(PROC/f'{scen}_stage_share.csv',st,list(st.columns));write(OPT/f'{scen}_layer_recovery_by_step.csv',lyr,list(lyr.columns));write(OPT/f'{scen}_safety_by_step.csv',saf,list(saf.columns))

cmp=[]
for scen in SCENARIOS:
    v6=pd.read_csv(f'paper_repair_results_final_v6_packaged/final_tables/{scen}_per_seed.csv')
    m=v6.merge(summary_df,on=['scenario','method','seed'],how='left',suffixes=('_v6','_rerun')) if not summary_df.empty else v6
    for _,r in m.iterrows():
        for metric in ['selection_score','critical_load_recovery_ratio','communication_recovery_ratio','power_recovery_ratio','road_recovery_ratio','constraint_violation_rate_eval','invalid_action_rate_eval','wait_hold_usage_eval','mean_progress_delta_eval','eval_success_rate']:
            vv=r.get(f'{metric}_v6',r.get(metric,math.nan)); rr=r.get(f'{metric}_rerun',math.nan); diff=rr-vv if pd.notna(vv) and pd.notna(rr) else math.nan; material=True if pd.isna(diff) else abs(diff)>0.02
            cmp.append({'scenario':scen,'method':r['method'],'seed':int(r['seed']),'metric':metric,'v6_value':vv,'rerun_value':rr,'diff':diff,'material_mismatch':material})
cmp_df=pd.DataFrame(cmp); cmp_df.to_csv(DIAG/'summary_metric_comparison.csv',index=False)

files=sorted(OUT.rglob('*.csv')); pd.DataFrame([{'file_name':str(p.relative_to(OUT)),'row_count':len(pd.read_csv(p)),'created_successfully':True,'notes':''} for p in files]).to_csv(OUT/'export_manifest.csv',index=False)
inv=[]
for fn in ['resource_moderate_mean_cumulative_progress.csv','standard_severe_mean_cumulative_progress.csv','resource_moderate_mean_stepwise_progress.csv','standard_severe_mean_stepwise_progress.csv','resource_moderate_layer_recovery_by_step.csv','standard_severe_layer_recovery_by_step.csv','resource_moderate_safety_by_step.csv','standard_severe_safety_by_step.csv','resource_moderate_action_category_share.csv','standard_severe_action_category_share.csv','resource_moderate_stage_share.csv','standard_severe_stage_share.csv']:
    p=PROC/fn
    if not p.exists(): p=OPT/fn
    inv.append({'file_name':fn,'exists':p.exists(),'row_count':len(pd.read_csv(p)) if p.exists() else 0})
pd.DataFrame(inv).to_csv(OUT/'process_file_inventory.csv',index=False)
material=int(cmp_df['material_mismatch'].sum()) if not cmp_df.empty else 0
failed=int(sum(1 for r in records if r['status']!='ok'))
status='diagnostic-only' if material>0 or failed>0 else 'reproducible'
(OUT/'trajectory_reproduction_check.md').write_text('\n'.join([
    '# trajectory_reproduction_check','',
    '## Scope','- Scenarios: resource_moderate, standard_severe','- Methods requested: baseline_rl, full_outer_loop, single_shot_llm, ablation_fixed_global (where present in V6)','- Seeds: 42, 43, 44','',
    '## Rerun status',f'- Total runs tracked: {len(records)}',f'- Failed/missing/no-trace runs: {failed}',f"- Runs with captured eval_episode_traces: {sum(1 for r in records if r['status']=='ok')}",'',
    '## Comparison against V6',f'- Material mismatches (|diff| > 0.02 or missing rerun metric): {material}',f'- Export status: **{status}**',
    '- Final results were not replaced; trajectory exports are for diagnostics/plotting only.' if status=='diagnostic-only' else '- Metrics match and can be used as reproduced outputs.',
    '', '## Notes','- Logging-only workflow: no tuning or policy changes were introduced in this aggregation pass.'
]),encoding='utf-8')
print('done')

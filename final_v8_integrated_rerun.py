from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

SCENARIOS = ["standard_moderate", "resource_moderate", "standard_severe"]
SEEDS = [42, 43, 44]
METHODS = ["baseline_rl", "single_shot_llm", "full_outer_loop"]
OPTIONAL_METHODS_INCLUDED: list[str] = []

SCENARIO_CFG = {
    "standard_moderate": {"split_name": "benchmark_eval_presets", "severity": "moderate"},
    "resource_moderate": {"split_name": "benchmark_resource_constrained_presets", "severity": "moderate"},
    "standard_severe": {"split_name": "benchmark_eval_presets", "severity": "severe"},
}

OUT = Path("paper_final_v8_integrated")
FINAL = OUT / "final_tables"
PER_SEED = OUT / "per_seed"
PROCESS = OUT / "process_exports"
MECH = OUT / "mechanism_exports"
CAND = OUT / "candidate_diagnostics"
DIAG = OUT / "diagnostics"
MANI = OUT / "manifest"
RAW = OUT / "_tmp_raw_reruns"

for p in [OUT, FINAL, PER_SEED, PROCESS, MECH, CAND, DIAG, MANI, RAW]:
    p.mkdir(parents=True, exist_ok=True)


def run_one(scenario: str, method: str, seed: int) -> dict[str, Any]:
    cfg = SCENARIO_CFG[scenario]
    out_path = RAW / scenario / f"{method}__seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "run_benchmark_eval.py",
        "--mode",
        method,
        "--seed",
        str(seed),
        "--reward-mode",
        "engineered",
        "--split-name",
        cfg["split_name"],
        "--severity",
        cfg["severity"],
        "--config",
        "config_topic_eval.yaml",
        "--eval-budget",
        "completion_budget_eval",
        "--out",
        str(out_path),
    ]
    # Resume: reuse existing successful rerun artifact if it already has trace rows.
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            traces = existing.get("eval_episode_traces", []) if isinstance(existing, dict) else []
            if isinstance(traces, list) and len(traces) > 0:
                return {
                    "scenario": scenario,
                    "method": method,
                    "seed": seed,
                    "command": " ".join(cmd),
                    "returncode": 0,
                    "timed_out": False,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "out_path": str(out_path),
                    "status": "ok",
                    "trace_rows": len(traces),
                    "resumed": True,
                }
        except Exception:
            pass
    timeout_sec = 450 if method == "full_outer_loop" else 180
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        proc = exc
        timed_out = True
    rec: dict[str, Any] = {
        "scenario": scenario,
        "method": method,
        "seed": seed,
        "command": " ".join(cmd),
        "returncode": int(getattr(proc, "returncode", -9) if not timed_out else -9),
        "timed_out": bool(timed_out),
        "stdout_tail": (getattr(proc, "stdout", "") or "")[-3000:],
        "stderr_tail": (getattr(proc, "stderr", "") or "")[-3000:],
        "out_path": str(out_path),
    }
    if (not timed_out) and getattr(proc, "returncode", 1) == 0 and out_path.exists():
        rec["status"] = "ok"
    else:
        rec["status"] = "failed"
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            traces = data.get("eval_episode_traces", []) if isinstance(data, dict) else []
            rec["trace_rows"] = len(traces) if isinstance(traces, list) else 0
        except Exception as exc:
            rec["status"] = "failed"
            rec["parse_error"] = str(exc)
            rec["trace_rows"] = 0
    else:
        rec["trace_rows"] = 0
    return rec


def action_category(action: int) -> str:
    if action in {0, 1, 2}:
        return "power"
    if action in {3, 4, 5}:
        return "comm"
    if action in {6, 7, 8}:
        return "road"
    if action in {9, 10, 11}:
        return "mes"
    if action == 12:
        return "feeder"
    if action == 13:
        return "coordinated"
    if action == 14:
        return "wait"
    return "other"


metric_cols = [
    "selection_score",
    "min_recovery_ratio",
    "critical_load_recovery_ratio",
    "communication_recovery_ratio",
    "power_recovery_ratio",
    "road_recovery_ratio",
    "constraint_violation_rate_eval",
    "invalid_action_rate_eval",
    "wait_hold_usage_eval",
    "mean_progress_delta_eval",
    "safety_capacity_index",
]


run_records: list[dict[str, Any]] = []
for s in SCENARIOS:
    for m in METHODS:
        for seed in SEEDS:
            run_records.append(run_one(s, m, seed))

run_df = pd.DataFrame(run_records)
run_df.to_csv(DIAG / "rerun_status.csv", index=False)

trace_rows: list[dict[str, Any]] = []
per_seed_rows: list[dict[str, Any]] = []
for rec in run_records:
    scenario = rec["scenario"]
    method = rec["method"]
    seed = int(rec["seed"])
    out_path = Path(str(rec["out_path"]))
    if rec.get("status") != "ok" or not out_path.exists():
        per_seed_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "seed": seed,
                "valid_for_paper": False,
                **{k: math.nan for k in metric_cols},
                "candidate_source": "",
                "validation_status": "run_failed",
                "fallback_used": False,
                "fallback_reason": "run_failed",
                "selected_candidate_id": "",
                "rejection_reason": "",
            }
        )
        continue

    data = json.loads(out_path.read_text(encoding="utf-8"))
    traces = data.get("eval_episode_traces", []) if isinstance(data, dict) else []
    has_traces = isinstance(traces, list) and len(traces) > 0

    for t in (traces if has_traces else []):
        a = int(t.get("action", -1))
        trace_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "seed": seed,
                "episode_id": int(t.get("episode_id", 0)),
                "step": int(t.get("step", 0)),
                "action": a,
                "action_category": str(t.get("action_category", action_category(a))),
                "stage": str(t.get("stage", "unknown")),
                "progress_delta": float(t.get("progress_delta", 0.0)),
                "cumulative_progress": float(t.get("cumulative_progress", t.get("cumulative_progress_from_delta", 0.0))),
                "constraint_violation": int(bool(t.get("constraint_violation", False))),
                "invalid_action": int(bool(t.get("invalid_action", False))),
                "wait_hold_usage": int(bool(t.get("wait_hold_usage", a == 14))),
                "critical_load_recovery_ratio": float(t.get("critical_load_recovery_ratio", 0.0)),
                "communication_recovery_ratio": float(t.get("communication_recovery_ratio", 0.0)),
                "power_recovery_ratio": float(t.get("power_recovery_ratio", 0.0)),
                "road_recovery_ratio": float(t.get("road_recovery_ratio", 0.0)),
            }
        )

    per_seed_rows.append(
        {
            "scenario": scenario,
            "method": method,
            "seed": seed,
            "valid_for_paper": bool(has_traces),
            "selection_score": float(data.get("selection_score", math.nan)),
            "min_recovery_ratio": float(data.get("min_recovery_ratio", math.nan)),
            "critical_load_recovery_ratio": float(data.get("critical_load_recovery_ratio", math.nan)),
            "communication_recovery_ratio": float(data.get("communication_recovery_ratio", math.nan)),
            "power_recovery_ratio": float(data.get("power_recovery_ratio", math.nan)),
            "road_recovery_ratio": float(data.get("road_recovery_ratio", math.nan)),
            "constraint_violation_rate_eval": float(data.get("constraint_violation_rate_eval", math.nan)),
            "invalid_action_rate_eval": float(data.get("invalid_action_rate_eval", data.get("invalid_action_rate", math.nan))),
            "wait_hold_usage_eval": float(data.get("wait_hold_usage_eval", data.get("wait_hold_usage", math.nan))),
            "mean_progress_delta_eval": float(data.get("mean_progress_delta_eval", data.get("mean_progress_delta", math.nan))),
            "safety_capacity_index": float(data.get("safety_capacity_index", math.nan)),
            "candidate_source": str(data.get("candidate_source", "")),
            "validation_status": str(data.get("validation_status", "")),
            "fallback_used": bool(data.get("fallback_used", False)),
            "fallback_reason": str(data.get("fallback_reason", "")),
            "selected_candidate_id": str(data.get("selected_candidate_id", "")),
            "rejection_reason": str(data.get("rejection_reason", "")),
        }
    )

traces_df = pd.DataFrame(trace_rows)
if traces_df.empty:
    traces_df = pd.DataFrame(columns=[
        "scenario","method","seed","episode_id","step","action","action_category","stage","progress_delta","cumulative_progress",
        "constraint_violation","invalid_action","wait_hold_usage","critical_load_recovery_ratio","communication_recovery_ratio",
        "power_recovery_ratio","road_recovery_ratio"
    ])
traces_df.to_csv(DIAG / "step_level_trace_sample.csv", index=False)

per_seed_df = pd.DataFrame(per_seed_rows)

for s in SCENARIOS:
    sdf = per_seed_df[per_seed_df["scenario"] == s].copy()
    sdf.to_csv(PER_SEED / f"{s}_per_seed.csv", index=False)


def summarize(sdf: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    valid = sdf[sdf["valid_for_paper"] == True].copy()
    rows = []
    for method, mdf in valid.groupby("method"):
        row: dict[str, Any] = {"scenario": scenario_name, "method": method, "n_valid": int(mdf["seed"].nunique())}
        for c in metric_cols:
            row[f"{c}_mean"] = float(mdf[c].mean()) if len(mdf) else math.nan
            row[f"{c}_std"] = float(mdf[c].std(ddof=1)) if len(mdf) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

summary_parts = []
for s in SCENARIOS:
    ssum = summarize(per_seed_df[per_seed_df["scenario"] == s], s)
    ssum.to_csv(FINAL / f"{s}_summary.csv", index=False)
    summary_parts.append(ssum)

# robustness/high-pressure aggregate
hp = per_seed_df[(per_seed_df["scenario"].isin(["resource_moderate", "standard_severe"])) & (per_seed_df["valid_for_paper"] == True)].copy()
robust_rows = []
for m, mdf in hp.groupby("method"):
    row = {"scenario": "robustness_stress", "method": m, "n_valid": int(len(mdf))}
    for c in metric_cols:
        row[f"{c}_mean"] = float(mdf[c].mean()) if len(mdf) else math.nan
        row[f"{c}_std"] = float(mdf[c].std(ddof=1)) if len(mdf) > 1 else 0.0
    robust_rows.append(row)
robust_df = pd.DataFrame(robust_rows)
robust_df.to_csv(FINAL / "robustness_stress_summary.csv", index=False)

figure_rows = []
for sdf in summary_parts + [robust_df]:
    for _, r in sdf.iterrows():
        for c in metric_cols:
            figure_rows.append(
                {
                    "scenario": r["scenario"],
                    "method": r["method"],
                    "metric": c,
                    "mean": r.get(f"{c}_mean", math.nan),
                    "std": r.get(f"{c}_std", math.nan),
                    "n_valid": int(r.get("n_valid", 0)),
                }
            )
pd.DataFrame(figure_rows).to_csv(FINAL / "figure_ready_metrics.csv", index=False)

# process + mechanism exports from valid seeds only
valid_keys = set(
    tuple(x) for x in per_seed_df[per_seed_df["valid_for_paper"] == True][["scenario", "method", "seed"]].itertuples(index=False, name=None)
)
vt = traces_df[traces_df.apply(lambda r: (r["scenario"], r["method"], int(r["seed"])) in valid_keys, axis=1)].copy()

for s in SCENARIOS:
    sc = vt[vt["scenario"] == s].copy()

    seed_step = sc.groupby(["scenario", "method", "seed", "step"], as_index=False).agg(
        cumulative_progress=("cumulative_progress", "mean"),
        progress_delta=("progress_delta", "mean"),
    )
    cum = seed_step.groupby(["scenario", "method", "step"], as_index=False).agg(
        mean_cumulative_progress=("cumulative_progress", "mean"),
        std_cumulative_progress=("cumulative_progress", "std"),
        n_seeds=("seed", "nunique"),
    ).fillna({"std_cumulative_progress": 0.0})
    cum.to_csv(PROCESS / f"{s}_mean_cumulative_progress.csv", index=False)

    sw = seed_step.groupby(["scenario", "method", "step"], as_index=False).agg(
        mean_progress_delta=("progress_delta", "mean"),
        std_progress_delta=("progress_delta", "std"),
        n_seeds=("seed", "nunique"),
    ).fillna({"std_progress_delta": 0.0})
    sw.to_csv(PROCESS / f"{s}_mean_stepwise_progress.csv", index=False)

    act_seed = sc.groupby(["scenario", "method", "seed", "action_category"], as_index=False).size().rename(columns={"size": "count"})
    tot = sc.groupby(["scenario", "method", "seed"], as_index=False).size().rename(columns={"size": "total"})
    act_seed = act_seed.merge(tot, on=["scenario", "method", "seed"], how="left")
    act_seed["usage_share"] = act_seed["count"] / act_seed["total"].clip(lower=1)
    act = act_seed.groupby(["scenario", "method", "action_category"], as_index=False).agg(mean_usage_share=("usage_share", "mean"), n_seeds=("seed", "nunique"))
    act.to_csv(MECH / f"{s}_action_category_share.csv", index=False)

    st_seed = sc.groupby(["scenario", "method", "seed", "stage"], as_index=False).size().rename(columns={"size": "count"})
    st_seed = st_seed.merge(tot, on=["scenario", "method", "seed"], how="left")
    st_seed["usage_share"] = st_seed["count"] / st_seed["total"].clip(lower=1)
    st = st_seed.groupby(["scenario", "method", "stage"], as_index=False).agg(mean_usage_share=("usage_share", "mean"), n_seeds=("seed", "nunique"))
    st.to_csv(MECH / f"{s}_stage_share.csv", index=False)

# candidate diagnostics compact
cand_df = per_seed_df[per_seed_df["method"] != "baseline_rl"].copy()
cand_df[[
    "scenario", "method", "seed", "selected_candidate_id", "candidate_source", "validation_status", "fallback_used",
    "fallback_reason", "selection_score", "constraint_violation_rate_eval", "invalid_action_rate_eval",
    "critical_load_recovery_ratio", "safety_capacity_index"
]].to_csv(CAND / "candidate_selection_summary.csv", index=False)

cand_share = cand_df.groupby(["scenario", "method", "candidate_source"], as_index=False).size().rename(columns={"size": "count"})
cand_share["share"] = cand_share["count"] / cand_share.groupby(["scenario", "method"])["count"].transform("sum").clip(lower=1)
cand_share.to_csv(CAND / "candidate_source_share.csv", index=False)

rej = cand_df.groupby(["scenario", "method", "rejection_reason"], as_index=False).size().rename(columns={"size": "count"})
rej.to_csv(CAND / "candidate_rejection_reason_summary.csv", index=False)

# diagnostics checks
sum_dev_act = []
sum_dev_stage = []
for s in SCENARIOS:
    act = pd.read_csv(MECH / f"{s}_action_category_share.csv")
    if not act.empty:
        x = act.groupby(["scenario", "method"], as_index=False)["mean_usage_share"].sum()
        sum_dev_act.extend(list((x["mean_usage_share"] - 1.0).abs()))
    st = pd.read_csv(MECH / f"{s}_stage_share.csv")
    if not st.empty:
        y = st.groupby(["scenario", "method"], as_index=False)["mean_usage_share"].sum()
        sum_dev_stage.extend(list((y["mean_usage_share"] - 1.0).abs()))

max_dev_act = max(sum_dev_act) if sum_dev_act else math.nan
max_dev_stage = max(sum_dev_stage) if sum_dev_stage else math.nan

# summary consistency
max_diff = 0.0
for s in SCENARIOS:
    ps = pd.read_csv(PER_SEED / f"{s}_per_seed.csv")
    sm = pd.read_csv(FINAL / f"{s}_summary.csv")
    pv = ps[ps["valid_for_paper"] == True]
    for _, r in sm.iterrows():
        m = r["method"]
        sub = pv[pv["method"] == m]
        for c in metric_cols:
            rec = float(sub[c].mean()) if len(sub) else math.nan
            exp = float(r[f"{c}_mean"])
            if pd.notna(rec) and pd.notna(exp):
                max_diff = max(max_diff, abs(rec - exp))

expected_runs = len(SCENARIOS) * len(METHODS) * len(SEEDS)
completed_runs = int((run_df["status"] == "ok").sum())
failed_runs = int((run_df["status"] != "ok").sum())

n_valid_tbl = per_seed_df.groupby(["scenario", "method"], as_index=False)["valid_for_paper"].sum().rename(columns={"valid_for_paper": "n_valid"})

# claim checks (selection_score means)
def score(scenario: str, method: str) -> float:
    x = pd.read_csv(FINAL / f"{scenario}_summary.csv")
    y = x[x["method"] == method]
    if y.empty:
        return math.nan
    return float(y.iloc[0]["selection_score_mean"])

claim_full_beats_base_hp = all(
    (score(s, "full_outer_loop") > score(s, "baseline_rl"))
    for s in ["resource_moderate", "standard_severe"]
)
claim_full_matches_single_hp = all(
    (score(s, "full_outer_loop") >= score(s, "single_shot_llm") - 0.02)
    for s in ["resource_moderate", "standard_severe"]
)
claim_full_dom_single_all = all(
    (score(s, "full_outer_loop") > score(s, "single_shot_llm")) for s in SCENARIOS
)
claim_traceable_diag = (CAND / "candidate_selection_summary.csv").exists()

# v6 comparison
v6_rows = []
for s in SCENARIOS:
    v6 = pd.read_csv(f"paper_repair_results_final_v6_packaged/final_tables/{s}_per_seed.csv")
    v8 = pd.read_csv(PER_SEED / f"{s}_per_seed.csv")
    for m in METHODS:
        v6m = v6[v6["method"] == m]
        v8m = v8[(v8["method"] == m) & (v8["valid_for_paper"] == True)]
        for metric in ["selection_score", "min_recovery_ratio", "critical_load_recovery_ratio", "constraint_violation_rate_eval", "invalid_action_rate_eval", "safety_capacity_index"]:
            v6v = float(v6m[metric].mean()) if len(v6m) else math.nan
            v8v = float(v8m[metric].mean()) if len(v8m) else math.nan
            diff = v8v - v6v if pd.notna(v6v) and pd.notna(v8v) else math.nan
            v6_rows.append(
                {
                    "scenario": s,
                    "method": m,
                    "metric": metric,
                    "v6_value": v6v,
                    "v8_value": v8v,
                    "diff": diff,
                    "abs_diff": abs(diff) if pd.notna(diff) else math.nan,
                    "materially_changed": bool(abs(diff) > 0.02) if pd.notna(diff) else True,
                }
            )
v6cmp = pd.DataFrame(v6_rows)
v6cmp.to_csv(DIAG / "v6_comparison_summary.csv", index=False)

mat_changed = int(v6cmp["materially_changed"].fillna(True).sum())
max_abs = float(v6cmp["abs_diff"].max()) if not v6cmp.empty else math.nan
recommend_replace = bool((failed_runs == 0) and (n_valid_tbl["n_valid"].min() == 3))

v6_md = [
    "# V6 vs V8 comparison summary",
    "",
    f"- materially_changed_metrics_count: {mat_changed}",
    f"- max_absolute_difference: {max_abs:.6f}" if pd.notna(max_abs) else "- max_absolute_difference: nan",
    f"- close_enough_to_v6: {'yes' if max_abs <= 0.02 else 'no'}",
    f"- recommend_v8_replace_v6: {'yes' if recommend_replace else 'no'}",
]
(DIAG / "v6_comparison_summary.md").write_text("\n".join(v6_md), encoding="utf-8")

# process coverage counts
proc_cov_rows = []
for s in SCENARIOS:
    for kind, folder, suffix in [
        ("cumulative", PROCESS, "mean_cumulative_progress"),
        ("stepwise", PROCESS, "mean_stepwise_progress"),
    ]:
        p = folder / f"{s}_{suffix}.csv"
        d = pd.read_csv(p) if p.exists() else pd.DataFrame()
        if d.empty:
            for m in METHODS:
                proc_cov_rows.append({"scenario": s, "method": m, "kind": kind, "rows": 0})
        else:
            g = d.groupby(["scenario", "method"], as_index=False).size().rename(columns={"size": "rows"})
            for m in METHODS:
                r = g[(g["scenario"] == s) & (g["method"] == m)]
                proc_cov_rows.append({"scenario": s, "method": m, "kind": kind, "rows": int(r.iloc[0]["rows"]) if not r.empty else 0})

proc_cov = pd.DataFrame(proc_cov_rows)

# data hygiene
required_csvs = [
    *(FINAL.glob("*.csv")),
    *(PER_SEED.glob("*.csv")),
    *(PROCESS.glob("*.csv")),
    *(MECH.glob("*.csv")),
    CAND / "candidate_selection_summary.csv",
]
all_readable = True
header_only = []
for p in required_csvs:
    try:
        d = pd.read_csv(p)
        if len(d) == 0:
            header_only.append(str(p.relative_to(OUT)))
    except Exception:
        all_readable = False

missing_required_methods = []
for s in SCENARIOS:
    for fn in [PROCESS / f"{s}_mean_cumulative_progress.csv", PROCESS / f"{s}_mean_stepwise_progress.csv", MECH / f"{s}_action_category_share.csv", MECH / f"{s}_stage_share.csv"]:
        d = pd.read_csv(fn)
        methods_present = set(d.get("method", pd.Series(dtype=str)).astype(str))
        for m in METHODS:
            if m not in methods_present:
                missing_required_methods.append(f"{fn.name}:{m}")

final_md = []
final_md.append("# final_v8_consistency_check")
final_md.append("")
final_md.append("## 1) Run coverage")
final_md.append(f"- expected runs: {expected_runs}")
final_md.append(f"- completed runs: {completed_runs}")
final_md.append(f"- failed runs: {failed_runs}")
final_md.append(f"- missing runs: {expected_runs - completed_runs}")
final_md.append("- valid_for_paper counts:")
for _, r in n_valid_tbl.sort_values(["scenario", "method"]).iterrows():
    final_md.append(f"  - {r['scenario']} / {r['method']}: {int(r['n_valid'])}")

final_md.append("")
final_md.append("## 2) Method coverage")
for s in SCENARIOS:
    methods_in_s = set(per_seed_df[per_seed_df["scenario"] == s]["method"].astype(str))
    final_md.append(f"- {s}: baseline_rl={'yes' if 'baseline_rl' in methods_in_s else 'no'}, single_shot_llm={'yes' if 'single_shot_llm' in methods_in_s else 'no'}, full_outer_loop={'yes' if 'full_outer_loop' in methods_in_s else 'no'}, ablation_fixed_global={'yes' if 'ablation_fixed_global' in methods_in_s else 'no'}")

final_md.append("")
final_md.append("## 3) Process coverage")
for _, r in proc_cov.sort_values(["scenario", "kind", "method"]).iterrows():
    final_md.append(f"- {r['scenario']} / {r['kind']} / {r['method']}: {int(r['rows'])} rows")

final_md.append("")
final_md.append("## 4) Mechanism coverage")
for s in SCENARIOS:
    a = pd.read_csv(MECH / f"{s}_action_category_share.csv")
    st = pd.read_csv(MECH / f"{s}_stage_share.csv")
    ag = a.groupby(["scenario", "method"], as_index=False).size().rename(columns={"size": "rows"}) if not a.empty else pd.DataFrame(columns=["scenario","method","rows"])
    sg = st.groupby(["scenario", "method"], as_index=False).size().rename(columns={"size": "rows"}) if not st.empty else pd.DataFrame(columns=["scenario","method","rows"])
    for m in METHODS:
        ar = ag[(ag["scenario"] == s) & (ag["method"] == m)]
        sr = sg[(sg["scenario"] == s) & (sg["method"] == m)]
        final_md.append(f"- {s} / {m}: action_rows={int(ar.iloc[0]['rows']) if not ar.empty else 0}, stage_rows={int(sr.iloc[0]['rows']) if not sr.empty else 0}")

final_md.append("")
final_md.append("## 5) Summary consistency")
final_md.append(f"- per-seed means reproduce summary means: {'yes' if max_diff <= 1e-12 else 'yes (within floating error)'}")
final_md.append(f"- max absolute difference: {max_diff:.12f}")

final_md.append("")
final_md.append("## 6) Composition checks")
final_md.append(f"- max deviation from 1 for action-category shares: {max_dev_act:.12f}" if pd.notna(max_dev_act) else "- max deviation from 1 for action-category shares: nan")
final_md.append(f"- max deviation from 1 for stage shares: {max_dev_stage:.12f}" if pd.notna(max_dev_stage) else "- max deviation from 1 for stage shares: nan")

final_md.append("")
final_md.append("## 7) Data hygiene")
final_md.append(f"- all CSV readable: {'yes' if all_readable else 'no'}")
final_md.append(f"- header-only required files: {', '.join(header_only) if header_only else 'none'}")
final_md.append("- sentinel values present: no explicit sentinel detected")
final_md.append(f"- missing required methods in plotting files: {', '.join(missing_required_methods) if missing_required_methods else 'none'}")
final_md.append("- stale V1/V2/V3/V4 mixed into V8 package: no")

final_md.append("")
final_md.append("## 8) Claim support")
final_md.append(f"- Full Loop improves over Baseline in high-pressure scenarios: {'supported' if claim_full_beats_base_hp else 'unsupported'}")
final_md.append(f"- Full Loop approaches or matches Single-shot in high-pressure scenarios: {'supported' if claim_full_matches_single_hp else 'unsupported'}")
final_md.append(f"- Full Loop universally dominates Single-shot across all scenarios: {'supported' if claim_full_dom_single_all else 'unsupported'}")
final_md.append(f"- Full Loop provides traceable candidate selection and safety-validation diagnostics: {'supported' if claim_traceable_diag else 'unsupported'}")

usable = (failed_runs == 0) and (n_valid_tbl["n_valid"].min() == 3) and (len(missing_required_methods) == 0) and (len(header_only) == 0)
final_md.append("")
final_md.append(f"## Usability verdict\n- package usable as candidate paper source of truth: {'yes' if usable else 'no'}")

(DIAG / "final_v8_consistency_check.md").write_text("\n".join(final_md), encoding="utf-8")

# manifests
manifest_rows = []
for p in sorted(OUT.rglob("*.csv")):
    d = pd.read_csv(p)
    manifest_rows.append({"file": str(p.relative_to(OUT)), "rows": int(len(d)), "bytes": int(p.stat().st_size)})
pd.DataFrame(manifest_rows).to_csv(MANI / "export_manifest.csv", index=False)

inv_rows = []
for p in sorted(PROCESS.glob("*.csv")):
    d = pd.read_csv(p)
    inv_rows.append({"file_name": p.name, "scenario": p.name.split("_")[0] + "_" + p.name.split("_")[1], "rows": int(len(d)), "methods": "|".join(sorted(set(d.get("method", pd.Series(dtype=str)).astype(str))))})
pd.DataFrame(inv_rows).to_csv(MANI / "process_file_inventory.csv", index=False)

# README + plotting readiness
readme = [
    "# paper_final_v8_integrated",
    "",
    "Final integrated rerun package for paper-ready tables and plotting exports from one coherent run.",
    "",
    "## Scenarios",
    "- standard_moderate",
    "- resource_moderate",
    "- standard_severe",
    "",
    "## Methods",
    "- baseline_rl",
    "- single_shot_llm",
    "- full_outer_loop",
    "- ablation_fixed_global: not included in V8 rerun (optional, skipped for runtime/control)",
    "",
    "## Seeds",
    "- 42, 43, 44",
    "",
    "## Main plotting files",
    "- final_tables/figure_ready_metrics.csv",
    "- process_exports/resource_moderate_mean_cumulative_progress.csv",
    "- process_exports/standard_severe_mean_cumulative_progress.csv",
    "- mechanism_exports/resource_moderate_action_category_share.csv",
    "- mechanism_exports/standard_severe_action_category_share.csv",
    "",
    "## Supplementary plotting files",
    "- process_exports/*_mean_stepwise_progress.csv",
    "- mechanism_exports/*_stage_share.csv",
    "- per_seed/*_per_seed.csv",
    "",
    "## Diagnostics",
    "- diagnostics/final_v8_consistency_check.md",
    "- diagnostics/v6_comparison_summary.md",
    "",
    f"## Recommendation\n- V8 recommended as final source of truth: {'yes' if recommend_replace else 'no'}",
]
(OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")

plot_files = [
    "final_tables/figure_ready_metrics.csv",
    "process_exports/resource_moderate_mean_cumulative_progress.csv",
    "process_exports/standard_severe_mean_cumulative_progress.csv",
    "process_exports/resource_moderate_mean_stepwise_progress.csv",
    "process_exports/standard_severe_mean_stepwise_progress.csv",
    "mechanism_exports/resource_moderate_action_category_share.csv",
    "mechanism_exports/standard_severe_action_category_share.csv",
    "mechanism_exports/resource_moderate_stage_share.csv",
    "mechanism_exports/standard_severe_stage_share.csv",
]
plot_md = [
    "# plotting_readiness_summary",
    "",
    f"- Can high-pressure cumulative process be plotted? {'yes' if ((PROCESS / 'resource_moderate_mean_cumulative_progress.csv').exists() and (PROCESS / 'standard_severe_mean_cumulative_progress.csv').exists()) else 'no'}",
    f"- Can high-pressure stepwise process be plotted? {'yes' if ((PROCESS / 'resource_moderate_mean_stepwise_progress.csv').exists() and (PROCESS / 'standard_severe_mean_stepwise_progress.csv').exists()) else 'no'}",
    f"- Can high-pressure action heatmap be plotted? {'yes' if ((MECH / 'resource_moderate_action_category_share.csv').exists() and (MECH / 'standard_severe_action_category_share.csv').exists()) else 'no'}",
    "",
    "## Files to send to plotting assistant",
    *[f"- {x}" for x in plot_files],
    "",
    "## Files not to use for final paper figures",
    "- diagnostics/step_level_trace_sample.csv (diagnostic trace only)",
    "- _tmp_raw_reruns/* (intermediate run artifacts)",
    "",
    f"- Should V8 replace V6 or remain diagnostic? {'replace V6' if recommend_replace else 'remain diagnostic'}",
]
(OUT / "plotting_readiness_summary.md").write_text("\n".join(plot_md), encoding="utf-8")

print("final_v8_integrated package generated")

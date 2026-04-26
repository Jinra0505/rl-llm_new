from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

SCENARIOS = ["resource_moderate", "standard_severe"]
SEEDS = [42, 43, 44]
METHODS = ["baseline_rl", "full_outer_loop", "single_shot_llm", "ablation_fixed_global"]

SCENARIO_CFG = {
    "resource_moderate": {"split_name": "benchmark_resource_constrained_presets", "severity": "moderate"},
    "standard_severe": {"split_name": "benchmark_eval_presets", "severity": "severe"},
}

OUT = Path("paper_final_trajectory_exports_v1")
RAW = OUT / "raw_reruns"
PROC = OUT / "process_exports"
OPT = OUT / "optional_exports"
DIAG = OUT / "diagnostics"
for p in [RAW, PROC, OPT, DIAG]:
    p.mkdir(parents=True, exist_ok=True)


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
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
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
        "stdout_tail": (getattr(proc, "stdout", "") or "")[-4000:],
        "stderr_tail": (getattr(proc, "stderr", "") or "")[-4000:],
        "timed_out": bool(timed_out),
        "out_path": str(out_path),
        "status": "ok" if (not timed_out and getattr(proc, "returncode", 1) == 0 and out_path.exists()) else "failed",
    }
    if out_path.exists():
        try:
            d = json.loads(out_path.read_text(encoding="utf-8"))
            rec["has_eval_episode_traces"] = isinstance(d, dict) and isinstance(d.get("eval_episode_traces"), list)
            rec["trace_rows"] = len(d.get("eval_episode_traces", [])) if isinstance(d, dict) else 0
        except Exception as exc:
            rec["status"] = "failed"
            rec["parse_error"] = str(exc)
    return rec


records: list[dict[str, Any]] = []
for scenario in SCENARIOS:
    v6 = pd.read_csv(f"paper_repair_results_final_v6_packaged/final_tables/{scenario}_per_seed.csv")
    methods_present = [m for m in METHODS if m in set(v6["method"].astype(str))]
    for method in methods_present:
        for seed in SEEDS:
            records.append(run_one(scenario, method, seed))

pd.DataFrame(records).to_csv(DIAG / "rerun_status.csv", index=False)

trace_rows: list[dict[str, Any]] = []
summary_rows: list[dict[str, Any]] = []
for rec in records:
    if rec.get("status") != "ok":
        continue
    out_path = Path(str(rec["out_path"]))
    if not out_path.exists():
        continue
    data = json.loads(out_path.read_text(encoding="utf-8"))
    traces = data.get("eval_episode_traces", []) if isinstance(data, dict) else []
    if not isinstance(traces, list) or not traces:
        continue
    scenario = str(rec["scenario"])
    method = str(rec["method"])
    seed = int(rec["seed"])
    for t in traces:
        a = int(t.get("action", -1))
        trace_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "seed": seed,
                "episode_id": int(t.get("episode_id", 0)),
                "step": int(t.get("step", 0)),
                "action": a,
                "action_category": action_category(a),
                "stage": str(t.get("stage", "unknown")),
                "progress_delta": float(t.get("progress_delta", 0.0)),
                "cumulative_progress": float(t.get("cumulative_progress", 0.0)),
                "critical_load_recovery_ratio": float(t.get("critical_load_recovery_ratio", 0.0)),
                "communication_recovery_ratio": float(t.get("communication_recovery_ratio", 0.0)),
                "power_recovery_ratio": float(t.get("power_recovery_ratio", 0.0)),
                "road_recovery_ratio": float(t.get("road_recovery_ratio", 0.0)),
                "constraint_violation": int(bool(t.get("constraint_violation", False))),
                "invalid_action": int(bool(t.get("invalid_action", False))),
                "wait_hold_usage": int(a == 14),
            }
        )
    summary_rows.append(
        {
            "scenario": scenario,
            "method": method,
            "seed": seed,
            "selection_score": float(data.get("selection_score", math.nan)),
            "critical_load_recovery_ratio": float(data.get("critical_load_recovery_ratio", math.nan)),
            "communication_recovery_ratio": float(data.get("communication_recovery_ratio", math.nan)),
            "power_recovery_ratio": float(data.get("power_recovery_ratio", math.nan)),
            "road_recovery_ratio": float(data.get("road_recovery_ratio", math.nan)),
            "constraint_violation_rate_eval": float(data.get("constraint_violation_rate_eval", math.nan)),
            "invalid_action_rate_eval": float(data.get("invalid_action_rate_eval", data.get("invalid_action_rate", math.nan))),
            "wait_hold_usage_eval": float(data.get("wait_hold_usage_eval", data.get("wait_hold_usage", math.nan))),
            "mean_progress_delta_eval": float(data.get("mean_progress_delta_eval", data.get("mean_progress_delta", math.nan))),
            "eval_success_rate": float(data.get("eval_success_rate", data.get("success_rate", math.nan))),
        }
    )

if trace_rows:
    trace_df = pd.DataFrame(trace_rows)
    trace_df.to_csv(OUT / "step_level_trajectories_long.csv", index=False)
else:
    trace_df = pd.DataFrame(columns=[
        "scenario","method","seed","episode_id","step","action","action_category","stage","progress_delta","cumulative_progress",
        "critical_load_recovery_ratio","communication_recovery_ratio","power_recovery_ratio","road_recovery_ratio",
        "constraint_violation","invalid_action","wait_hold_usage"
    ])
    trace_df.to_csv(OUT / "step_level_trajectories_long.csv", index=False)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "reproduced_summary_metrics.csv", index=False)


def write_or_empty(path: Path, df: pd.DataFrame, cols: list[str]) -> None:
    if df.empty:
        pd.DataFrame(columns=cols).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


for scenario in SCENARIOS:
    sc = trace_df[trace_df["scenario"] == scenario].copy()
    if sc.empty:
        write_or_empty(PROC / f"{scenario}_mean_cumulative_progress.csv", pd.DataFrame(), ["scenario", "method", "step", "mean_cumulative_progress", "std_cumulative_progress", "n_seeds"])
        write_or_empty(PROC / f"{scenario}_mean_stepwise_progress.csv", pd.DataFrame(), ["scenario", "method", "step", "mean_progress_delta", "std_progress_delta", "n_seeds"])
        write_or_empty(PROC / f"{scenario}_action_category_share.csv", pd.DataFrame(), ["scenario", "method", "action_category", "mean_usage_share", "n_seeds"])
        write_or_empty(PROC / f"{scenario}_stage_share.csv", pd.DataFrame(), ["scenario", "method", "stage", "mean_usage_share", "n_seeds"])
        write_or_empty(OPT / f"{scenario}_layer_recovery_by_step.csv", pd.DataFrame(), ["scenario", "method", "step", "layer", "mean_recovery_ratio", "std_recovery_ratio", "n_seeds"])
        write_or_empty(OPT / f"{scenario}_safety_by_step.csv", pd.DataFrame(), ["scenario", "method", "step", "mean_constraint_violation", "std_constraint_violation", "mean_invalid_action", "std_invalid_action", "mean_wait_hold_usage", "std_wait_hold_usage", "n_seeds"])
        continue

    seed_step = sc.groupby(["scenario", "method", "seed", "step"], as_index=False).agg(
        cumulative_progress=("cumulative_progress", "mean"),
        progress_delta=("progress_delta", "mean"),
        critical_load_recovery_ratio=("critical_load_recovery_ratio", "mean"),
        communication_recovery_ratio=("communication_recovery_ratio", "mean"),
        power_recovery_ratio=("power_recovery_ratio", "mean"),
        road_recovery_ratio=("road_recovery_ratio", "mean"),
        constraint_violation=("constraint_violation", "mean"),
        invalid_action=("invalid_action", "mean"),
        wait_hold_usage=("wait_hold_usage", "mean"),
    )

    cum = seed_step.groupby(["scenario", "method", "step"], as_index=False).agg(
        mean_cumulative_progress=("cumulative_progress", "mean"),
        std_cumulative_progress=("cumulative_progress", "std"),
        n_seeds=("seed", "nunique"),
    ).fillna({"std_cumulative_progress": 0.0})
    write_or_empty(PROC / f"{scenario}_mean_cumulative_progress.csv", cum, ["scenario", "method", "step", "mean_cumulative_progress", "std_cumulative_progress", "n_seeds"])

    stepwise = seed_step.groupby(["scenario", "method", "step"], as_index=False).agg(
        mean_progress_delta=("progress_delta", "mean"),
        std_progress_delta=("progress_delta", "std"),
        n_seeds=("seed", "nunique"),
    ).fillna({"std_progress_delta": 0.0})
    write_or_empty(PROC / f"{scenario}_mean_stepwise_progress.csv", stepwise, ["scenario", "method", "step", "mean_progress_delta", "std_progress_delta", "n_seeds"])

    act_seed = sc.groupby(["scenario", "method", "seed", "action_category"], as_index=False).size().rename(columns={"size": "count"})
    tot = sc.groupby(["scenario", "method", "seed"], as_index=False).size().rename(columns={"size": "total"})
    act_seed = act_seed.merge(tot, on=["scenario", "method", "seed"], how="left")
    act_seed["usage_share"] = act_seed["count"] / act_seed["total"].clip(lower=1)
    act = act_seed.groupby(["scenario", "method", "action_category"], as_index=False).agg(
        mean_usage_share=("usage_share", "mean"), n_seeds=("seed", "nunique")
    )
    write_or_empty(PROC / f"{scenario}_action_category_share.csv", act, ["scenario", "method", "action_category", "mean_usage_share", "n_seeds"])

    st_seed = sc.groupby(["scenario", "method", "seed", "stage"], as_index=False).size().rename(columns={"size": "count"})
    st_seed = st_seed.merge(tot, on=["scenario", "method", "seed"], how="left")
    st_seed["usage_share"] = st_seed["count"] / st_seed["total"].clip(lower=1)
    st = st_seed.groupby(["scenario", "method", "stage"], as_index=False).agg(
        mean_usage_share=("usage_share", "mean"), n_seeds=("seed", "nunique")
    )
    write_or_empty(PROC / f"{scenario}_stage_share.csv", st, ["scenario", "method", "stage", "mean_usage_share", "n_seeds"])

    layers_long = pd.concat(
        [
            seed_step[["scenario", "method", "seed", "step", "critical_load_recovery_ratio"]].rename(columns={"critical_load_recovery_ratio": "recovery_ratio"}).assign(layer="critical_load"),
            seed_step[["scenario", "method", "seed", "step", "communication_recovery_ratio"]].rename(columns={"communication_recovery_ratio": "recovery_ratio"}).assign(layer="communication"),
            seed_step[["scenario", "method", "seed", "step", "power_recovery_ratio"]].rename(columns={"power_recovery_ratio": "recovery_ratio"}).assign(layer="power"),
            seed_step[["scenario", "method", "seed", "step", "road_recovery_ratio"]].rename(columns={"road_recovery_ratio": "recovery_ratio"}).assign(layer="road"),
        ],
        ignore_index=True,
    )
    layers = layers_long.groupby(["scenario", "method", "step", "layer"], as_index=False).agg(
        mean_recovery_ratio=("recovery_ratio", "mean"),
        std_recovery_ratio=("recovery_ratio", "std"),
        n_seeds=("seed", "nunique"),
    ).fillna({"std_recovery_ratio": 0.0})
    write_or_empty(OPT / f"{scenario}_layer_recovery_by_step.csv", layers, ["scenario", "method", "step", "layer", "mean_recovery_ratio", "std_recovery_ratio", "n_seeds"])

    safety = seed_step.groupby(["scenario", "method", "step"], as_index=False).agg(
        mean_constraint_violation=("constraint_violation", "mean"),
        std_constraint_violation=("constraint_violation", "std"),
        mean_invalid_action=("invalid_action", "mean"),
        std_invalid_action=("invalid_action", "std"),
        mean_wait_hold_usage=("wait_hold_usage", "mean"),
        std_wait_hold_usage=("wait_hold_usage", "std"),
        n_seeds=("seed", "nunique"),
    ).fillna({
        "std_constraint_violation": 0.0,
        "std_invalid_action": 0.0,
        "std_wait_hold_usage": 0.0,
    })
    write_or_empty(OPT / f"{scenario}_safety_by_step.csv", safety, ["scenario", "method", "step", "mean_constraint_violation", "std_constraint_violation", "mean_invalid_action", "std_invalid_action", "mean_wait_hold_usage", "std_wait_hold_usage", "n_seeds"])

# compare with v6 final per-seed tables
cmp_rows: list[dict[str, Any]] = []
for scenario in SCENARIOS:
    v6 = pd.read_csv(f"paper_repair_results_final_v6_packaged/final_tables/{scenario}_per_seed.csv")
    if summary_df.empty:
        continue
    m = v6.merge(summary_df, on=["scenario", "method", "seed"], how="left", suffixes=("_v6", "_rerun"))
    for _, r in m.iterrows():
        for metric in [
            "selection_score",
            "critical_load_recovery_ratio",
            "communication_recovery_ratio",
            "power_recovery_ratio",
            "road_recovery_ratio",
            "constraint_violation_rate_eval",
            "invalid_action_rate_eval",
            "wait_hold_usage_eval",
            "mean_progress_delta_eval",
            "eval_success_rate",
        ]:
            v6_val = r.get(f"{metric}_v6", math.nan)
            rr_val = r.get(f"{metric}_rerun", math.nan)
            if pd.isna(v6_val) or pd.isna(rr_val):
                diff = math.nan
                material = True
            else:
                diff = float(rr_val) - float(v6_val)
                material = abs(diff) > 0.02
            cmp_rows.append({
                "scenario": r["scenario"],
                "method": r["method"],
                "seed": int(r["seed"]),
                "metric": metric,
                "v6_value": v6_val,
                "rerun_value": rr_val,
                "diff": diff,
                "material_mismatch": material,
            })

cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(DIAG / "summary_metric_comparison.csv", index=False)

files = sorted([p for p in OUT.rglob("*.csv") if p.is_file()])
manifest_rows: list[dict[str, Any]] = []
for p in files:
    rel = str(p.relative_to(OUT))
    d = pd.read_csv(p)
    manifest_rows.append(
        {
            "file_name": rel,
            "row_count": len(d),
            "created_successfully": True,
            "notes": "",
        }
    )
pd.DataFrame(manifest_rows).to_csv(OUT / "export_manifest.csv", index=False)

inventory_target = [
    "resource_moderate_mean_cumulative_progress.csv",
    "standard_severe_mean_cumulative_progress.csv",
    "resource_moderate_mean_stepwise_progress.csv",
    "standard_severe_mean_stepwise_progress.csv",
    "resource_moderate_layer_recovery_by_step.csv",
    "standard_severe_layer_recovery_by_step.csv",
    "resource_moderate_safety_by_step.csv",
    "standard_severe_safety_by_step.csv",
    "resource_moderate_action_category_share.csv",
    "standard_severe_action_category_share.csv",
    "resource_moderate_stage_share.csv",
    "standard_severe_stage_share.csv",
]
inv_rows = []
for fn in inventory_target:
    p = PROC / fn
    if not p.exists():
        p = OPT / fn
    exists = p.exists()
    rows = len(pd.read_csv(p)) if exists else 0
    inv_rows.append({"file_name": fn, "exists": bool(exists), "row_count": int(rows)})
pd.DataFrame(inv_rows).to_csv(OUT / "process_file_inventory.csv", index=False)

material_cnt = int(cmp_df["material_mismatch"].fillna(True).sum()) if not cmp_df.empty else 0
failed_runs = int(sum(1 for r in records if r.get("status") != "ok"))
trace_status = "diagnostic-only" if (material_cnt > 0 or failed_runs > 0) else "reproducible"

md = []
md.append("# trajectory_reproduction_check")
md.append("")
md.append("## Scope")
md.append("- Scenarios: resource_moderate, standard_severe")
md.append("- Methods requested: baseline_rl, full_outer_loop, single_shot_llm, ablation_fixed_global (where present in V6)")
md.append("- Seeds: 42, 43, 44")
md.append("")
md.append("## Rerun status")
md.append(f"- Total runs attempted: {len(records)}")
md.append(f"- Failed runs: {failed_runs}")
md.append(f"- Runs with captured eval_episode_traces: {sum(1 for r in records if r.get('trace_rows', 0) > 0)}")
md.append("")
md.append("## Comparison against V6")
md.append(f"- Material mismatches (|diff| > 0.02 or missing rerun metric): {material_cnt}")
md.append(f"- Export status: **{trace_status}**")
if trace_status == "diagnostic-only":
    md.append("- Final results were not replaced; trajectory exports are for diagnostics/plotting only.")
md.append("")
md.append("## Notes")
md.append("- This rerun is logging-only: no model/environment/reward/selection/protocol changes were introduced by this script.")
md.append("- Exact reproducibility may differ if remote LLM candidate generation is unavailable or non-deterministic.")

(OUT / "trajectory_reproduction_check.md").write_text("\n".join(md), encoding="utf-8")
print("done")

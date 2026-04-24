from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def export_process_data(selected_runs: list[dict[str, Any]], process_dir: Path, diagnostics_dir: Path) -> dict[str, Any]:
    rep_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    per_preset_rows: list[dict[str, Any]] = []

    round_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    llm_rows: list[dict[str, Any]] = []

    missing_outer_trace: list[str] = []

    for item in selected_runs:
        scenario, method, seed = item["scenario"], item["method"], item["seed"]
        data = _load_json(Path(item["path"]))

        # representative trace
        rep = data.get("representative_eval_trace", [])
        if isinstance(rep, list):
            for i, step in enumerate(rep):
                if not isinstance(step, dict):
                    continue
                rep_rows.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                        "step": i,
                        "action": step.get("action", ""),
                        "action_category": step.get("action_category", ""),
                        "progress_delta": _f(step.get("progress_delta", step.get("delta_progress", 0.0))),
                        "stage": step.get("stage", ""),
                        "invalid_action": int(bool(step.get("invalid_action", False))),
                        "invalid_reason": step.get("invalid_reason", ""),
                        "constraint_violation": int(bool(step.get("constraint_violation", False))),
                    }
                )

        traj = data.get("eval_trajectory_summary", {})
        if isinstance(traj, dict):
            traj_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "seed": seed,
                    "mean_steps": _f(traj.get("mean_steps", data.get("eval_max_steps", 0.0))),
                    "terminated_rate": _f(traj.get("terminated_rate", data.get("eval_terminated_count", 0.0))),
                    "truncated_rate": _f(traj.get("truncated_rate", data.get("eval_truncated_count", 0.0))),
                    "mean_invalid_action_rate": _f(data.get("invalid_action_rate_eval", data.get("invalid_action_rate", 0.0))),
                    "mean_constraint_violation_rate": _f(data.get("constraint_violation_rate_eval", 0.0)),
                    "mean_progress_delta": _f(data.get("mean_progress_delta_eval", data.get("mean_progress_delta", 0.0))),
                }
            )

        action_usage = data.get("action_usage", {})
        if isinstance(action_usage, dict):
            categories = data.get("action_category_usage", {}) if isinstance(data.get("action_category_usage"), dict) else {}
            for action, usage in action_usage.items():
                action_rows.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                        "action": action,
                        "action_category": categories.get(action, ""),
                        "usage_rate": _f(usage),
                    }
                )

        stage_dist = data.get("stage_distribution_eval", data.get("stage_distribution", {}))
        if isinstance(stage_dist, dict):
            for stage, usage in stage_dist.items():
                stage_rows.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                        "stage": stage,
                        "usage_rate": _f(usage),
                    }
                )

        resource_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "seed": seed,
                "mes_soc_end_mean": _f(data.get("mes_soc_end_mean", data.get("mes_soc_mean_end", 0.0))),
                "material_stock_end_mean": _f(data.get("material_stock_end_mean", data.get("material_stock_mean_end", 0.0))),
                "switching_capability_end_mean": _f(data.get("switching_capability_end_mean", data.get("switching_capability_mean_end", 0.0))),
                "crew_power_status_end_mean": _f(data.get("crew_power_status_end_mean", data.get("crew_power_status_mean_end", 0.0))),
                "crew_comm_status_end_mean": _f(data.get("crew_comm_status_end_mean", data.get("crew_comm_status_mean_end", 0.0))),
                "crew_road_status_end_mean": _f(data.get("crew_road_status_end_mean", data.get("crew_road_status_mean_end", 0.0))),
            }
        )

        for zone in ["A", "B", "C"]:
            for layer, key in [
                ("power", f"zone_{zone}_power_ratio"),
                ("communication", f"zone_{zone}_comm_ratio"),
                ("road", f"zone_{zone}_road_ratio"),
                ("critical_load", f"zone_{zone}_critical_load_ratio"),
            ]:
                zone_rows.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                        "zone": zone,
                        "layer": layer,
                        "recovery_ratio": _f(data.get(key, 0.0)),
                    }
                )

        for phase, key in [("train", "episode_rewards"), ("eval", "eval_rewards")]:
            vals = data.get(key, [])
            if isinstance(vals, list):
                for ep, rv in enumerate(vals):
                    reward_rows.append(
                        {
                            "scenario": scenario,
                            "method": method,
                            "seed": seed,
                            "phase": phase,
                            "episode": ep,
                            "reward": _f(rv),
                        }
                    )

        # per preset if present
        preset_rows = data.get("per_preset_metrics", [])
        if isinstance(preset_rows, list):
            for pr in preset_rows:
                if not isinstance(pr, dict):
                    continue
                per_preset_rows.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                        "preset_name": pr.get("preset_name", ""),
                        "preset_group": pr.get("preset_group", ""),
                        "split_name": pr.get("split_name", data.get("split_name", "")),
                        "selection_score": _f(pr.get("selection_score", data.get("selection_score", 0.0))),
                        "min_recovery_ratio": _f(pr.get("min_recovery_ratio", data.get("min_recovery_ratio", 0.0))),
                        "critical_load_recovery_ratio": _f(pr.get("critical_load_recovery_ratio", data.get("critical_load_recovery_ratio", 0.0))),
                        "constraint_violation_rate_eval": _f(pr.get("constraint_violation_rate_eval", data.get("constraint_violation_rate_eval", 0.0))),
                        "invalid_action_rate_eval": _f(pr.get("invalid_action_rate_eval", data.get("invalid_action_rate_eval", 0.0))),
                        "wait_hold_usage_eval": _f(pr.get("wait_hold_usage_eval", data.get("wait_hold_usage_eval", 0.0))),
                        "mean_progress_delta_eval": _f(pr.get("mean_progress_delta_eval", data.get("mean_progress_delta_eval", 0.0))),
                    }
                )

        artifact = data.get("artifact_run_dir", "")
        if artifact:
            rdir = Path(artifact)
            if rdir.exists():
                status_path = rdir / "run_status.json"
                if status_path.exists():
                    status = _load_json(status_path)
                    llm_rows.append(
                        {
                            "scenario": scenario,
                            "method": method,
                            "seed": seed,
                            "response_kind": "run_status",
                            "model": "",
                            "success": int(bool(status.get("completed", False) and not bool(status.get("failed", False)))),
                            "latency_sec": "",
                            "content_len": "",
                            "reasoning_content_len": "",
                            "error": status.get("error", ""),
                        }
                    )
                round_dirs = sorted([p for p in rdir.glob("round_*") if p.is_dir()])
                for round_dir in round_dirs:
                    rnum = int(round_dir.name.split("_")[-1]) if "_" in round_dir.name else 0
                    summary_path = round_dir / "summary.json"
                    if summary_path.exists():
                        s = _load_json(summary_path)
                        bc = s.get("best_candidate", {}) if isinstance(s.get("best_candidate"), dict) else {}
                        bm = bc.get("metrics", {}) if isinstance(bc.get("metrics"), dict) else {}
                        round_rows.append(
                            {
                                "scenario": scenario,
                                "method": method,
                                "seed": seed,
                                "round": rnum,
                                "selected_candidate_id": s.get("best_candidate_id", bc.get("candidate_id", "")),
                                "candidate_origin": bc.get("candidate_origin", ""),
                                "task_mode": s.get("selected_task", ""),
                                "phase_mode": s.get("planning", {}).get("phase_mode", "") if isinstance(s.get("planning"), dict) else "",
                                "selection_score": _f(bm.get("selection_score", s.get("best_value", 0.0))),
                                "min_recovery_ratio": _f(bm.get("min_recovery_ratio", s.get("min_recovery_ratio", 0.0))),
                                "critical_load_recovery_ratio": _f(bm.get("critical_load_recovery_ratio", s.get("critical_load_recovery_ratio", 0.0))),
                                "constraint_violation_rate_eval": _f(bm.get("constraint_violation_rate_eval", s.get("constraint_violation_rate_eval", 0.0))),
                                "invalid_action_rate_eval": _f(bm.get("invalid_action_rate_eval", bm.get("invalid_action_rate", s.get("invalid_action_rate_eval", 0.0)))),
                                "wait_hold_usage_eval": _f(bm.get("wait_hold_usage_eval", bm.get("wait_hold_usage", s.get("wait_hold_usage_eval", 0.0)))),
                            }
                        )
                    route_path = round_dir / "route.json"
                    if route_path.exists():
                        route = _load_json(route_path)
                        routing_rows.append(
                            {
                                "scenario": scenario,
                                "method": method,
                                "seed": seed,
                                "round": rnum,
                                "task_mode": route.get("task_mode", route.get("final_task", "")),
                                "confidence": _f(route.get("confidence", 0.0)),
                                "dominant_signal": route.get("dominant_signal", ""),
                                "competing_signal": route.get("competing_signal", ""),
                                "reason": route.get("reason", ""),
                                "source": route.get("source", ""),
                            }
                        )

                    for cpath in sorted(round_dir.glob("r*/candidate_record.json")):
                        c = _load_json(cpath)
                        cm = c.get("metrics", {}) if isinstance(c.get("metrics"), dict) else {}
                        candidate_rows.append(
                            {
                                "scenario": scenario,
                                "method": method,
                                "seed": seed,
                                "round": rnum,
                                "candidate_id": c.get("candidate_id", ""),
                                "candidate_origin": c.get("candidate_origin", ""),
                                "valid": int(bool(c.get("validation", {}).get("valid", True))) if isinstance(c.get("validation"), dict) else 1,
                                "selected": int(bool(c.get("selected", False))),
                                "rejected": int(bool(c.get("error") or c.get("probe_ok") is False)),
                                "rejection_reasons": "|".join(c.get("probe_reject_reasons", []) or []),
                                "selection_score": _f(cm.get("selection_score", 0.0)),
                                "min_recovery_ratio": _f(cm.get("min_recovery_ratio", 0.0)),
                                "critical_load_recovery_ratio": _f(cm.get("critical_load_recovery_ratio", 0.0)),
                                "constraint_violation_rate_eval": _f(cm.get("constraint_violation_rate_eval", 0.0)),
                                "invalid_action_rate_eval": _f(cm.get("invalid_action_rate_eval", cm.get("invalid_action_rate", 0.0))),
                                "wait_hold_usage_eval": _f(cm.get("wait_hold_usage_eval", cm.get("wait_hold_usage", 0.0))),
                            }
                        )

                    llm_log = round_dir / "llm_call_log.json"
                    if llm_log.exists():
                        calls = _load_json(llm_log)
                        if isinstance(calls, list):
                            for rec in calls:
                                if not isinstance(rec, dict):
                                    continue
                                llm_rows.append(
                                    {
                                        "scenario": scenario,
                                        "method": method,
                                        "seed": seed,
                                        "response_kind": rec.get("response_kind", ""),
                                        "model": rec.get("model", ""),
                                        "success": int(bool(rec.get("success", False))),
                                        "latency_sec": _f(rec.get("latency_sec", 0.0)),
                                        "content_len": rec.get("content_len", ""),
                                        "reasoning_content_len": rec.get("reasoning_content_len", ""),
                                        "error": rec.get("error", ""),
                                    }
                                )
            else:
                missing_outer_trace.append(str(rdir))

    _write_csv(
        process_dir / "representative_eval_trace_long.csv",
        rep_rows,
        ["scenario", "method", "seed", "step", "action", "action_category", "progress_delta", "stage", "invalid_action", "invalid_reason", "constraint_violation"],
    )
    _write_csv(
        process_dir / "eval_trajectory_summary.csv",
        traj_rows,
        ["scenario", "method", "seed", "mean_steps", "terminated_rate", "truncated_rate", "mean_invalid_action_rate", "mean_constraint_violation_rate", "mean_progress_delta"],
    )
    _write_csv(process_dir / "action_usage_long.csv", action_rows, ["scenario", "method", "seed", "action", "action_category", "usage_rate"])
    _write_csv(process_dir / "stage_distribution_long.csv", stage_rows, ["scenario", "method", "seed", "stage", "usage_rate"])
    _write_csv(
        process_dir / "resource_end_summary.csv",
        resource_rows,
        ["scenario", "method", "seed", "mes_soc_end_mean", "material_stock_end_mean", "switching_capability_end_mean", "crew_power_status_end_mean", "crew_comm_status_end_mean", "crew_road_status_end_mean"],
    )
    _write_csv(process_dir / "zone_layer_recovery_long.csv", zone_rows, ["scenario", "method", "seed", "zone", "layer", "recovery_ratio"])
    _write_csv(process_dir / "reward_curves_long.csv", reward_rows, ["scenario", "method", "seed", "phase", "episode", "reward"])

    if per_preset_rows:
        _write_csv(
            process_dir / "per_preset_metrics.csv",
            per_preset_rows,
            [
                "scenario",
                "method",
                "seed",
                "preset_name",
                "preset_group",
                "split_name",
                "selection_score",
                "min_recovery_ratio",
                "critical_load_recovery_ratio",
                "constraint_violation_rate_eval",
                "invalid_action_rate_eval",
                "wait_hold_usage_eval",
                "mean_progress_delta_eval",
            ],
        )
    else:
        (diagnostics_dir / "per_preset_metrics_unavailable.md").write_text(
            "Per-preset metrics were not found in selected result JSONs.\n"
            "Future work: in train_rl.py, persist per-eval-episode / per-preset records\n"
            "(preset_group, preset_name, split_name, and the key eval metrics) into output JSON.",
            encoding="utf-8",
        )

    _write_csv(
        process_dir / "outer_loop_round_summary.csv",
        round_rows,
        [
            "scenario",
            "method",
            "seed",
            "round",
            "selected_candidate_id",
            "candidate_origin",
            "task_mode",
            "phase_mode",
            "selection_score",
            "min_recovery_ratio",
            "critical_load_recovery_ratio",
            "constraint_violation_rate_eval",
            "invalid_action_rate_eval",
            "wait_hold_usage_eval",
        ],
    )
    _write_csv(
        process_dir / "candidate_selection_trace.csv",
        candidate_rows,
        [
            "scenario",
            "method",
            "seed",
            "round",
            "candidate_id",
            "candidate_origin",
            "valid",
            "selected",
            "rejected",
            "rejection_reasons",
            "selection_score",
            "min_recovery_ratio",
            "critical_load_recovery_ratio",
            "constraint_violation_rate_eval",
            "invalid_action_rate_eval",
            "wait_hold_usage_eval",
        ],
    )
    _write_csv(
        process_dir / "routing_trace.csv",
        routing_rows,
        ["scenario", "method", "seed", "round", "task_mode", "confidence", "dominant_signal", "competing_signal", "reason", "source"],
    )
    _write_csv(
        process_dir / "llm_call_summary.csv",
        llm_rows,
        ["scenario", "method", "seed", "response_kind", "model", "success", "latency_sec", "content_len", "reasoning_content_len", "error"],
    )

    unavailable_msgs = []
    if not round_rows:
        unavailable_msgs.append("No round_*/summary.json data found for selected runs.")
    if not candidate_rows:
        unavailable_msgs.append("No candidate_record.json files found for selected runs.")
    if not routing_rows:
        unavailable_msgs.append("No route.json files found for selected runs.")
    if not llm_rows:
        unavailable_msgs.append("No llm_call_log.json or run_status.json files found for selected runs.")
    if missing_outer_trace:
        unavailable_msgs.append("Missing artifact_run_dir paths:\n- " + "\n- ".join(sorted(set(missing_outer_trace))))
    if unavailable_msgs:
        (diagnostics_dir / "outer_loop_traces_unavailable.md").write_text(
            "\n\n".join(unavailable_msgs)
            + "\n\nFuture reruns should keep artifact_run_dir trees and optional llm_call_log.json enabled.",
            encoding="utf-8",
        )

    return {
        "representative_eval_trace_rows": len(rep_rows),
        "eval_trajectory_rows": len(traj_rows),
        "action_usage_rows": len(action_rows),
        "stage_distribution_rows": len(stage_rows),
        "resource_end_rows": len(resource_rows),
        "zone_layer_rows": len(zone_rows),
        "reward_rows": len(reward_rows),
        "per_preset_rows": len(per_preset_rows),
        "outer_loop_round_rows": len(round_rows),
        "candidate_trace_rows": len(candidate_rows),
        "routing_rows": len(routing_rows),
        "llm_rows": len(llm_rows),
    }


if __name__ == "__main__":
    raise SystemExit("Use export_process_data.export_process_data(...) from a pipeline script.")

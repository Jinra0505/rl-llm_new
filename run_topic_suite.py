from __future__ import annotations

import copy
import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

from run_benchmark_eval import DEFAULT_CFG

SCENARIOS: dict[str, dict[str, str]] = {
    "standard_moderate": {
        "split_name": "benchmark_eval_presets",
        "severity": "moderate",
        "eval_budget": "completion_budget_eval",
    },
    "standard_severe": {
        "split_name": "benchmark_eval_presets",
        "severity": "severe",
        "eval_budget": "completion_budget_eval",
    },
    "resource_moderate": {
        "split_name": "benchmark_resource_constrained_presets",
        "severity": "moderate",
        "eval_budget": "completion_budget_eval",
    },
}

METHODS = ["baseline_rl", "single_shot_llm", "full_outer_loop", "ablation_fixed_global"]
SEEDS = [42, 43, 44]
METRICS = [
    "selection_score",
    "min_recovery_ratio",
    "critical_load_recovery_ratio",
    "constraint_violation_rate_eval",
    "invalid_action_rate_eval",
    "wait_hold_usage_eval",
    "safety_capacity_index",
]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return {"mean": float(m), "std": 0.0}
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return {"mean": float(m), "std": float(math.sqrt(max(0.0, var)))}


def _safety_capacity_index(d: dict[str, Any]) -> float:
    crit = _safe_float(d.get("critical_load_recovery_ratio", 0.0))
    min_rec = _safe_float(d.get("min_recovery_ratio", 0.0))
    inv = _safe_float(d.get("invalid_action_rate_eval", d.get("invalid_action_rate", 0.0)))
    vio = _safe_float(d.get("constraint_violation_rate_eval", 0.0))
    return 0.35 * crit + 0.35 * min_rec + 0.15 * (1.0 - inv) + 0.15 * (1.0 - vio)


def ensure_topic_config(path: Path) -> Path:
    if path.exists():
        return path
    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg["benchmark"]["split_name"] = "benchmark_eval_presets"
    cfg["benchmark"]["fixed_severity"] = "moderate"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def run_one(scenario: str, method: str, seed: int, out_path: Path, cfg_path: Path) -> tuple[dict[str, Any], str]:
    spec = SCENARIOS[scenario]
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
        spec["split_name"],
        "--severity",
        spec["severity"],
        "--eval-budget",
        spec["eval_budget"],
        "--config",
        str(cfg_path),
        "--out",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"run failed: {' '.join(cmd)}\\nstdout=\\n{proc.stdout[-1500:]}\\nstderr=\\n{proc.stderr[-3000:]}")
    if not out_path.exists():
        raise RuntimeError(f"missing output json: {out_path}")
    data = json.loads(out_path.read_text(encoding="utf-8"))
    data["safety_capacity_index"] = _safety_capacity_index(data)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data, " ".join(cmd)


def aggregate(all_runs: dict[str, dict[str, dict[int, dict[str, Any]]]]) -> dict[str, Any]:
    by_scenario: dict[str, Any] = {}
    for scenario, methods in all_runs.items():
        by_scenario[scenario] = {"methods": {}}
        for method, seeded in methods.items():
            rows = [seeded[s] for s in SEEDS if s in seeded]
            metric_stats = {m: _mean_std([_safe_float(r.get(m, 0.0)) for r in rows]) for m in METRICS}
            by_scenario[scenario]["methods"][method] = {
                "n": len(rows),
                "completed": int(sum(1 for r in rows if bool(r.get("completed", True)))),
                "failed": int(sum(1 for r in rows if bool(r.get("failed", False)))),
                "metrics": metric_stats,
            }
    return {
        "scenarios": by_scenario,
    }


def detect_issues(all_runs: dict[str, dict[str, dict[int, dict[str, Any]]]], summary: dict[str, Any]) -> dict[str, Any]:
    hard_failures: list[str] = []
    severe: list[str] = []
    affected: set[tuple[str, str]] = set()
    llm_methods = ["single_shot_llm", "full_outer_loop", "ablation_fixed_global"]

    # hard failures: missing/crashed
    for scenario, methods in all_runs.items():
        for method in METHODS:
            for seed in SEEDS:
                if seed not in methods.get(method, {}):
                    hard_failures.append(f"missing:{scenario}:{method}:seed{seed}")
                    affected.add((scenario, method))
                    continue
                row = methods[method][seed]
                if bool(row.get("failed", False)):
                    hard_failures.append(f"failed:{scenario}:{method}:seed{seed}")
                    affected.add((scenario, method))

    if "scenarios" not in summary:
        hard_failures.append("summary_aggregation_failure")

    for scenario in ["standard_moderate", "resource_moderate", "standard_severe"]:
        bound = 0.10 if scenario == "standard_severe" else 0.05
        for method in llm_methods:
            ms = summary["scenarios"].get(scenario, {}).get("methods", {}).get(method, {})
            inv = _safe_float(ms.get("metrics", {}).get("invalid_action_rate_eval", {}).get("mean", 0.0))
            vio = _safe_float(ms.get("metrics", {}).get("constraint_violation_rate_eval", {}).get("mean", 0.0))
            if inv > bound:
                severe.append(f"high_invalid:{scenario}:{method}:{inv:.4f}>{bound}")
                affected.add((scenario, method))
            if vio > bound:
                severe.append(f"high_violation:{scenario}:{method}:{vio:.4f}>{bound}")
                affected.add((scenario, method))

    lower_min_count = 0
    for scenario in SCENARIOS:
        bm = summary["scenarios"].get(scenario, {}).get("methods", {}).get("baseline_rl", {})
        fm = summary["scenarios"].get(scenario, {}).get("methods", {}).get("full_outer_loop", {})
        b_min = _safe_float(bm.get("metrics", {}).get("min_recovery_ratio", {}).get("mean", 0.0))
        f_min = _safe_float(fm.get("metrics", {}).get("min_recovery_ratio", {}).get("mean", 0.0))
        if f_min < (b_min - 0.03):
            lower_min_count += 1
    if lower_min_count >= 2:
        severe.append("full_outer_loop_min_recovery_below_baseline_by_gt0.03_in_2plus_scenarios")
        for scenario in SCENARIOS:
            affected.add((scenario, "full_outer_loop"))

    for scenario in SCENARIOS:
        bm = summary["scenarios"].get(scenario, {}).get("methods", {}).get("baseline_rl", {})
        fm = summary["scenarios"].get(scenario, {}).get("methods", {}).get("full_outer_loop", {})
        b_sel = _safe_float(bm.get("metrics", {}).get("selection_score", {}).get("mean", 0.0))
        b_crit = _safe_float(bm.get("metrics", {}).get("critical_load_recovery_ratio", {}).get("mean", 0.0))
        f_sel = _safe_float(fm.get("metrics", {}).get("selection_score", {}).get("mean", 0.0))
        f_crit = _safe_float(fm.get("metrics", {}).get("critical_load_recovery_ratio", {}).get("mean", 0.0))
        if f_sel < b_sel and f_crit < b_crit:
            severe.append(f"full_outer_loop_below_baseline_on_selection_and_critical:{scenario}")
            affected.add((scenario, "full_outer_loop"))

    # collapse / material_shortage spikes
    for scenario, methods in all_runs.items():
        for method in llm_methods:
            rows = [methods.get(method, {}).get(s, {}) for s in SEEDS]
            for r in rows:
                wait = _safe_float(r.get("wait_hold_usage_eval", r.get("wait_hold_usage", 0.0)))
                prog = _safe_float(r.get("mean_progress_delta_eval", r.get("mean_progress_delta", 0.0)))
                if wait > 0.60 and prog < 0.001:
                    severe.append(f"collapse_wait_low_progress:{scenario}:{method}")
                    affected.add((scenario, method))
                    break
            if scenario == "resource_moderate":
                mat_spike = 0
                for r in rows:
                    reasons = r.get("invalid_reason_counts_eval", {}) if isinstance(r.get("invalid_reason_counts_eval"), dict) else {}
                    mat_cnt = int(reasons.get("material_shortage", 0))
                    inv = _safe_float(r.get("invalid_action_rate_eval", r.get("invalid_action_rate", 0.0)))
                    if mat_cnt >= 10 and inv > 0.10:
                        mat_spike += 1
                if mat_spike >= 2:
                    severe.append(f"resource_material_shortage_spike:{scenario}:{method}")
                    affected.add((scenario, method))

    return {
        "hard_failures": hard_failures,
        "severe_issues": severe,
        "affected_pairs": sorted(list(affected)),
    }


def write_markdown(summary: dict[str, Any], issues: dict[str, Any], out_md: Path, repair_iterations: int) -> None:
    lines = [
        "# Topic Suite Summary",
        "",
        f"- Repair iterations used: {repair_iterations}",
        f"- Hard failures: {len(issues.get('hard_failures', []))}",
        f"- Severe issues: {len(issues.get('severe_issues', []))}",
        "",
    ]
    for scenario, blob in summary.get("scenarios", {}).items():
        lines.append(f"## {scenario}")
        lines.append("")
        lines.append("| method | selection_score mean±std | min_recovery mean±std | critical_load mean±std | violation mean±std | invalid mean±std | wait mean±std | SCI mean±std | completed | failed |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method in METHODS:
            m = blob.get("methods", {}).get(method, {})
            metrics = m.get("metrics", {})
            def fmt(key: str) -> str:
                st = metrics.get(key, {"mean": 0.0, "std": 0.0})
                return f"{_safe_float(st.get('mean')):.4f}±{_safe_float(st.get('std')):.4f}"
            lines.append(
                f"| {method} | {fmt('selection_score')} | {fmt('min_recovery_ratio')} | {fmt('critical_load_recovery_ratio')} | {fmt('constraint_violation_rate_eval')} | {fmt('invalid_action_rate_eval')} | {fmt('wait_hold_usage_eval')} | {fmt('safety_capacity_index')} | {int(m.get('completed', 0))} | {int(m.get('failed', 0))} |"
            )
        lines.append("")
    if issues.get("hard_failures"):
        lines.append("## Hard failures")
        for x in issues["hard_failures"]:
            lines.append(f"- {x}")
        lines.append("")
    if issues.get("severe_issues"):
        lines.append("## Severe issues")
        for x in issues["severe_issues"]:
            lines.append(f"- {x}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def apply_repair_profile(config_path: Path, iteration: int) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        cfg = copy.deepcopy(DEFAULT_CFG)
    sel = cfg.setdefault("selection", {})
    st = sel.setdefault("stability", {})
    if iteration == 1:
        st["max_violation_regression"] = 0.008
        st["max_invalid_regression"] = 0.008
        st["max_wait_usage_regression"] = 0.12
        st["recovery_floor_baseline"] = 0.58
        st["recovery_floor_penalty_weight"] = 2.4
    elif iteration == 2:
        st["max_violation_regression"] = 0.004
        st["max_invalid_regression"] = 0.004
        st["max_wait_usage_regression"] = 0.08
        st["recovery_floor_baseline"] = 0.60
        st["recovery_floor_penalty_weight"] = 2.8
    else:
        st["max_violation_regression"] = 0.003
        st["max_invalid_regression"] = 0.003
        st["max_wait_usage_regression"] = 0.06
        st["recovery_floor_baseline"] = 0.62
        st["recovery_floor_penalty_weight"] = 3.0
    cfg.setdefault("training", {})["reward_mode"] = "engineered"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/topic_suite")
    args = parser.parse_args()

    root = Path(args.output_root)
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    cfg_path = ensure_topic_config(Path("config_topic_eval.yaml"))

    all_runs: dict[str, dict[str, dict[int, dict[str, Any]]]] = {s: {m: {} for m in METHODS} for s in SCENARIOS}
    commands_run: list[str] = []
    repair_iterations = 0

    def run_pairs(
        pairs: list[tuple[str, str]] | None = None,
        force_rerun_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        pairs_set = set(pairs or [(s, m) for s in SCENARIOS for m in METHODS])
        force_set = set(force_rerun_pairs or [])
        for scenario in SCENARIOS:
            scenario_dir = runs_root / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            for method in METHODS:
                if (scenario, method) not in pairs_set:
                    continue
                force_rerun = (scenario, method) in force_set
                for seed in SEEDS:
                    out = scenario_dir / f"{method}__seed{seed}.json"
                    if out.exists() and not force_rerun:
                        try:
                            data = json.loads(out.read_text(encoding="utf-8"))
                            data["safety_capacity_index"] = _safety_capacity_index(data)
                            all_runs[scenario][method][seed] = data
                            continue
                        except Exception:
                            pass
                    print(f"[topic_suite] running scenario={scenario} method={method} seed={seed}", flush=True)
                    data, cmd_str = run_one(scenario, method, seed, out, cfg_path)
                    all_runs[scenario][method][seed] = data
                    commands_run.append(cmd_str)

    run_pairs()
    summary = aggregate(all_runs)
    issues = detect_issues(all_runs, summary)

    max_repairs = 3
    while (issues.get("hard_failures") or issues.get("severe_issues")) and repair_iterations < max_repairs:
        repair_iterations += 1
        apply_repair_profile(cfg_path, repair_iterations)
        affected = [tuple(x) for x in issues.get("affected_pairs", [])] if issues.get("affected_pairs") else None
        run_pairs(affected, force_rerun_pairs=affected)
        summary = aggregate(all_runs)
        issues = detect_issues(all_runs, summary)

    final_payload = {
        "scenarios": summary.get("scenarios", {}),
        "issues": issues,
        "repair_iterations": repair_iterations,
        "config_path": str(cfg_path),
        "commands_run": commands_run,
    }
    (root / "topic_summary.json").write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    write_markdown(summary, issues, root / "topic_summary.md", repair_iterations)
    print(json.dumps({
        "topic_summary_json": str(root / "topic_summary.json"),
        "topic_summary_md": str(root / "topic_summary.md"),
        "repair_iterations": repair_iterations,
        "hard_failures": len(issues.get("hard_failures", [])),
        "severe_issues": len(issues.get("severe_issues", [])),
    }, indent=2))


if __name__ == "__main__":
    main()

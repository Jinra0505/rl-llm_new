from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class Condition:
    mode: str
    reward_mode: str

    @property
    def tag(self) -> str:
        return f"{self.mode}__{self.reward_mode}"


DEFAULT_CONDITIONS: tuple[Condition, ...] = (
    Condition(mode="baseline_rl", reward_mode="clean"),
    Condition(mode="baseline_rl", reward_mode="engineered"),
    Condition(mode="single_shot_llm", reward_mode="clean"),
    Condition(mode="single_shot_llm", reward_mode="engineered"),
    Condition(mode="full_outer_loop", reward_mode="clean"),
    Condition(mode="full_outer_loop", reward_mode="engineered"),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_eval(
    *,
    condition: Condition,
    seed: int,
    config: str,
    split_name: str,
    eval_budget: str,
    out_path: Path,
) -> dict[str, Any]:
    cmd = [
        "python3",
        "run_benchmark_eval.py",
        "--mode",
        condition.mode,
        "--reward-mode",
        condition.reward_mode,
        "--seed",
        str(seed),
        "--split-name",
        split_name,
        "--eval-budget",
        eval_budget,
        "--out",
        str(out_path),
    ]
    if config:
        cmd.extend(["--config", config])

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    if stdout:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            pass
    return _read_json(out_path)


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "selection_score": float(payload.get("selection_score", 0.0)),
        "min_recovery_ratio": float(payload.get("min_recovery_ratio", 0.0)),
        "constraint_violation_rate_eval": float(payload.get("constraint_violation_rate_eval", 0.0)),
        "invalid_action_rate_eval": float(payload.get("invalid_action_rate_eval", 0.0)),
        "wait_hold_usage_eval": float(payload.get("wait_hold_usage_eval", 0.0)),
        "eval_success_rate": float(payload.get("eval_success_rate", payload.get("success_rate", 0.0))),
        "eval_terminated_count": float(payload.get("eval_terminated_count", 0.0)),
        "eval_truncated_count": float(payload.get("eval_truncated_count", 0.0)),
        "completion_window_entries": float(payload.get("completion_window_entries", 0.0)),
        "late_finish_action_share_eval": float(payload.get("late_finish_action_share_eval", 0.0)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "selection_score",
        "min_recovery_ratio",
        "constraint_violation_rate_eval",
        "invalid_action_rate_eval",
        "wait_hold_usage_eval",
        "eval_success_rate",
        "eval_terminated_count",
        "eval_truncated_count",
        "completion_window_entries",
        "late_finish_action_share_eval",
    ]
    return {f"mean_{k}": mean(float(r[k]) for r in rows) if rows else 0.0 for k in keys}


def _select_winner(summary: dict[str, dict[str, float]]) -> str:
    def sort_key(tag: str) -> tuple[float, float, float]:
        rec = summary[tag]
        return (
            rec["mean_selection_score"],
            rec["mean_min_recovery_ratio"],
            -rec["mean_constraint_violation_rate_eval"],
        )

    return sorted(summary.keys(), key=sort_key, reverse=True)[0]


def _evaluate_acceptance(*, split_name: str, eval_budget: str, summary: dict[str, dict[str, float]], winner: str) -> dict[str, Any]:
    full_eng = summary.get("full_outer_loop__engineered", {})
    full_clean = summary.get("full_outer_loop__clean", {})
    single_eng = summary.get("single_shot_llm__engineered", {})

    tie_break = {
        "selection_score_delta": float(full_eng.get("mean_selection_score", 0.0) - single_eng.get("mean_selection_score", 0.0)),
        "min_recovery_ratio_delta": float(full_eng.get("mean_min_recovery_ratio", 0.0) - single_eng.get("mean_min_recovery_ratio", 0.0)),
        "completion_window_entries_delta": float(full_eng.get("mean_completion_window_entries", 0.0) - single_eng.get("mean_completion_window_entries", 0.0)),
        "late_finish_action_share_delta": float(
            full_eng.get("mean_late_finish_action_share_eval", 0.0) - single_eng.get("mean_late_finish_action_share_eval", 0.0)
        ),
    }
    main_full_vs_single_distinct = any(abs(v) >= 1e-3 for v in tie_break.values())

    completion_profile = {
        "completion_window_entries_gain": float(
            full_eng.get("mean_completion_window_entries", 0.0) - full_clean.get("mean_completion_window_entries", 0.0)
        ),
        "late_finish_action_share_gain": float(
            full_eng.get("mean_late_finish_action_share_eval", 0.0) - full_clean.get("mean_late_finish_action_share_eval", 0.0)
        ),
        "terminated_minus_truncated_gain": float(
            (full_eng.get("mean_eval_terminated_count", 0.0) - full_eng.get("mean_eval_truncated_count", 0.0))
            - (full_clean.get("mean_eval_terminated_count", 0.0) - full_clean.get("mean_eval_truncated_count", 0.0))
        ),
    }
    main_completion_profile_ok = (
        completion_profile["completion_window_entries_gain"] >= 0.0
        and completion_profile["late_finish_action_share_gain"] >= 0.0
        and completion_profile["terminated_minus_truncated_gain"] >= 0.0
    )
    main_safety_ok = (
        float(full_eng.get("mean_constraint_violation_rate_eval", 1.0)) <= 0.03
        and float(full_eng.get("mean_invalid_action_rate_eval", 1.0)) <= 0.03
    )

    resource_checks = {
        "wait_hold_usage_ok": float(full_eng.get("mean_wait_hold_usage_eval", 1.0)) <= 0.25,
        "completion_window_ok": float(full_eng.get("mean_completion_window_entries", 0.0)) >= 1.0,
        "min_recovery_ok": float(full_eng.get("mean_min_recovery_ratio", 0.0)) >= 0.60,
        "constraint_zero_ok": float(full_eng.get("mean_constraint_violation_rate_eval", 1.0)) == 0.0,
        "invalid_zero_ok": float(full_eng.get("mean_invalid_action_rate_eval", 1.0)) == 0.0,
    }
    resource_completion_ok = sum(1 for ok in resource_checks.values() if ok) >= 3

    overall_accept = False
    if split_name == "benchmark_eval_presets" and eval_budget == "completion_budget_eval":
        overall_accept = main_full_vs_single_distinct and main_completion_profile_ok and main_safety_ok
    elif split_name == "benchmark_resource_constrained_presets" and eval_budget == "completion_budget_eval":
        overall_accept = resource_completion_ok

    return {
        "winner": winner,
        "main_full_vs_single_distinct": bool(main_full_vs_single_distinct),
        "main_full_vs_single_detail": tie_break,
        "main_completion_profile_ok": bool(main_completion_profile_ok),
        "main_completion_profile_detail": completion_profile,
        "main_safety_ok": bool(main_safety_ok),
        "resource_completion_ok": bool(resource_completion_ok),
        "resource_completion_detail": {"passed_count": sum(1 for ok in resource_checks.values() if ok), **resource_checks},
        "overall_accept": bool(overall_accept),
    }


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_iterations: list[dict[str, Any]] = []
    seeds = [int(s) for s in args.seeds]

    for iteration in range(args.max_iterations + 1):
        iteration_root = out_root / f"iteration_{iteration}"
        iteration_root.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict[str, Any]] = []
        per_condition: dict[str, list[dict[str, Any]]] = {}

        for cond in DEFAULT_CONDITIONS:
            tag = cond.tag
            per_condition[tag] = []
            for seed in seeds:
                out_path = iteration_root / f"{tag}__seed{seed}.json"
                if args.resume and out_path.exists():
                    payload = _read_json(out_path)
                else:
                    payload = _run_eval(
                        condition=cond,
                        seed=seed,
                        config=args.config,
                        split_name=args.split_name,
                        eval_budget=args.eval_budget,
                        out_path=out_path,
                    )
                    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                metrics = _extract_metrics(payload)
                row = {
                    "iteration": iteration,
                    "condition": tag,
                    "seed": seed,
                    "path": str(out_path),
                    **metrics,
                }
                per_condition[tag].append(row)
                all_rows.append(row)

        summary = {tag: _aggregate(rows) for tag, rows in per_condition.items()}
        winner = _select_winner(summary)
        check_detail = _evaluate_acceptance(
            split_name=args.split_name,
            eval_budget=args.eval_budget,
            summary=summary,
            winner=winner,
        )
        passed = bool(check_detail.get("overall_accept", False))

        diag = {
            "iteration": iteration,
            "winner": winner,
            "summary": summary,
            "threshold_check": {
                "passed": passed,
                "detail": check_detail,
            },
            "rows": all_rows,
        }
        (iteration_root / "diagnosis.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
        all_iterations.append(diag)

        if passed:
            break

    final = {
        "split_name": args.split_name,
        "eval_budget": args.eval_budget,
        "seeds": seeds,
        "max_iterations": args.max_iterations,
        "iterations_ran": len(all_iterations),
        "iterations": all_iterations,
    }
    (out_root / "protocol_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a repeatable 6-condition x N-seed diagnosis protocol with optional rerun iterations "
            "until threshold checks pass."
        )
    )
    parser.add_argument("--config", default="config_outer_loop_real_small.yaml")
    parser.add_argument("--split-name", default="benchmark_resource_constrained_presets")
    parser.add_argument("--eval-budget", choices=["auto", "fixed_budget_eval", "completion_budget_eval"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--min-selection-delta", type=float, default=0.0)
    parser.add_argument("--max-violation-delta", type=float, default=0.02)
    parser.add_argument("--output-root", default="outputs/diagnosis_protocol")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run_protocol(args)
    print(json.dumps({
        "iterations_ran": report["iterations_ran"],
        "final_passed": bool(report["iterations"][-1]["threshold_check"]["passed"]),
        "report": str(Path(args.output_root) / "protocol_report.json"),
    }, indent=2))


if __name__ == "__main__":
    main()

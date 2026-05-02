from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from mock_recovery_env import ProjectRecoveryEnv
from train_rl import _valid_action_mask
from train_rl import run_training

DEFAULT_CFG: dict[str, Any] = {
    "llm": {
        "base_url": "https://api.deepseek.com",
        "model_chat": "deepseek-chat",
        "model_reasoner": "deepseek-reasoner",
        "timeout_seconds": 60,
        "max_retries": 2,
        "temperature": 0.3,
        "max_tokens": 1800,
    },
    "outer_loop": {"rounds": 3, "candidates_per_round": 2},
    "env": {"name": "project_recovery", "max_steps": 30},
    "scenario": {"severity": "moderate"},
    "benchmark": {
        "enabled": True,
        "mode": "suite",
        "split_name": "benchmark_eval_presets",
        "preset_group": "",
        "preset_name": "",
        "preset_jitter": 0.0,
        "fixed_severity": "moderate",
    },
    "evaluation": {
        "fixed_budget": {"enabled": True, "max_steps": 15},
        "completion_budget": {"enabled": True, "max_steps": 40},
    },
    "training": {
        "train_episodes": 20,
        "eval_episodes": 6,
        "gamma": 0.98,
        "batch_size": 64,
        "replay_size": 30000,
        "min_replay_size": 500,
        "train_freq": 1,
        "target_update_interval": 250,
        "learning_rate": 0.0008,
        "hidden_dim": 128,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 12000,
        "reward_mode": "engineered",
    },
    "selection": {"higher_is_better": True, "task_mode_metric_weights": {}},
    "paths": {
        "generated_dir": "generated",
        "outputs_dir": "outputs/benchmark_eval/full_outer_loop",
        "formal_baseline_result": "outputs/exp1_baseline_realcheck_v4.json",
        "formal_outer_loop_dir": "outputs/real_outer_loop_v4",
    },
}


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_cfg(config_path: str) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CFG)
    if not config_path:
        return cfg
    loaded = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        _deep_update(cfg, loaded)
    return cfg


def resolve_eval_budget(cfg: dict[str, Any], requested_mode: str) -> tuple[str, int]:
    ev = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}
    fixed = ev.get("fixed_budget", {}) if isinstance(ev.get("fixed_budget"), dict) else {}
    completion = ev.get("completion_budget", {}) if isinstance(ev.get("completion_budget"), dict) else {}
    env_steps = int(cfg.get("env", {}).get("max_steps", 15))

    if requested_mode == "completion_budget_eval":
        return "completion_budget_eval", int(completion.get("max_steps", max(env_steps, 40)))
    if requested_mode == "fixed_budget_eval":
        return "fixed_budget_eval", int(fixed.get("max_steps", min(env_steps, 15)))

    if bool(completion.get("enabled", False)):
        return "completion_budget_eval", int(completion.get("max_steps", max(env_steps, 40)))
    if bool(fixed.get("enabled", False)):
        return "fixed_budget_eval", int(fixed.get("max_steps", min(env_steps, 15)))
    return "fixed_budget_eval", env_steps


def resolve_train_horizon(cfg: dict[str, Any], eval_budget_mode: str) -> int:
    env_steps = int(cfg.get("env", {}).get("max_steps", 15))
    if eval_budget_mode == "completion_budget_eval":
        return max(30, env_steps)
    return env_steps


def build_reset_options(*, benchmark_mode: str, split_name: str, preset_group: str, preset_name: str, preset_jitter: float, severity: str) -> callable:
    def _resolver(phase: str, episode_idx: int) -> dict[str, Any]:
        phase_split = split_name
        if phase == "eval" and split_name == "benchmark_train_presets":
            phase_split = "benchmark_eval_presets"
        return {
            "benchmark_mode": benchmark_mode,
            "split_name": phase_split,
            "preset_group": preset_group,
            "preset_name": preset_name,
            "preset_index": int(episode_idx),
            "preset_jitter": float(preset_jitter),
            "severity": severity,
        }

    return _resolver


def run_baseline(seed: int, reward_mode: str, split_name: str, preset_group: str, preset_name: str, preset_jitter: float, severity: str, out_path: Path, cfg: dict[str, Any], eval_budget_mode: str, eval_max_steps: int) -> dict[str, Any]:
    dqn_cfg = dict(cfg["training"])
    dqn_cfg["reward_mode"] = reward_mode
    train_max_steps = int(resolve_train_horizon(cfg, eval_budget_mode))
    metrics = run_training(
        revise_module_path=Path("baseline_noop.py"),
        env_name=str(cfg["env"]["name"]),
        train_episodes=int(dqn_cfg.get("train_episodes", 20)),
        eval_episodes=int(dqn_cfg.get("eval_episodes", 6)),
        max_steps_per_episode=train_max_steps,
        train_max_steps_per_episode=train_max_steps,
        eval_max_steps_per_episode=int(eval_max_steps),
        gamma=float(dqn_cfg.get("gamma", 0.98)),
        task_mode="global_efficiency_priority",
        llm_mode="real",
        output_json_path=out_path,
        seed=seed,
        dqn_cfg=dqn_cfg,
        severity=severity,
        intrinsic_mode="off",
        intrinsic_scale=1.0,
        env_reset_options=build_reset_options(
            benchmark_mode="suite",
            split_name=split_name,
            preset_group=preset_group,
            preset_name=preset_name,
            preset_jitter=preset_jitter,
            severity=severity,
        ),
        eval_budget_mode=eval_budget_mode,
    )
    metrics["train_max_steps"] = train_max_steps
    metrics["eval_budget_mode"] = eval_budget_mode
    metrics["eval_max_steps"] = int(eval_max_steps)
    return metrics


def run_outer_pipeline(mode: str, seed: int, reward_mode: str, split_name: str, preset_group: str, preset_name: str, preset_jitter: float, severity: str, out_path: Path, cfg: dict[str, Any], eval_budget_mode: str, eval_max_steps: int) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg["training"]["reward_mode"] = reward_mode
    cfg["benchmark"]["split_name"] = split_name
    cfg["benchmark"]["preset_group"] = preset_group
    cfg["benchmark"]["preset_name"] = preset_name
    cfg["benchmark"]["preset_jitter"] = preset_jitter
    cfg["benchmark"]["fixed_severity"] = severity
    cfg["scenario"]["severity"] = severity
    cfg["env"]["max_steps"] = int(resolve_train_horizon(cfg, eval_budget_mode))
    run_root = out_path.parent / "outer_loop_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["outputs_dir"] = str(run_root)
    cfg["benchmark_runtime"] = {"eval_max_steps": int(eval_max_steps), "eval_budget_mode": eval_budget_mode}
    cfg_path = out_path.parent / f"tmp_cfg_{mode}_{split_name}_{eval_budget_mode}_seed{seed}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    outer_cfg = cfg.get("outer_loop", {}) if isinstance(cfg.get("outer_loop"), dict) else {}
    fixed_task_mode = ""
    if mode == "single_shot_llm":
        rounds = "1"
        candidates = "1"
    elif mode == "ablation_fixed_global":
        rounds = str(max(2, int(outer_cfg.get("rounds", 3))))
        candidates = str(max(1, int(outer_cfg.get("candidates_per_round", 2))))
        fixed_task_mode = "global_efficiency_priority"
    else:
        # Keep full outer-loop materially different from single-shot.
        rounds = "3"
        candidates = "2"
    before = {p.name for p in run_root.glob("run_*") if p.is_dir()}
    cmd = [
        "python3",
        "run_outer_loop.py",
        "--env",
        "project_recovery",
        "--llm-mode",
        "real",
        "--router-mode",
        "llm",
        "--reroute-each-round",
        "--rounds-override",
        rounds,
        "--candidates-override",
        candidates,
        "--base-seed",
        str(seed),
        "--config",
        str(cfg_path),
    ]
    if fixed_task_mode:
        cmd.extend(["--fixed-task-mode", fixed_task_mode])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        err_tail = (exc.stderr or "").strip()[-3000:]
        out_tail = (exc.stdout or "").strip()[-1000:]
        raise RuntimeError(
            "Outer-loop subprocess failed.\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout_tail:\n{out_tail}\n"
            f"stderr_tail:\n{err_tail}"
        ) from exc
    all_runs = sorted([p for p in run_root.glob("run_*") if p.is_dir()])
    candidate_runs = [p for p in all_runs if p.name not in before] or all_runs
    latest = None
    summary = None
    run_status = None
    for rdir in reversed(candidate_runs):
        status_path = rdir / "run_status.json"
        if not status_path.exists():
            continue
        rs = json.loads(status_path.read_text(encoding="utf-8"))
        round_dirs = sorted([p for p in rdir.glob("round_*") if p.is_dir()])
        if not round_dirs:
            continue
        summary_path = round_dirs[-1] / "summary.json"
        if not summary_path.exists():
            continue
        latest = rdir
        run_status = rs
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        break
    if latest is None or summary is None or run_status is None:
        raise RuntimeError("No completed outer-loop run with round summary found for benchmark eval.")
    best_candidate = summary.get("best_candidate", {}) if isinstance(summary.get("best_candidate"), dict) else {}
    selection_diag = summary.get("selection_diagnostics", {}) if isinstance(summary.get("selection_diagnostics"), dict) else {}
    metrics = dict(best_candidate.get("metrics", {}))
    metrics["completed"] = bool(run_status.get("completed", False))
    metrics["failed"] = bool(run_status.get("failed", True))
    metrics["artifact_run_dir"] = str(latest)
    metrics["eval_budget_mode"] = eval_budget_mode
    metrics["eval_max_steps"] = int(eval_max_steps)
    # Provenance / explainability fields for downstream diagnostics and paper tables.
    metrics["candidate_source"] = str(selection_diag.get("winner_source", summary.get("winner_source", "")))
    metrics["selected_candidate_id"] = str(best_candidate.get("candidate_id", summary.get("best_candidate_id", "")))
    metrics["fallback_used"] = bool(selection_diag.get("fallback_used", summary.get("fallback_used", False)))
    metrics["fallback_reason"] = str(selection_diag.get("fallback_reason", ""))
    validation_blob = best_candidate.get("validation", {}) if isinstance(best_candidate.get("validation"), dict) else {}
    metrics["validation_status"] = "valid" if bool(validation_blob.get("valid", True)) else "invalid"
    reject_blob = selection_diag.get("rejection_reasons", {})
    if isinstance(reject_blob, dict):
        metrics["rejection_reason"] = "|".join(sorted({reason for reasons in reject_blob.values() for reason in (reasons or [])}))
    else:
        metrics["rejection_reason"] = ""
    metrics["score_components"] = {
        "selection_score": float(metrics.get("selection_score", 0.0)),
        "stability_adjusted_selection_score": float(
            metrics.get("stability_adjusted_selection_score", metrics.get("selection_score", 0.0))
        ),
        "recovery_adjusted_selection_score": float(
            metrics.get("recovery_adjusted_selection_score", metrics.get("selection_score", 0.0))
        ),
        "round_delta_summary": selection_diag.get("round_delta_summary", {}),
    }
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _action_type_proxy_score(action: int, info: dict[str, Any], stage_frac: float) -> float:
    critical_def = max(0.0, 1.0 - float(info.get("critical_load_recovery_ratio", 0.0)))
    power_def = max(0.0, 1.0 - float(info.get("power_recovery_ratio", 0.0)))
    comm_def = max(0.0, 1.0 - float(info.get("communication_recovery_ratio", 0.0)))
    road_def = max(0.0, 1.0 - float(info.get("road_recovery_ratio", 0.0)))
    min_def = max(0.0, 1.0 - min(1.0 - power_def, 1.0 - comm_def, 1.0 - road_def, 1.0 - critical_def))
    weak_layer = str(info.get("weakest_layer", "0"))
    weak_zone = str(info.get("weakest_zone", "A"))
    zone_idx = {"A": 0, "B": 1, "C": 2}.get(weak_zone, 0)
    mes_soc = float(info.get("mes_soc", 1.0))
    switching = float(info.get("switching_capability", 0.0))
    backbone_comm = float(info.get("backbone_comm_ratio", 0.0))
    if action in {0, 1, 2}:
        return 1.2 * road_def + 0.5 * power_def + (0.3 if action == zone_idx else 0.0)
    if action in {3, 4, 5}:
        return 1.25 * power_def + 1.0 * critical_def + (0.25 if weak_layer == "0" else 0.0)
    if action in {6, 7, 8}:
        return 1.2 * comm_def + 0.4 * switching + (0.25 if weak_layer == "1" else 0.0)
    if action in {9, 10, 11}:
        soc_penalty = 0.8 * max(0.0, 0.2 - mes_soc)
        return 1.3 * critical_def + 0.8 * power_def - soc_penalty
    if action == 12:
        imbalance = float(abs(power_def - comm_def) + abs(comm_def - road_def))
        return 0.7 * backbone_comm + 0.8 * switching + 0.8 * imbalance
    if action == 13:
        mean_def = (critical_def + power_def + comm_def + road_def) / 4.0
        return 1.2 * min_def + 0.8 * mean_def + 0.35 * stage_frac
    if action == 14:
        return -0.08
    return -0.1


def run_rule_based_greedy(seed: int, split_name: str, preset_group: str, preset_name: str, preset_jitter: float, severity: str, out_path: Path, cfg: dict[str, Any], eval_budget_mode: str, eval_max_steps: int) -> dict[str, Any]:
    env = ProjectRecoveryEnv(max_steps=int(eval_max_steps), seed=seed, severity=severity)
    eval_episodes = int(cfg.get("training", {}).get("eval_episodes", 6))
    trace_rows: list[dict[str, Any]] = []
    per_ep_summary: list[dict[str, Any]] = []
    finals: list[dict[str, float]] = []
    tot_steps = tot_invalid = tot_violate = tot_wait = 0
    action_usage = {str(i): 0 for i in range(int(env.action_space.n))}
    for ep in range(eval_episodes):
        obs, info = env.reset(seed=seed + ep, options={"benchmark_mode": "suite", "split_name": split_name, "preset_group": preset_group, "preset_name": preset_name, "preset_index": int(ep), "preset_jitter": float(preset_jitter), "severity": severity})
        cumulative_progress = 0.0
        for step in range(int(eval_max_steps)):
            valid_mask = _valid_action_mask(int(env.action_space.n), info, eval_budget_mode=eval_budget_mode, phase_contract=None)
            feasible = [a for a in range(int(env.action_space.n)) if bool(valid_mask[a])]
            if not feasible:
                feasible = [14] if int(env.action_space.n) > 14 else [0]
            stage_frac = float(step + 1) / float(max(1, eval_max_steps))
            scores: dict[int, float] = {}
            for a in feasible:
                try:
                    env_clone = copy.deepcopy(env)
                    _, _, _, _, ninfo = env_clone.step(int(a))
                    delta_crit = float(ninfo.get("critical_load_recovery_ratio", 0.0)) - float(info.get("critical_load_recovery_ratio", 0.0))
                    delta_pow = float(ninfo.get("power_recovery_ratio", 0.0)) - float(info.get("power_recovery_ratio", 0.0))
                    delta_comm = float(ninfo.get("communication_recovery_ratio", 0.0)) - float(info.get("communication_recovery_ratio", 0.0))
                    delta_road = float(ninfo.get("road_recovery_ratio", 0.0)) - float(info.get("road_recovery_ratio", 0.0))
                    min_prev = min(float(info.get("power_recovery_ratio", 0.0)), float(info.get("communication_recovery_ratio", 0.0)), float(info.get("road_recovery_ratio", 0.0)), float(info.get("critical_load_recovery_ratio", 0.0)))
                    min_next = min(float(ninfo.get("power_recovery_ratio", 0.0)), float(ninfo.get("communication_recovery_ratio", 0.0)), float(ninfo.get("road_recovery_ratio", 0.0)), float(ninfo.get("critical_load_recovery_ratio", 0.0)))
                    delta_min = min_next - min_prev
                    delta_prog = float(ninfo.get("progress_delta", 0.0))
                    mat_use = max(0.0, float(info.get("material_stock", 0.0)) - float(ninfo.get("material_stock", 0.0)))
                    soc_use = max(0.0, float(info.get("mes_soc", 0.0)) - float(ninfo.get("mes_soc", 0.0)))
                    scores[a] = 3.0 * delta_crit + 1.8 * delta_min + 1.2 * delta_pow + 1.0 * delta_comm + 0.8 * delta_road + 2.0 * delta_prog - 0.6 * mat_use - 0.4 * soc_use
                except Exception:
                    scores[a] = _action_type_proxy_score(a, info, stage_frac)
            non_wait_scores = [v for k, v in scores.items() if k != 14]
            best_non_wait = max(non_wait_scores) if non_wait_scores else -1e9
            if 14 in feasible and (best_non_wait < 0.005):
                act = 14
            else:
                act = max(scores.items(), key=lambda kv: kv[1])[0]
            nobs, _, terminated, truncated, ninfo = env.step(int(act))
            action_usage[str(int(act))] += 1
            tot_steps += 1
            tot_invalid += int(bool(ninfo.get("invalid_action", False)))
            tot_violate += int(bool(ninfo.get("constraint_violation", False)))
            tot_wait += int(int(act) == 14)
            cumulative_progress += float(ninfo.get("progress_delta", 0.0))
            trace_rows.append({"episode_id": ep, "step": step, "action": int(act), "action_category": "", "stage": str(ninfo.get("stage", "unknown")), "progress_delta": float(ninfo.get("progress_delta", 0.0)), "cumulative_progress": float(cumulative_progress), "constraint_violation": bool(ninfo.get("constraint_violation", False)), "invalid_action": bool(ninfo.get("invalid_action", False)), "wait_hold_usage": bool(int(act) == 14), "critical_load_recovery_ratio": float(ninfo.get("critical_load_recovery_ratio", 0.0)), "communication_recovery_ratio": float(ninfo.get("communication_recovery_ratio", 0.0)), "power_recovery_ratio": float(ninfo.get("power_recovery_ratio", 0.0)), "road_recovery_ratio": float(ninfo.get("road_recovery_ratio", 0.0))})
            obs, info = nobs, ninfo
            if terminated or truncated:
                break
        per_ep_summary.append({"episode_id": ep, "final_cumulative_progress": float(cumulative_progress)})
        finals.append({k: float(info.get(k, 0.0)) for k in ["critical_load_recovery_ratio", "communication_recovery_ratio", "power_recovery_ratio", "road_recovery_ratio"]})
    crit = sum(x["critical_load_recovery_ratio"] for x in finals) / max(1, len(finals))
    comm = sum(x["communication_recovery_ratio"] for x in finals) / max(1, len(finals))
    power = sum(x["power_recovery_ratio"] for x in finals) / max(1, len(finals))
    road = sum(x["road_recovery_ratio"] for x in finals) / max(1, len(finals))
    min_rec = min(crit, comm, power, road)
    result = {"selection_score": float((crit + comm + power + road + min_rec) / 5.0 - 0.2 * (tot_invalid / max(1, tot_steps)) - 0.2 * (tot_violate / max(1, tot_steps))), "critical_load_recovery_ratio": crit, "min_recovery_ratio": min_rec, "power_recovery_ratio": power, "communication_recovery_ratio": comm, "road_recovery_ratio": road, "constraint_violation_rate_eval": float(tot_violate) / float(max(1, tot_steps)), "invalid_action_rate_eval": float(tot_invalid) / float(max(1, tot_steps)), "wait_hold_usage_eval": float(tot_wait) / float(max(1, tot_steps)), "mean_progress_delta_eval": float(sum(float(t.get("progress_delta", 0.0)) for t in trace_rows)) / float(max(1, tot_steps)), "cumulative_progress": float(sum(float(ep.get("final_cumulative_progress", 0.0)) for ep in per_ep_summary)) / float(max(1, len(per_ep_summary))), "eval_episode_traces": trace_rows, "per_episode_eval_summary": per_ep_summary, "action_usage": action_usage, "eval_budget_mode": eval_budget_mode, "eval_max_steps": int(eval_max_steps), "train_max_steps": int(eval_max_steps), "candidate_source": "rule_based_greedy", "selected_candidate_id": "rule_based_greedy", "fallback_used": False, "fallback_reason": "", "validation_status": "valid", "rejection_reason": "", "completed": True, "failed": False}
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run aligned benchmark evaluation for baseline/single/full pipelines.")
    parser.add_argument("--mode", choices=["baseline_rl", "rule_based_greedy", "greedy_feasible_restoration_policy", "single_shot_llm", "full_outer_loop", "ablation_fixed_global"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-mode", choices=["clean", "engineered"], default="engineered")
    parser.add_argument("--split-name", default="benchmark_eval_presets")
    parser.add_argument("--preset-group", default="")
    parser.add_argument("--preset-name", default="")
    parser.add_argument("--preset-jitter", type=float, default=0.0)
    parser.add_argument("--severity", choices=["mild", "moderate", "severe"], default="moderate")
    parser.add_argument("--config", default="")
    parser.add_argument("--eval-budget", choices=["auto", "fixed_budget_eval", "completion_budget_eval"], default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    eval_budget_mode, eval_max_steps = resolve_eval_budget(cfg, args.eval_budget)

    out_path = Path(args.out) if args.out else Path("outputs") / "benchmark_eval" / f"{args.mode}__{args.reward_mode}__{args.split_name}__{eval_budget_mode}__seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "baseline_rl":
        metrics = run_baseline(
            args.seed,
            args.reward_mode,
            args.split_name,
            args.preset_group,
            args.preset_name,
            args.preset_jitter,
            args.severity,
            out_path,
            cfg,
            eval_budget_mode,
            eval_max_steps,
        )
    elif args.mode in {"rule_based_greedy", "greedy_feasible_restoration_policy"}:
        metrics = run_rule_based_greedy(args.seed, args.split_name, args.preset_group, args.preset_name, args.preset_jitter, args.severity, out_path, cfg, eval_budget_mode, eval_max_steps)
    else:
        metrics = run_outer_pipeline(
            args.mode,
            args.seed,
            args.reward_mode,
            args.split_name,
            args.preset_group,
            args.preset_name,
            args.preset_jitter,
            args.severity,
            out_path,
            cfg,
            eval_budget_mode,
            eval_max_steps,
        )

    outer_cfg = cfg.get("outer_loop", {}) if isinstance(cfg.get("outer_loop"), dict) else {}
    summary_rounds = 1 if args.mode == "single_shot_llm" else max(2, int(outer_cfg.get("rounds", 2)))
    summary_candidates = 1 if args.mode == "single_shot_llm" else max(1, int(outer_cfg.get("candidates_per_round", 2)))
    summary = {
        "mode": args.mode,
        "seed": int(args.seed),
        "split_name": args.split_name,
        "reward_mode": args.reward_mode,
        "eval_budget_mode": eval_budget_mode,
        "train_max_steps": int(metrics.get("train_max_steps", cfg.get("env", {}).get("max_steps", eval_max_steps))),
        "eval_max_steps": int(eval_max_steps),
        "rounds": int(summary_rounds),
        "candidates_per_round": int(summary_candidates),
        "selection_score": float(metrics.get("selection_score", 0.0)),
        "min_recovery_ratio": float(metrics.get("min_recovery_ratio", 0.0)),
        "constraint_violation_rate_eval": float(metrics.get("constraint_violation_rate_eval", 0.0)),
        "invalid_action_rate_eval": float(metrics.get("invalid_action_rate_eval", metrics.get("invalid_action_rate", 0.0))),
        "lipschitz_mean": float(metrics.get("lipschitz_mean", 0.0)),
        "wait_hold_usage_eval": float(metrics.get("wait_hold_usage_eval", metrics.get("wait_hold_usage", 0.0))),
        "eval_success_rate": float(metrics.get("eval_success_rate", metrics.get("success_rate", 0.0))),
        "eval_terminated_count": int(metrics.get("eval_terminated_count", 0)),
        "eval_truncated_count": int(metrics.get("eval_truncated_count", 0)),
        "completion_window_entries": float(metrics.get("completion_window_entries", 0.0)),
        "late_finish_action_share_eval": float(metrics.get("late_finish_action_share_eval", 0.0)),
        "completed": bool(metrics.get("completed", True)),
        "failed": bool(metrics.get("failed", False)),
        "output_json": str(out_path),
        "artifact_run_dir": str(metrics.get("artifact_run_dir", "")),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

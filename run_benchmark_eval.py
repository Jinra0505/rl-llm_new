from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

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
    metrics = dict(summary.get("best_candidate", {}).get("metrics", {}))
    metrics["completed"] = bool(run_status.get("completed", False))
    metrics["failed"] = bool(run_status.get("failed", True))
    metrics["artifact_run_dir"] = str(latest)
    metrics["eval_budget_mode"] = eval_budget_mode
    metrics["eval_max_steps"] = int(eval_max_steps)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run aligned benchmark evaluation for baseline/single/full pipelines.")
    parser.add_argument("--mode", choices=["baseline_rl", "single_shot_llm", "full_outer_loop", "ablation_fixed_global"], required=True)
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

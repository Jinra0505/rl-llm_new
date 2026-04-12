from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_rl import run_training


DEFAULT_CFG: dict[str, Any] = {
    "env": {"name": "project_recovery", "max_steps": 40},
    "scenario": {"severity": "moderate"},
    "training": {
        "train_episodes": 30,
        "eval_episodes": 8,
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
    },
}


def build_reset_options(
    *,
    benchmark_mode: str,
    split_name: str,
    preset_group: str,
    preset_name: str,
    preset_jitter: float,
    severity: str,
) -> callable:
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


def resolve_module_path(mode: str) -> Path:
    if mode == "baseline_rl":
        return Path("baseline_noop.py")
    return Path("generated/static_task_modules/global_efficiency_priority_module.py")


def resolve_intrinsic_mode(mode: str) -> str:
    if mode == "baseline_rl":
        return "off"
    if mode == "state_only":
        return "state_only"
    return "full"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight benchmark evaluation on fixed preset splits.")
    parser.add_argument("--config", default="")
    parser.add_argument("--mode", choices=["baseline_rl", "state_only", "full_outer_loop"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-mode", choices=["clean", "engineered"], default="engineered")
    parser.add_argument("--split-name", default="benchmark_eval_presets")
    parser.add_argument("--preset-group", default="")
    parser.add_argument("--preset-name", default="")
    parser.add_argument("--preset-jitter", type=float, default=0.0)
    parser.add_argument("--severity", choices=["mild", "moderate", "severe"], default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cfg = json.loads(json.dumps(DEFAULT_CFG))
    severity = args.severity or str(cfg["scenario"]["severity"])
    dqn_cfg = dict(cfg["training"])
    dqn_cfg["reward_mode"] = args.reward_mode
    out_path = Path(args.out) if args.out else Path("outputs") / "benchmark_eval" / f"{args.mode}__{args.reward_mode}__seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = run_training(
        revise_module_path=resolve_module_path(args.mode),
        env_name=str(cfg["env"]["name"]),
        train_episodes=int(dqn_cfg.get("train_episodes", 30)),
        eval_episodes=int(dqn_cfg.get("eval_episodes", 8)),
        max_steps_per_episode=int(cfg["env"]["max_steps"]),
        gamma=float(dqn_cfg.get("gamma", 0.98)),
        task_mode="global_efficiency_priority",
        llm_mode="real",
        output_json_path=out_path,
        seed=int(args.seed),
        dqn_cfg=dqn_cfg,
        severity=severity,
        intrinsic_mode=resolve_intrinsic_mode(args.mode),
        intrinsic_scale=1.0,
        env_reset_options=build_reset_options(
            benchmark_mode="suite",
            split_name=args.split_name,
            preset_group=args.preset_group,
            preset_name=args.preset_name,
            preset_jitter=float(args.preset_jitter),
            severity=severity,
        ),
    )
    summary = {
        "mode": args.mode,
        "reward_mode": args.reward_mode,
        "seed": int(args.seed),
        "selection_score": float(metrics.get("selection_score", 0.0)),
        "eval_success_rate": float(metrics.get("eval_success_rate", 0.0)),
        "benchmark_mode": str(metrics.get("benchmark_mode", "")),
        "split_name": str(metrics.get("split_name", "")),
        "preset_names_used": metrics.get("preset_names_used", []),
        "preset_groups_used": metrics.get("preset_groups_used", []),
        "benchmark_severities_used": metrics.get("benchmark_severities_used", []),
        "output_json": str(out_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

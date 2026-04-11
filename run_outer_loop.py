from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from llm_client import LLMClient
from mock_recovery_env import ProjectRecoveryEnv
from prompts import CODEGEN_PROMPT, FEEDBACK_PROMPT, PLANNING_PROMPT, SYSTEM_PROMPT
from task_recognizer import ScenarioTaskRecognizer, summarize_trajectory
from train_rl import run_training

LOGGER = logging.getLogger(__name__)
TASK_MODE_ALLOWED = {
    "critical_load_priority",
    "restoration_capability_priority",
    "global_efficiency_priority",
}
ALLOWED_IMPORTS = {"numpy", "math", "__future__"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "input"}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _clear_directory_contents(target_dir: Path) -> None:
    if not target_dir.exists():
        return
    for child in target_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _prune_unused_artifacts(run_dir: Path) -> None:
    removable_names = {"planning_raw.txt", "prompt.txt", "raw_response.txt"}
    for path in run_dir.rglob("*"):
        if path.is_file() and path.name in removable_names:
            path.unlink()


def _write_artifact_manifest(run_dir: Path) -> None:
    files = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())
    payload = {"run_dir": str(run_dir), "artifact_count": len(files), "artifacts": files}
    (run_dir / "artifact_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_json_with_repair(raw: str) -> tuple[dict[str, Any], bool]:
    def _extract_json_code_block(text: str) -> str:
        marker = "```json"
        lo = text.lower()
        start = lo.find(marker)
        if start != -1:
            start = start + len(marker)
            end = lo.find("```", start)
            if end != -1:
                return text[start:end].strip()
        if text.strip().startswith("```"):
            lines = text.strip().splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return ""

    def _extract_first_json_object(text: str) -> str:
        start = text.find("{")
        if start == -1:
            return ""
        depth = 0
        in_str = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return ""

    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        block = _extract_json_code_block(raw)
        if block:
            try:
                return json.loads(block), True
            except json.JSONDecodeError:
                pass
        first_obj = _extract_first_json_object(raw)
        if first_obj:
            try:
                return json.loads(first_obj), True
            except json.JSONDecodeError:
                pass
    return {}, True


def _write_failure_artifacts(run_dir: Path, failed_stage: str, error: Exception, client: LLMClient) -> None:
    payload = {
        "failed_stage": failed_stage,
        "last_error": str(error),
        "llm_requested_mode": client.mode,
        "llm_effective_mode": client.effective_mode(),
        "real_llm_call_count": client.call_count,
    }
    (run_dir / "run_failure.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "llm_call_log.json").write_text(json.dumps(client.call_history, indent=2), encoding="utf-8")


def _semantic_validate_candidate(code: str, max_revised_dim: int | None = None) -> list[str]:
    errors: list[str] = []
    sandbox: dict[str, Any] = {}
    try:
        exec(compile(code, "<generated-semantic-check>", "exec"), sandbox, sandbox)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return [f"Semantic load error: {exc}"]

    revise_state = sandbox.get("revise_state")
    intrinsic_reward = sandbox.get("intrinsic_reward")
    if not callable(revise_state):
        errors.append("Semantic validation: revise_state is not callable")
        return errors
    if not callable(intrinsic_reward):
        errors.append("Semantic validation: intrinsic_reward is not callable")
        return errors

    base_state = np.linspace(0.1, 0.9, 24, dtype=float)
    synth_inputs = [
        (base_state.copy(), {"stage": "early", "weakest_zone": "A", "weakest_layer": "2", "constraint_violation": False}),
        (base_state[::-1].copy(), {"stage": "middle", "weakest_zone": "B", "weakest_layer": "1", "constraint_violation": False}),
        (np.clip(base_state * 0.5, 0.0, 1.0), {"stage": "late", "weakest_zone": "C", "weakest_layer": "0", "constraint_violation": True}),
    ]
    revised_lens: list[int] = []
    revised_samples: list[np.ndarray] = []
    max_abs_appended = 0.0
    for idx, (state, info) in enumerate(synth_inputs):
        try:
            rs = revise_state(state, info)
        except TypeError:
            rs = revise_state(state)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Semantic validation: revise_state failed on sample {idx}: {exc}")
            continue
        try:
            arr = np.asarray(rs, dtype=float)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Semantic validation: revise_state output is not numeric array-like on sample {idx}: {exc}")
            continue
        if arr.ndim != 1:
            errors.append(f"Semantic validation: revise_state output must be 1-D, got ndim={arr.ndim} on sample {idx}")
            continue
        if not np.isfinite(arr).all():
            errors.append(f"Semantic validation: revise_state output has NaN/inf on sample {idx}")
            continue
        if arr.shape[0] < state.shape[0]:
            errors.append(
                f"Semantic validation: revise_state output shorter than raw state on sample {idx} ({arr.shape[0]} < {state.shape[0]})"
            )
        if max_revised_dim is not None and arr.shape[0] > int(max_revised_dim):
            errors.append(
                f"Semantic validation: revise_state output exceeds max_revised_dim on sample {idx} ({arr.shape[0]} > {max_revised_dim})"
            )
        revised_lens.append(int(arr.shape[0]))
        if arr.shape[0] > state.shape[0]:
            appended = arr[state.shape[0] :]
            max_abs_appended = max(max_abs_appended, float(np.max(np.abs(appended))) if appended.size else 0.0)
        revised_samples.append(arr)

    if revised_lens and len(set(revised_lens)) != 1:
        errors.append(f"Semantic validation: revise_state output length is not stable across inputs: {revised_lens}")
    if revised_lens and min(revised_lens) < 24:
        errors.append(f"Semantic validation: revise_state output length must be >= 24, got {revised_lens}")
    if max_abs_appended > 5.0:
        errors.append(f"Semantic validation: appended revise_state features are too large (max_abs={max_abs_appended:.4f})")

    for idx, (state, info) in enumerate(synth_inputs):
        if idx >= len(revised_samples):
            break
        next_state = np.clip(state + 0.01, 0.0, 1.0)
        try:
            ir = intrinsic_reward(state, 3, next_state, info, revised_samples[idx])
        except TypeError:
            try:
                ir = intrinsic_reward(state, 3, next_state, info)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Semantic validation: intrinsic_reward failed on sample {idx}: {exc}")
                continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Semantic validation: intrinsic_reward failed on sample {idx}: {exc}")
            continue
        if isinstance(ir, (list, tuple, dict, np.ndarray)):
            errors.append(f"Semantic validation: intrinsic_reward must return scalar float on sample {idx}, got {type(ir)}")
            continue
        try:
            ir_f = float(ir)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Semantic validation: intrinsic_reward not castable to float on sample {idx}: {exc}")
            continue
        if not np.isfinite(ir_f):
            errors.append(f"Semantic validation: intrinsic_reward returned NaN/inf on sample {idx}")
            continue
        if abs(ir_f) > 5.0:
            errors.append(f"Semantic validation: intrinsic_reward magnitude too large on sample {idx}: {ir_f}")

    return errors


def validate_candidate_payload(payload: dict[str, Any], max_revised_dim: int | None = None) -> dict[str, Any]:
    errors: list[str] = []
    required = ["file_name", "rationale", "code", "expected_behavior"]
    for key in required:
        if key not in payload:
            errors.append(f"Missing key: {key}")

    file_name = str(payload.get("file_name", ""))
    code = str(payload.get("code", ""))
    normalized = {
        "file_name": file_name,
        "rationale": str(payload.get("rationale", "")),
        "code": code,
        "expected_behavior": str(payload.get("expected_behavior", "")),
    }

    if not file_name.endswith(".py"):
        errors.append("file_name must end with .py")
    if not code.strip():
        errors.append("code is empty")

    if code.strip():
        try:
            tree = ast.parse(code)
            compile(tree, "<generated>", "exec")
            fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            if "revise_state" not in fn_names:
                errors.append("revise_state not found")
            if "intrinsic_reward" not in fn_names:
                errors.append("intrinsic_reward not found")
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                            errors.append(f"Import not allowed: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    if root not in ALLOWED_IMPORTS:
                        errors.append(f"Import-from not allowed: {node.module}")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_CALLS:
                        errors.append(f"Forbidden call: {node.func.id}")
            intrinsic_nodes = [
                n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "intrinsic_reward"
            ]
            if intrinsic_nodes:
                intrinsic_fn = intrinsic_nodes[0]
                branch_count = sum(isinstance(n, (ast.If, ast.IfExp)) for n in ast.walk(intrinsic_fn))
                if branch_count > 8:
                    errors.append(
                        f"Semantic validation: intrinsic_reward is too branch-heavy ({branch_count} conditional branches > 8)."
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Code validation error: {exc}")

    if code.strip() and not errors:
        errors.extend(_semantic_validate_candidate(code, max_revised_dim=max_revised_dim))

    return {"valid": len(errors) == 0, "errors": errors, "normalized_payload": normalized}


def _action_category(action: int) -> str:
    if action in {0, 1, 2}:
        return "road"
    if action in {3, 4, 5}:
        return "power"
    if action in {6, 7, 8}:
        return "comm"
    if action in {9, 10, 11}:
        return "mes"
    if action == 12:
        return "feeder"
    if action == 14:
        return "wait"
    return "coordinated"


def _aggregate_action_category_distribution(action_usage: dict[str, Any]) -> dict[str, float]:
    cats = {"road": 0.0, "power": 0.0, "comm": 0.0, "mes": 0.0, "feeder": 0.0, "coordinated": 0.0, "wait": 0.0}
    for action_str, val in action_usage.items():
        try:
            action = int(action_str)
            cats[_action_category(action)] += float(val)
        except (TypeError, ValueError):
            continue
    total = sum(cats.values())
    if total > 0.0:
        return {k: v / total for k, v in cats.items()}
    return cats


def _load_revise_fn(module_path: Path | None):
    if not module_path or not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "revise_state", None)
    return fn if callable(fn) else None


def _call_revise(fn: Any, state: Any, info: dict[str, Any]) -> Any:
    if fn is None:
        return state
    try:
        return fn(state, info)
    except TypeError:
        return fn(state)


def _greedy_probe_rollout(env: ProjectRecoveryEnv, revise_fn: Any, horizon: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state, info = env.reset(seed=101)
    trajectory: list[dict[str, Any]] = []
    weakest_zone_freq = {"A": 0, "B": 0, "C": 0}
    for step_idx in range(horizon):
        _ = _call_revise(revise_fn, state, info)
        weakest_zone = str(info.get("weakest_zone", "A"))
        weakest_layer = str(info.get("weakest_layer", "0"))
        zone_to_idx = {"A": 0, "B": 1, "C": 2}
        zone_idx = zone_to_idx.get(weakest_zone, 0)
        weakest_zone_freq[weakest_zone] = weakest_zone_freq.get(weakest_zone, 0) + 1

        if bool(info.get("constraint_violation", False)):
            action = 13
        elif weakest_layer == "2":
            action = zone_idx
        elif weakest_layer == "1":
            action = 6 + zone_idx
        elif weakest_layer == "0":
            action = 3 + zone_idx
        elif float(info.get("mes_soc", 0.0)) > 0.2 and float(info.get("critical_load_shortfall", 1.0)) > 0.3:
            action = 9 + zone_idx
        else:
            action = 13

        next_state, _, terminated, truncated, info = env.step(action)
        trajectory.append({"step": step_idx, "action": action, "info": info})
        state = next_state
        if terminated or truncated:
            break
    return trajectory, weakest_zone_freq


def collect_routing_context(
    env_name: str,
    previous_metrics: dict[str, Any],
    cfg: dict[str, Any],
    previous_best_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if env_name not in {"project_recovery", "mock_recovery"}:
        raise ValueError("Supported env names: project_recovery or mock_recovery")

    enough_previous = all(
        k in previous_metrics
        for k in ["communication_recovery_ratio", "power_recovery_ratio", "road_recovery_ratio", "constraint_violation_rate_eval"]
    )
    if enough_previous:
        env_summary = {
            "communication_recovery_ratio": float(previous_metrics.get("communication_recovery_ratio", 0.0)),
            "power_recovery_ratio": float(previous_metrics.get("power_recovery_ratio", 0.0)),
            "road_recovery_ratio": float(previous_metrics.get("road_recovery_ratio", 0.0)),
            "critical_load_shortfall": float(max(0.0, 1.0 - float(previous_metrics.get("critical_load_recovery_ratio", 0.0)))),
            "backbone_comm_ratio": float(previous_metrics.get("backbone_comm_ratio", previous_metrics.get("communication_recovery_ratio", 0.0))),
            "backbone_power_ratio": float(previous_metrics.get("backbone_power_ratio", previous_metrics.get("power_recovery_ratio", 0.0))),
            "backbone_road_ratio": float(previous_metrics.get("backbone_road_ratio", previous_metrics.get("road_recovery_ratio", 0.0))),
            "material_stock": float(previous_metrics.get("material_stock_end_mean", previous_metrics.get("material_stock_mean_end", 1.0))),
            "weakest_zone": str(previous_metrics.get("weakest_zone", "A")),
            "weakest_layer": str(previous_metrics.get("weakest_layer", "0")),
            "constraint_violation_count": int(previous_metrics.get("constraint_violation_count", 0)),
        }
        trajectory_summary = {
            "mean_progress_delta": float(previous_metrics.get("mean_progress_delta_eval", previous_metrics.get("mean_progress_delta", 0.0))),
            "invalid_action_rate": float(previous_metrics.get("invalid_action_rate_eval", previous_metrics.get("invalid_action_rate", 0.0))),
            "constraint_violation_rate": float(previous_metrics.get("constraint_violation_rate_eval", previous_metrics.get("constraint_violation_rate", 0.0))),
            "stage_distribution": dict(previous_metrics.get("stage_distribution_eval", previous_metrics.get("stage_distribution", {}))),
            "action_category_distribution": _aggregate_action_category_distribution(dict(previous_metrics.get("action_usage", {}))),
            "weakest_zone_frequency": dict(previous_metrics.get("weakest_zone_frequency", {})),
            "source": "previous_metrics",
        }
    else:
        env = ProjectRecoveryEnv(
            max_steps=int(cfg["env"].get("max_steps", 60)),
            seed=101,
            severity=str(cfg.get("scenario", {}).get("severity", "moderate")),
            reward_weights=cfg.get("reward_weights", {}),
        )
        module_path = Path(str(previous_best_candidate.get("candidate_path", ""))) if previous_best_candidate else None
        revise_fn = _load_revise_fn(module_path)
        trajectory, weakest_zone_freq = _greedy_probe_rollout(env, revise_fn=revise_fn, horizon=12)
        last_info = trajectory[-1]["info"] if trajectory else {}
        env_summary = {
            "communication_recovery_ratio": float(last_info.get("communication_recovery_ratio", 0.0)),
            "power_recovery_ratio": float(last_info.get("power_recovery_ratio", 0.0)),
            "road_recovery_ratio": float(last_info.get("road_recovery_ratio", 0.0)),
            "critical_load_shortfall": float(last_info.get("critical_load_shortfall", 1.0)),
            "backbone_comm_ratio": float(last_info.get("backbone_comm_ratio", last_info.get("communication_recovery_ratio", 0.0))),
            "backbone_power_ratio": float(last_info.get("backbone_power_ratio", last_info.get("power_recovery_ratio", 0.0))),
            "backbone_road_ratio": float(last_info.get("backbone_road_ratio", last_info.get("road_recovery_ratio", 0.0))),
            "material_stock": float(last_info.get("material_stock", 1.0)),
            "weakest_zone": str(last_info.get("weakest_zone", "A")),
            "weakest_layer": str(last_info.get("weakest_layer", "0")),
            "constraint_violation_count": int(last_info.get("constraint_violation_count", 0)),
        }
        trajectory_summary = summarize_trajectory(trajectory)
        probe_action_usage: dict[str, float] = {}
        for item in trajectory:
            akey = str(item["action"])
            probe_action_usage[akey] = probe_action_usage.get(akey, 0.0) + 1.0
        trajectory_summary["action_category_distribution"] = _aggregate_action_category_distribution(
            probe_action_usage
        )
        trajectory_summary["weakest_zone_frequency"] = weakest_zone_freq
        trajectory_summary["source"] = "greedy_probe_rollout"

    return {
        "env_summary": env_summary,
        "trajectory_summary": trajectory_summary,
        "previous_metrics": previous_metrics,
    }


def build_feedback(
    best_candidate: dict[str, Any],
    score_metric: str,
    reference_metrics: dict[str, Any] | None = None,
    planning_summary: dict[str, Any] | None = None,
    previous_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = best_candidate.get("metrics", {})
    reference_metrics = reference_metrics or {}
    hints: list[str] = []
    if int(metrics.get("constraint_violation_count", 0)) > 5:
        hints.append("Constraint violations are frequent.")
    if float(metrics.get("critical_load_recovery_ratio", 0.0)) < 0.6:
        hints.append("Critical load recovery is still low.")
    if float(metrics.get("road_recovery_ratio", 0.0)) < 0.6:
        hints.append("Road restoration is lagging and may bottleneck repairs.")
    if not hints:
        hints.append("No major failure mode detected.")

    core_keys = [
        "success_rate",
        "critical_load_recovery_ratio",
        "communication_recovery_ratio",
        "power_recovery_ratio",
        "road_recovery_ratio",
        "constraint_violation_rate_eval",
        "invalid_action_rate_eval",
        "mean_progress_delta_eval",
        "wait_hold_usage_eval",
    ]
    candidate_core = {k: metrics.get(k, 0.0) for k in core_keys}
    baseline_core = {k: reference_metrics.get(k, 0.0) for k in core_keys}
    core_delta = {
        k: float(_safe_float(candidate_core.get(k, 0.0)) - _safe_float(baseline_core.get(k, 0.0)))
        for k in core_keys
    }
    lipschitz_keys = ["lipschitz_mean", "lipschitz_max", "lipschitz_min"]
    candidate_lipschitz = {k: _safe_float(metrics.get(k, 0.0)) for k in lipschitz_keys}
    baseline_lipschitz = {k: _safe_float(reference_metrics.get(k, 0.0)) for k in lipschitz_keys}
    lipschitz_delta = {k: float(candidate_lipschitz[k] - baseline_lipschitz[k]) for k in lipschitz_keys}
    candidate_lipschitz.update(
        {
            "top_unstable_dims": metrics.get("lipschitz_top_unstable_dims", [])[:3]
            if isinstance(metrics.get("lipschitz_top_unstable_dims"), list)
            else [],
            "top_stable_dims": metrics.get("lipschitz_top_stable_dims", [])[:3]
            if isinstance(metrics.get("lipschitz_top_stable_dims"), list)
            else [],
            "low_sample_episodes": int(_safe_float(metrics.get("lipschitz_low_sample_episodes", 0))),
        }
    )
    baseline_lipschitz.update(
        {
            "top_unstable_dims": reference_metrics.get("lipschitz_top_unstable_dims", [])[:3]
            if isinstance(reference_metrics.get("lipschitz_top_unstable_dims"), list)
            else [],
            "top_stable_dims": reference_metrics.get("lipschitz_top_stable_dims", [])[:3]
            if isinstance(reference_metrics.get("lipschitz_top_stable_dims"), list)
            else [],
        }
    )
    if candidate_lipschitz["lipschitz_mean"] > 1.0:
        hints.append("State-reward smoothness appears unstable (high Lipschitz mean).")

    return {
        "task_mode": str(best_candidate.get("task_mode", metrics.get("task_mode_used", ""))),
        "primary_score_metric": score_metric,
        "primary_score_value": metrics.get(score_metric, 0.0),
        "candidate_core_metrics": candidate_core,
        "baseline_core_metrics": baseline_core,
        "candidate_vs_baseline_delta": core_delta,
        "lipschitz_candidate_summary": candidate_lipschitz,
        "lipschitz_baseline_summary": baseline_lipschitz,
        "lipschitz_candidate_vs_baseline_delta": lipschitz_delta,
        "failure_mode_hints": hints,
        "has_violation": bool(_safe_float(metrics.get("constraint_violation_rate_eval", 0.0)) > 0.0),
        "has_invalid_action": bool(_safe_float(metrics.get("invalid_action_rate_eval", 0.0)) > 0.0),
        "no_improvement_vs_baseline": bool(all(abs(v) < 1e-6 or v <= 0.0 for v in core_delta.values())),
        "planning_summary": planning_summary or {},
        "previous_feedback_summary": _summarize_feedback(previous_feedback),
        "module_change_summary": {
            "file_name": best_candidate.get("candidate", {}).get("file_name", ""),
            "rationale": best_candidate.get("candidate", {}).get("rationale", ""),
            "expected_behavior": str(best_candidate.get("candidate", {}).get("expected_behavior", ""))[:300],
        },
    }


def _summarize_feedback(previous_feedback: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(previous_feedback, dict):
        return {}
    return {
        "improvement_focus": previous_feedback.get("improvement_focus", ""),
        "keep_signals": previous_feedback.get("keep_signals", [])[:5] if isinstance(previous_feedback.get("keep_signals"), list) else [],
        "avoid_patterns": previous_feedback.get("avoid_patterns", [])[:5] if isinstance(previous_feedback.get("avoid_patterns"), list) else [],
        "finish_strategy_adjustments": previous_feedback.get("finish_strategy_adjustments", ""),
    }


def build_planning_payload(route: dict[str, Any], routing_context: dict[str, Any], previous_feedback: dict[str, Any] | None = None) -> dict[str, Any]:
    env = routing_context.get("env_summary", {}) if isinstance(routing_context, dict) else {}
    traj = routing_context.get("trajectory_summary", {}) if isinstance(routing_context, dict) else {}
    return {
        "task_mode": str(route.get("task_mode", "global_efficiency_priority")),
        "stage": str(route.get("stage", "middle")),
        "route_reason": str(route.get("reason", "")),
        "weakest_layer": str(env.get("weakest_layer", "0")),
        "weakest_zone": str(env.get("weakest_zone", "A")),
        "critical_load_shortfall": float(env.get("critical_load_shortfall", 1.0)),
        "backbone_comm_ratio": float(env.get("backbone_comm_ratio", env.get("communication_recovery_ratio", 0.0))),
        "constraint_violation_rate": float(traj.get("constraint_violation_rate", 0.0)),
        "invalid_action_rate": float(traj.get("invalid_action_rate", 0.0)),
        "mean_progress_delta": float(traj.get("mean_progress_delta", 0.0)),
        "latest_feedback_summary": _summarize_feedback(previous_feedback),
    }


def select_best(results: list[dict[str, Any]], metric: str, higher_is_better: bool) -> dict[str, Any]:
    return sorted(results, key=lambda x: x.get(metric, 0.0), reverse=higher_is_better)[0]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _reference_metrics(previous_best: dict[str, Any] | None, cfg: dict[str, Any], outputs_root: Path) -> dict[str, Any]:
    if previous_best and isinstance(previous_best.get("metrics"), dict):
        return dict(previous_best["metrics"])
    baseline_path = Path(str(cfg.get("paths", {}).get("formal_baseline_result", outputs_root / "exp1_baseline_realcheck_v4.json")))
    if baseline_path.exists():
        try:
            return json.loads(baseline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def select_best_candidate(round_candidates: list[dict[str, Any]], reference_metrics: dict[str, Any], higher_is_better: bool) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected_ids: list[str] = []
    rejection_reasons: dict[str, list[str]] = {}

    ref_critical = _safe_float(reference_metrics.get("critical_load_recovery_ratio", 0.0))
    ref_progress = _safe_float(reference_metrics.get("mean_progress_delta_eval", reference_metrics.get("mean_progress_delta", 0.0)))
    ref_power = _safe_float(reference_metrics.get("power_recovery_ratio", 0.0))
    ref_comm = _safe_float(reference_metrics.get("communication_recovery_ratio", 0.0))
    ref_road = _safe_float(reference_metrics.get("road_recovery_ratio", 0.0))
    ref_avg_recovery = (ref_power + ref_comm + ref_road) / 3.0
    ref_violation = _safe_float(reference_metrics.get("constraint_violation_rate_eval", 1.0), default=1.0)

    for cand in round_candidates:
        cid = str(cand.get("candidate_id", "unknown"))
        metrics = cand.get("metrics", {})
        reasons: list[str] = []
        if not isinstance(metrics, dict) or "selection_score" not in metrics:
            reasons.append("missing_metrics")
        else:
            success = _safe_float(metrics.get("success_rate", 0.0))
            critical = _safe_float(metrics.get("critical_load_recovery_ratio", 0.0))
            progress = _safe_float(metrics.get("mean_progress_delta_eval", metrics.get("mean_progress_delta", 0.0)))
            power = _safe_float(metrics.get("power_recovery_ratio", 0.0))
            comm = _safe_float(metrics.get("communication_recovery_ratio", 0.0))
            road = _safe_float(metrics.get("road_recovery_ratio", 0.0))
            violation = _safe_float(metrics.get("constraint_violation_rate_eval", 1.0), default=1.0)
            material_end = _safe_float(metrics.get("material_stock_end_mean", metrics.get("material_stock_mean_end", 0.0)))
            invalid_rate = _safe_float(metrics.get("invalid_action_rate_eval", metrics.get("invalid_action_rate", 1.0)))
            wait_usage = _safe_float(metrics.get("wait_hold_usage_eval", metrics.get("wait_hold_usage", 0.0)))
            rep = metrics.get("representative_eval_summary", {}) if isinstance(metrics.get("representative_eval_summary"), dict) else {}
            final_progress_delta = _safe_float(rep.get("final_progress_delta", 0.0))
            final_stage = str(rep.get("final_stage", "unknown"))

            if success <= 0.0:
                if ref_critical > 0.0 and critical < (ref_critical - 0.08):
                    reasons.append("critical_recovery_too_low_vs_reference")
                if ref_progress > 0.0 and progress < (0.6 * ref_progress):
                    reasons.append("progress_delta_too_low_vs_reference")
                low_recovery_layers = sum(
                    [
                        power < (ref_power - 0.10) if ref_power > 0 else False,
                        comm < (ref_comm - 0.10) if ref_comm > 0 else False,
                        road < (ref_road - 0.10) if ref_road > 0 else False,
                    ]
                )
                if low_recovery_layers >= 2:
                    reasons.append("multi_layer_recovery_regression")
                if final_progress_delta == 0.0 and final_stage != "late":
                    reasons.append("no_final_progress_and_not_late_stage")
                avg_recovery = (power + comm + road) / 3.0
                if violation < ref_violation and avg_recovery < (ref_avg_recovery - 0.08):
                    reasons.append("violation_improved_but_recovery_collapsed")
            if material_end < 0.03 and (invalid_rate > 0.45 or progress < 0.001):
                reasons.append("resource_sustainability_collapse")
            if final_stage != "late" and progress < 0.0009 and critical < 0.55:
                reasons.append("not_finish_oriented_under_zero_success")
            if wait_usage > 0.42 and progress < 0.006:
                reasons.append("wait_hold_overuse_with_low_progress")

        if reasons:
            rejected_ids.append(cid)
            rejection_reasons[cid] = reasons
        else:
            accepted.append(cand)

    pool = accepted if accepted else [c for c in round_candidates if isinstance(c.get("metrics"), dict)]
    if not pool:
        raise RuntimeError("No candidate with metrics is available for selection.")
    all_zero_success = all(_safe_float(c.get("metrics", {}).get("success_rate", 0.0)) <= 0.0 for c in pool)
    if all_zero_success:
        def zero_success_rank_key(c: dict[str, Any]) -> tuple[float, float, float, float, float]:
            m = c.get("metrics", {})
            stage_close = _safe_float(m.get("mean_stage_indicator_eval", 0.0))
            critical = _safe_float(m.get("critical_load_recovery_ratio", 0.0))
            prog = _safe_float(m.get("mean_progress_delta_eval", m.get("mean_progress_delta", 0.0)))
            material = _safe_float(m.get("material_stock_end_mean", m.get("material_stock_mean_end", 0.0)))
            wait_usage = _safe_float(m.get("wait_hold_usage_eval", m.get("wait_hold_usage", 0.0)))
            violation = _safe_float(m.get("constraint_violation_rate_eval", 1.0), default=1.0)
            return (stage_close, critical, prog, material, -wait_usage, -violation)
        best_candidate = sorted(pool, key=zero_success_rank_key, reverse=True)[0]
        best_metrics = best_candidate["metrics"]
    else:
        best_metrics = select_best([c["metrics"] for c in pool], "selection_score", higher_is_better)
        best_candidate = next(c for c in pool if c["metrics"] is best_metrics)
    return {
        "best_candidate": best_candidate,
        "selection_diagnostics": {
            "selection_policy": "gate_then_score",
            "zero_success_policy_applied": all_zero_success,
            "accepted_ids": [str(c.get("candidate_id", "")) for c in accepted],
            "rejected_ids": rejected_ids,
            "rejection_reasons": rejection_reasons,
            "used_fallback_pool": len(accepted) == 0,
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="LLM outer loop for project-grade tri-layer recovery env.")
    parser.add_argument("--env", default="project_recovery")
    parser.add_argument("--llm-mode", choices=["real"], default="real")
    parser.add_argument("--router-mode", choices=["llm"], default="llm")
    parser.add_argument("--fixed-task-mode", default="")
    parser.add_argument("--reroute-each-round", action="store_true")
    parser.add_argument("--rounds-override", type=int, default=0)
    parser.add_argument("--candidates-override", type=int, default=0)
    parser.add_argument("--intrinsic-mode", choices=["off", "state_only", "full"], default="full")
    parser.add_argument("--intrinsic-scale", type=float, default=1.0)
    parser.add_argument("--disable-feedback", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if args.llm_mode != "real":
        raise RuntimeError("Formal run requires llm_mode=real.")
    if args.router_mode != "llm":
        raise RuntimeError("Formal run requires router_mode=llm.")
    if args.fixed_task_mode:
        raise RuntimeError("Formal run does not allow fixed-task-mode override.")

    cfg = load_yaml(Path(args.config))
    rounds = args.rounds_override or int(cfg["outer_loop"]["rounds"])
    candidates_per_round = args.candidates_override or int(cfg["outer_loop"]["candidates_per_round"])
    higher_is_better = bool(cfg["selection"].get("higher_is_better", True))

    generated_dir = Path(cfg["paths"]["generated_dir"])
    outputs_root = Path(cfg["paths"]["outputs_dir"])
    generated_dir.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    llm_cfg = cfg.get("llm", {})
    if llm_cfg.get("model_chat") and not os.getenv("DEEPSEEK_MODEL_CHAT"):
        os.environ["DEEPSEEK_MODEL_CHAT"] = str(llm_cfg["model_chat"])
    if llm_cfg.get("model_reasoner") and not os.getenv("DEEPSEEK_MODEL_REASONER"):
        os.environ["DEEPSEEK_MODEL_REASONER"] = str(llm_cfg["model_reasoner"])
    if llm_cfg.get("base_url") and not os.getenv("DEEPSEEK_BASE_URL"):
        os.environ["DEEPSEEK_BASE_URL"] = str(llm_cfg["base_url"])

    client = LLMClient(
        mode=args.llm_mode,
        timeout_seconds=int(cfg["llm"]["timeout_seconds"]),
        max_retries=int(cfg["llm"]["max_retries"]),
        temperature=float(cfg["llm"]["temperature"]),
        max_tokens=int(cfg["llm"]["max_tokens"]),
    )

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = outputs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        client.preflight_check()
    except Exception as exc:  # noqa: BLE001
        _write_failure_artifacts(run_dir, "preflight", exc, client)
        raise

    (run_dir / "run_snapshot.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "config": cfg,
                "llm_requested_mode": args.llm_mode,
                "llm_effective_mode": client.effective_mode(),
                "router_mode": args.router_mode,
                "api_provider": client.api_provider,
                "base_url": client.base_url,
                "chat_model": client.chat_model,
                "reasoner_model": client.reasoner_model,
                "api_key_present": bool(client.api_key),
                "preflight_ok": True,
                "preflight_chat_ok": client.preflight_chat_ok,
                "preflight_reasoner_ok": client.preflight_reasoner_ok,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    history: list[dict[str, Any]] = []
    route: dict[str, Any] = {}
    recognizer = ScenarioTaskRecognizer()

    for round_idx in range(rounds):
        previous_best = history[-1].get("best_candidate", {}) if history else None
        prev_metrics = previous_best.get("metrics", {}) if previous_best else {}
        routing_context = collect_routing_context(args.env, prev_metrics, cfg, previous_best_candidate=previous_best)

        previous_task = str(history[-1].get("selected_task", "")) if history else ""
        previous_round_failed = bool(float(prev_metrics.get("success_rate", 0.0)) <= 0.0) if previous_best else False
        try:
            route = recognizer.recognize_with_llm(
                client=client,
                system_prompt=SYSTEM_PROMPT,
                routing_context=routing_context,
                previous_task=previous_task,
                previous_round_failed=previous_round_failed,
            )
        except Exception as exc:  # noqa: BLE001
            _write_failure_artifacts(run_dir, "router", exc, client)
            raise
        if route["task_mode"] not in TASK_MODE_ALLOWED:
            raise RuntimeError(f"Router returned unsupported task mode in formal run: {route['task_mode']}")
        route["task_switched_vs_prev_round"] = bool(round_idx > 0 and route.get("task_mode") != previous_task)

        round_dir = run_dir / f"round_{round_idx+1}"
        round_dir.mkdir(parents=True, exist_ok=True)
        route["source"] = "llm"
        (round_dir / "route.json").write_text(json.dumps(route, indent=2), encoding="utf-8")
        planning_payload = build_planning_payload(
            route=route,
            routing_context=routing_context,
            previous_feedback=(history[-1].get("llm_feedback", {}) if history else None),
        )
        try:
            planning_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PLANNING_PROMPT + "\n\n" + json.dumps(planning_payload, indent=2)},
            ]
            planning_json = client.chat_json(planning_messages, response_kind="planning", sample_idx=round_idx)
            planning_raw = json.dumps(planning_json, ensure_ascii=False, indent=2)
            planning_repaired = False
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if ("response_kind=planning" in msg) and ("finish_reason=length" in msg):
                compressed_payload = dict(planning_payload)
                compressed_payload["route_reason"] = str(compressed_payload.get("route_reason", ""))[:240]
                compressed_payload["latest_feedback_summary"] = _summarize_feedback(
                    history[-1].get("llm_feedback", {}) if history else None
                )
                compressed_payload["compression_retry"] = True
                try:
                    planning_messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": PLANNING_PROMPT + "\n\n" + json.dumps(compressed_payload, indent=2)},
                    ]
                    planning_json = client.chat_json(
                        planning_messages,
                        response_kind="planning",
                        sample_idx=round_idx + 1000,
                    )
                    planning_payload = compressed_payload
                    planning_raw = json.dumps(planning_json, ensure_ascii=False, indent=2)
                    planning_repaired = False
                except Exception as retry_exc:  # noqa: BLE001
                    _write_failure_artifacts(run_dir, "planning", retry_exc, client)
                    raise
            else:
                _write_failure_artifacts(run_dir, "planning", exc, client)
                raise
        (round_dir / "planning_raw.txt").write_text(planning_raw, encoding="utf-8")
        (round_dir / "planning.json").write_text(
            json.dumps({"source": "llm", "payload": planning_payload, "planning": planning_json, "repaired_from_raw": planning_repaired}, indent=2),
            encoding="utf-8",
        )

        round_candidates: list[dict[str, Any]] = []
        for sample_idx in range(candidates_per_round):
            cid = f"r{round_idx+1}_c{sample_idx+1}"
            cdir = round_dir / cid
            cdir.mkdir(parents=True, exist_ok=True)

            prompt = CODEGEN_PROMPT.format(
                task_mode=route["task_mode"],
                stage=route["stage"],
                observation_schema=str(cfg["env"]),
                planning_json=json.dumps(planning_json, indent=2, ensure_ascii=False),
            )
            prompt += "\n\nReturn compact JSON and keep generated code concise (<= 45 lines)."
            prompt += (
                "\nCode must include explicit finish-oriented shaping: reward entering late stage, reward completion, "
                "penalize prolonged middle-stage tiny progress, and penalize resource collapse."
            )
            if history:
                prompt += "\n\nLatest feedback:\n" + json.dumps(history[-1].get("feedback_payload", {}), indent=2)
            raw = "{}"
            parsed: dict[str, Any] = {}
            repaired = True
            report: dict[str, Any] = {"valid": False, "errors": ["not attempted"], "normalized_payload": {}}
            for attempt in range(4):
                attempt_prompt = prompt
                if attempt > 0:
                    attempt_prompt += (
                        "\n\nPrevious output was invalid. Respond with ONLY one JSON object with keys: "
                        "file_name, rationale, code, expected_behavior."
                    )
                    attempt_prompt += (
                        "\nThe 'code' field must be a valid Python source string with escaped newlines, "
                        "and must parse successfully."
                    )
                    attempt_prompt += (
                        "\nRetry hard constraints:"
                        "\n- revise_state must include original 24D raw state (no compression/subset)."
                        "\n- revise_state output must be fixed-length and never shorter than 24."
                        "\n- prefer simple appended summary features over handcrafted compression."
                        "\n- intrinsic_reward must be smooth, conservative, mostly delta-based, and small magnitude."
                    )
                    attempt_prompt += "\nPrevious validation errors: " + json.dumps(report.get("errors", []), ensure_ascii=False)
                try:
                    raw = client.chat(
                        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": attempt_prompt}],
                        response_kind="codegen",
                        sample_idx=sample_idx + round_idx * 10 + attempt,
                    )
                except Exception as exc:  # noqa: BLE001
                    if attempt < 3:
                        continue
                    raise RuntimeError(f"Codegen call failed for {cid} under formal real LLM mode: {exc}") from exc
                parsed, repaired = parse_json_with_repair(raw)
                if not parsed and attempt == 3:
                    raise RuntimeError(f"Codegen JSON parse failed for {cid} under formal real LLM mode.")
                report = validate_candidate_payload(
                    parsed,
                    max_revised_dim=(
                        int(cfg.get("state_representation", {}).get("max_revised_dim"))
                        if cfg.get("state_representation", {}).get("max_revised_dim") is not None
                        else None
                    ),
                )
                report["repaired_from_raw"] = repaired
                if report["valid"]:
                    break

            if not report["valid"]:
                raise RuntimeError(
                    f"Candidate {cid} failed validation in formal real LLM mode; errors={report.get('errors', [])}"
                )

            (cdir / "prompt.txt").write_text(prompt, encoding="utf-8")
            (cdir / "raw_response.txt").write_text(raw, encoding="utf-8")
            (cdir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

            record = {"candidate_id": cid, "validation": report, "candidate": report["normalized_payload"]}
            if report["valid"]:
                fname = report["normalized_payload"]["file_name"]
                code = report["normalized_payload"]["code"]
                candidate_path = generated_dir / fname
                candidate_path.write_text(code, encoding="utf-8")
                (cdir / fname).write_text(code, encoding="utf-8")

                metrics = run_training(
                    revise_module_path=candidate_path,
                    env_name=args.env,
                    train_episodes=int(cfg["training"]["train_episodes"]),
                    eval_episodes=int(cfg["training"]["eval_episodes"]),
                    max_steps_per_episode=int(cfg["env"]["max_steps"]),
                    gamma=float(cfg["training"]["gamma"]),
                    task_mode=route["task_mode"],
                    llm_mode="real",
                    output_json_path=cdir / "training_result.json",
                    seed=42 + round_idx * 10 + sample_idx,
                    max_revised_dim=(int(cfg.get("state_representation", {}).get("max_revised_dim")) if cfg.get("state_representation", {}).get("max_revised_dim") is not None else None),
                    task_mode_metric_weights=cfg.get("selection", {}).get("task_mode_metric_weights", {}),
                    dqn_cfg=cfg.get("training", {}),
                    severity=str(cfg.get("scenario", {}).get("severity", "moderate")),
                    intrinsic_mode=args.intrinsic_mode,
                    intrinsic_scale=args.intrinsic_scale,
                )
                metrics["selected_task"] = route.get("task_mode")
                metrics["llm_effective_mode"] = client.effective_mode()
                metrics["router_mode"] = args.router_mode
                metrics["wait_hold_usage_eval"] = float(metrics.get("wait_hold_usage_eval", metrics.get("wait_hold_usage", 0.0)))
                record["metrics"] = metrics
                record["candidate_path"] = str(candidate_path)
                record["task_mode"] = route["task_mode"]
                record["route_source"] = "llm"
                record["selection_score"] = float(metrics.get("selection_score", 0.0))
                record["representative_eval_summary"] = dict(metrics.get("representative_eval_summary", {}))
                (cdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            else:
                record["metrics"] = {"selection_score": -1e9 if higher_is_better else 1e9, "success_rate": 0.0}
                record["error"] = "Validation failed"

            (cdir / "candidate_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            round_candidates.append(record)

        selection_result = select_best_candidate(
            round_candidates=round_candidates,
            reference_metrics=_reference_metrics(previous_best, cfg, outputs_root),
            higher_is_better=higher_is_better,
        )
        best_candidate = selection_result["best_candidate"]
        selection_diagnostics = selection_result["selection_diagnostics"]

        feedback_payload = build_feedback(
            best_candidate,
            "selection_score",
            reference_metrics=_reference_metrics(previous_best, cfg, outputs_root),
            planning_summary={
                "weakest_layer": planning_json.get("weakest_layer", ""),
                "weakest_zone": planning_json.get("weakest_zone", ""),
                "finishing_strategy": planning_json.get("finishing_strategy", ""),
            },
            previous_feedback=(history[-1].get("llm_feedback", {}) if history else None),
        )
        feedback_fallback_used = False
        feedback_primary_model = client.reasoner_model
        feedback_final_model = feedback_primary_model
        if args.disable_feedback:
            feedback_json = {
                "improvement_focus": ["feedback disabled"],
                "keep_signals": [],
                "avoid_patterns": [],
                "finish_strategy_adjustments": [],
                "confidence": 1.0,
            }
        else:
            feedback_json = {}
            feedback_error: Exception | None = None
            feedback_raw = ""
            try:
                feedback_raw = client.chat(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": FEEDBACK_PROMPT + "\n\n" + json.dumps(feedback_payload, indent=2)}],
                    response_kind="feedback",
                )
                feedback_json, _ = parse_json_with_repair(feedback_raw)
                if not feedback_json:
                    raise RuntimeError("Feedback stage JSON parse failed under primary attempt.")
            except Exception as exc:  # noqa: BLE001
                feedback_error = exc
            if not feedback_json:
                compressed_feedback_payload = {
                    "round_index": int(round_idx + 1),
                    "task_mode": feedback_payload.get("task_mode", ""),
                    "best_candidate_metrics": feedback_payload.get("candidate_core_metrics", {}),
                    "baseline_metrics": feedback_payload.get("baseline_core_metrics", {}),
                    "delta_vs_baseline": feedback_payload.get("candidate_vs_baseline_delta", {}),
                    "lipschitz_summary": feedback_payload.get("lipschitz_candidate_summary", {}),
                    "lipschitz_delta_vs_baseline": feedback_payload.get("lipschitz_candidate_vs_baseline_delta", {}),
                    "planning_summary": feedback_payload.get("planning_summary", {}),
                }
                try:
                    feedback_raw = client.chat(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": FEEDBACK_PROMPT + "\n\n" + json.dumps(compressed_feedback_payload, indent=2)},
                        ],
                        response_kind="feedback",
                    )
                    feedback_json, _ = parse_json_with_repair(feedback_raw)
                    if not feedback_json:
                        raise RuntimeError("Feedback stage JSON parse failed under compressed retry.")
                    feedback_payload = compressed_feedback_payload
                except Exception as retry_exc:  # noqa: BLE001
                    feedback_error = retry_exc
            if not feedback_json:
                # Feedback-only final fallback to chat model with strict JSON output.
                from openai import OpenAI

                feedback_fallback_used = True
                feedback_final_model = client.chat_model
                fallback_payload = {
                    "round_index": int(round_idx + 1),
                    "task_mode": feedback_payload.get("task_mode", ""),
                    "best_candidate_metrics": feedback_payload.get("best_candidate_metrics", feedback_payload.get("candidate_core_metrics", {})),
                    "delta_vs_baseline": feedback_payload.get("delta_vs_baseline", feedback_payload.get("candidate_vs_baseline_delta", {})),
                    "lipschitz_summary": feedback_payload.get("lipschitz_summary", feedback_payload.get("lipschitz_candidate_summary", {})),
                    "lipschitz_delta_vs_baseline": feedback_payload.get(
                        "lipschitz_delta_vs_baseline", feedback_payload.get("lipschitz_candidate_vs_baseline_delta", {})
                    ),
                    "planning_summary": feedback_payload.get("planning_summary", {}),
                }
                try:
                    fallback_client = OpenAI(
                        api_key=client.api_key,
                        base_url=client._normalize_base_url(),
                        timeout=client.timeout_seconds,
                        max_retries=0,
                    )
                    t0 = time.time()
                    fallback_resp = fallback_client.chat.completions.create(
                        model=client.chat_model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": FEEDBACK_PROMPT + "\n\n" + json.dumps(fallback_payload, indent=2)},
                        ],
                        temperature=0.0,
                        max_tokens=int(max(4096, client.max_tokens)),
                        response_format={"type": "json_object"},
                    )
                    fallback_content = fallback_resp.choices[0].message.content or ""
                    feedback_json, _ = parse_json_with_repair(fallback_content)
                    if not feedback_json:
                        raise RuntimeError("Feedback fallback chat model JSON parse failed.")
                    feedback_payload = fallback_payload
                    choice0 = fallback_resp.choices[0]
                    msg_obj = getattr(choice0, "message", None)
                    reasoning_content = getattr(msg_obj, "reasoning_content", None) if msg_obj is not None else None
                    client._record_call(
                        response_kind="feedback",
                        model=client.chat_model,
                        success=True,
                        latency_sec=time.time() - t0,
                        error="",
                        finish_reason=str(getattr(choice0, "finish_reason", "") or ""),
                        content_len=len(fallback_content),
                        reasoning_content_len=(len(reasoning_content) if isinstance(reasoning_content, str) else 0),
                    )
                except Exception as fallback_exc:  # noqa: BLE001
                    _write_failure_artifacts(run_dir, "feedback", fallback_exc if feedback_error is None else feedback_error, client)
                    raise
        (round_dir / "feedback.json").write_text(json.dumps({"source": "llm", "feedback": feedback_json}, indent=2), encoding="utf-8")

        summary = {
            "round": round_idx + 1,
            "selected_task": route.get("task_mode"),
            "route": route,
            "planning": planning_json,
            "planning_repaired_from_raw": planning_repaired,
            "best_metric": "selection_score",
            "best_value": best_candidate["metrics"].get("selection_score"),
            "best_candidate_id": str(best_candidate.get("candidate_id", "")),
            "best_candidate_path": str(best_candidate.get("candidate_path", "")),
            "success_rate": float(best_candidate["metrics"].get("success_rate", 0.0)),
            "communication_recovery_ratio": float(best_candidate["metrics"].get("communication_recovery_ratio", 0.0)),
            "power_recovery_ratio": float(best_candidate["metrics"].get("power_recovery_ratio", 0.0)),
            "road_recovery_ratio": float(best_candidate["metrics"].get("road_recovery_ratio", 0.0)),
            "critical_load_recovery_ratio": float(best_candidate["metrics"].get("critical_load_recovery_ratio", 0.0)),
            "constraint_violation_rate_eval": float(best_candidate["metrics"].get("constraint_violation_rate_eval", 0.0)),
            "mean_progress_delta_eval": float(best_candidate["metrics"].get("mean_progress_delta_eval", 0.0)),
            "lipschitz_mean": float(best_candidate["metrics"].get("lipschitz_mean", 0.0)),
            "lipschitz_max": float(best_candidate["metrics"].get("lipschitz_max", 0.0)),
            "lipschitz_min": float(best_candidate["metrics"].get("lipschitz_min", 0.0)),
            "best_candidate": best_candidate,
            "feedback_payload": feedback_payload,
            "lipschitz_feedback_summary": {
                "candidate": feedback_payload.get("lipschitz_candidate_summary", {}),
                "baseline": feedback_payload.get("lipschitz_baseline_summary", {}),
                "delta": feedback_payload.get("lipschitz_candidate_vs_baseline_delta", {}),
            },
            "router_source": "llm",
            "planning_source": "llm",
            "feedback_source": "llm",
            "feedback_fallback_used": bool(feedback_fallback_used),
            "feedback_primary_model": feedback_primary_model,
            "feedback_final_model": feedback_final_model,
            "task_switched_vs_prev_round": bool(route.get("task_switched_vs_prev_round", False)),
            "selection_diagnostics": selection_diagnostics,
            "llm_feedback": feedback_json,
            "llm_effective_mode": client.effective_mode(),
            "router_mode": args.router_mode,
        }
        (round_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        history.append(summary)

    llm_audit = {
        "requested_mode": args.llm_mode,
        "effective_mode": client.effective_mode(),
        "router_mode": args.router_mode,
        "preflight_ok": True,
        "api_provider": client.api_provider,
        "base_url": client.base_url,
        "chat_model": client.chat_model,
        "reasoner_model": client.reasoner_model,
        "api_key_present": bool(client.api_key),
        "preflight_chat_ok": client.preflight_chat_ok,
        "preflight_reasoner_ok": client.preflight_reasoner_ok,
        "router_model": client.reasoner_model,
        "planning_model": client.reasoner_model,
        "codegen_model": client.chat_model,
        "feedback_model": client.reasoner_model,
        "feedback_fallback_used": any(bool(r.get("feedback_fallback_used", False)) for r in history),
        "feedback_primary_model": client.reasoner_model,
        "feedback_final_model": history[-1].get("feedback_final_model", client.reasoner_model) if history else client.reasoner_model,
        "real_llm_call_count": client.call_count,
        "no_mock_fallback": True,
        "no_codegen_baseline_fallback": True,
        "hard_fail_on_llm_error": True,
    }
    kind_counts: dict[str, int] = {}
    success_counts: dict[str, int] = {}
    for call in client.call_history:
        kind = str(call.get("response_kind", "unknown"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if bool(call.get("success", False)):
            success_counts[kind] = success_counts.get(kind, 0) + 1
    llm_audit["call_counts_by_kind"] = kind_counts
    llm_audit["successful_call_counts_by_kind"] = success_counts
    llm_audit["required_stages_present"] = all(k in kind_counts for k in ("router", "planning", "codegen", "feedback"))
    (run_dir / "llm_call_log.json").write_text(json.dumps(client.call_history, indent=2), encoding="utf-8")
    final_selection_diag = history[-1].get("selection_diagnostics", {}) if history else {}
    (run_dir / "outer_loop_final_summary.json").write_text(
        json.dumps(
            {
                "rounds": history,
                "selection_diagnostics": final_selection_diag,
                "llm_audit": llm_audit,
                "llm_effective_mode": client.effective_mode(),
                "router_mode": args.router_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _prune_unused_artifacts(run_dir)
    _write_artifact_manifest(run_dir)

    formal_dir_cfg = str(cfg.get("paths", {}).get("formal_outer_loop_dir", "")).strip()
    if formal_dir_cfg:
        formal_dir = Path(formal_dir_cfg)
        formal_dir.mkdir(parents=True, exist_ok=True)
        _clear_directory_contents(formal_dir)
        for src in run_dir.iterdir():
            dst = formal_dir / src.name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    LOGGER.info("Outer loop complete. Final summary: %s", run_dir / "outer_loop_final_summary.json")


if __name__ == "__main__":
    main()

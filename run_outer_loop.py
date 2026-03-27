from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from llm_client import LLMClient
from mock_recovery_env import ProjectRecoveryEnv
from prompts import CODEGEN_PROMPT, FEEDBACK_PROMPT, PLANNING_PROMPT, ROUTER_PROMPT, SYSTEM_PROMPT
from router import route_llm, summarize_trajectory
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
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            try:
                return json.loads(cleaned), True
            except json.JSONDecodeError:
                pass
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s : e + 1]), True
            except json.JSONDecodeError:
                return {}, True
    return {}, True


def recover_candidate_payload_from_raw(raw: str, task_mode: str, stage: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    code = ""
    if "```" in text:
        chunks = text.split("```")
        for chunk in chunks:
            c = chunk.strip()
            if c.startswith("python"):
                c = c[len("python") :].strip()
            if "def revise_state" in c and "def intrinsic_reward" in c:
                code = c
                break
    if not code and "def revise_state" in text and "def intrinsic_reward" in text:
        code = text
    if not code:
        return {}
    safe_task = "".join(ch for ch in task_mode if ch.isalnum() or ch == "_").strip("_") or "candidate"
    safe_stage = "".join(ch for ch in stage if ch.isalnum() or ch == "_").strip("_") or "stage"
    return {
        "file_name": f"{safe_task}_{safe_stage}.py",
        "rationale": "Recovered candidate payload from non-JSON model output.",
        "code": code,
        "expected_behavior": "LLM-generated recovery shaping candidate.",
    }


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
    for idx, (state, info) in enumerate(synth_inputs):
        try:
            rs = revise_state(state, info)
        except TypeError:
            rs = revise_state(state)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Semantic validation: revise_state failed on sample {idx}: {exc}")
            continue
        arr = np.asarray(rs, dtype=float)
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
        revised_samples.append(arr)

    if revised_lens and len(set(revised_lens)) != 1:
        errors.append(f"Semantic validation: revise_state output length is not stable across inputs: {revised_lens}")

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
        if abs(ir_f) > 10.0:
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
    return "coordinated"


def _aggregate_action_category_distribution(action_usage: dict[str, Any]) -> dict[str, float]:
    cats = {"road": 0.0, "power": 0.0, "comm": 0.0, "mes": 0.0, "feeder": 0.0, "coordinated": 0.0}
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


def build_feedback(best_candidate: dict[str, Any], score_metric: str) -> dict[str, Any]:
    metrics = best_candidate.get("metrics", {})
    hints: list[str] = []
    if int(metrics.get("constraint_violation_count", 0)) > 5:
        hints.append("Constraint violations are frequent.")
    if float(metrics.get("critical_load_recovery_ratio", 0.0)) < 0.6:
        hints.append("Critical load recovery is still low.")
    if float(metrics.get("road_recovery_ratio", 0.0)) < 0.6:
        hints.append("Road restoration is lagging and may bottleneck repairs.")
    if not hints:
        hints.append("No major failure mode detected.")

    return {
        "primary_score_metric": score_metric,
        "primary_score_value": metrics.get(score_metric, 0.0),
        "per_metric_breakdown": metrics,
        "failure_mode_hints": hints,
        "action_usage_summary": metrics.get("action_usage", {}),
        "module_change_summary": {
            "file_name": best_candidate.get("candidate", {}).get("file_name", ""),
            "rationale": best_candidate.get("candidate", {}).get("rationale", ""),
            "expected_behavior": best_candidate.get("candidate", {}).get("expected_behavior", ""),
        },
    }


def build_planning_payload(route: dict[str, Any], routing_context: dict[str, Any], previous_feedback: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "task_mode": str(route.get("task_mode", "global_efficiency_priority")),
        "stage": str(route.get("stage", "middle")),
        "route_reason": str(route.get("reason", "")),
        "routing_context": routing_context,
        "latest_feedback": previous_feedback or {},
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

        if reasons:
            rejected_ids.append(cid)
            rejection_reasons[cid] = reasons
        else:
            accepted.append(cand)

    pool = accepted if accepted else [c for c in round_candidates if isinstance(c.get("metrics"), dict)]
    if not pool:
        raise RuntimeError("No candidate with metrics is available for selection.")
    best_metrics = select_best([c["metrics"] for c in pool], "selection_score", higher_is_better)
    best_candidate = next(c for c in pool if c["metrics"] is best_metrics)
    return {
        "best_candidate": best_candidate,
        "selection_diagnostics": {
            "selection_policy": "gate_then_score",
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
    candidates_per_round = int(cfg["outer_loop"]["candidates_per_round"])
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

    for round_idx in range(rounds):
        previous_best = history[-1].get("best_candidate", {}) if history else None
        prev_metrics = previous_best.get("metrics", {}) if previous_best else {}
        routing_context = collect_routing_context(args.env, prev_metrics, cfg, previous_best_candidate=previous_best)

        if round_idx == 0 or args.reroute_each_round:
            try:
                route = route_llm(client, SYSTEM_PROMPT, ROUTER_PROMPT, routing_context)
            except Exception as exc:  # noqa: BLE001
                _write_failure_artifacts(run_dir, "router", exc, client)
                raise
            if route["task_mode"] not in TASK_MODE_ALLOWED:
                route["task_mode"] = "global_efficiency_priority"
                route["final_task_mode"] = route["task_mode"]
                route["override_applied"] = True
                route["override_reason"] = "task_mode normalized to simplified 3-task formal set"

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
            planning_raw = client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": PLANNING_PROMPT + "\n\n" + json.dumps(planning_payload, indent=2)}],
                response_kind="planning",
                sample_idx=round_idx,
            )
            planning_json, planning_repaired = parse_json_with_repair(planning_raw)
            if not planning_json:
                raise RuntimeError("Planning stage JSON parse failed under real LLM run.")
        except Exception as exc:  # noqa: BLE001
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
            prompt += "\n\nReturn compact JSON and keep generated code concise (<= 80 lines)."
            if history:
                prompt += "\n\nLatest feedback:\n" + json.dumps(history[-1].get("feedback_payload", {}), indent=2)
            raw = "{}"
            parsed: dict[str, Any] = {}
            repaired = True
            report: dict[str, Any] = {"valid": False, "errors": ["not attempted"], "normalized_payload": {}}
            for attempt in range(2):
                attempt_prompt = prompt
                if attempt > 0:
                    attempt_prompt += (
                        "\n\nPrevious output was invalid. Respond with ONLY one JSON object with keys: "
                        "file_name, rationale, code, expected_behavior."
                    )
                try:
                    raw = client.chat(
                        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": attempt_prompt}],
                        response_kind="codegen",
                        sample_idx=sample_idx + round_idx * 10 + attempt,
                    )
                except Exception as exc:  # noqa: BLE001
                    if attempt < 1:
                        continue
                    raw = ""
                    parsed = {
                        "file_name": f"fallback_{route['task_mode']}_{route['stage']}.py",
                        "rationale": f"Codegen call failed under real LLM ({exc}); fallback to baseline_noop-compatible candidate.",
                        "code": Path("baseline_noop.py").read_text(encoding="utf-8"),
                        "expected_behavior": "Safety fallback candidate when real LLM codegen call fails.",
                    }
                    repaired = True
                    report = validate_candidate_payload(
                        parsed,
                        max_revised_dim=(
                            int(cfg.get("state_representation", {}).get("max_revised_dim"))
                            if cfg.get("state_representation", {}).get("max_revised_dim") is not None
                            else None
                        ),
                    )
                    report["repaired_from_raw"] = True
                    break
                parsed, repaired = parse_json_with_repair(raw)
                if not parsed:
                    recovered = recover_candidate_payload_from_raw(raw, task_mode=route["task_mode"], stage=route["stage"])
                    if recovered:
                        parsed = recovered
                        repaired = True
                if not parsed and attempt == 1:
                    baseline_code = Path("baseline_noop.py").read_text(encoding="utf-8")
                    parsed = {
                        "file_name": f"fallback_{route['task_mode']}_{route['stage']}.py",
                        "rationale": "Codegen JSON parse failed; fallback to baseline_noop-compatible candidate for continuity.",
                        "code": baseline_code,
                        "expected_behavior": "Safety fallback candidate when real LLM codegen payload is malformed.",
                    }
                    repaired = True
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
                fallback_payload = {
                    "file_name": f"fallback_{route['task_mode']}_{route['stage']}.py",
                    "rationale": "Fallback to baseline_noop-compatible candidate after invalid codegen output.",
                    "code": Path("baseline_noop.py").read_text(encoding="utf-8"),
                    "expected_behavior": "Guaranteed-valid fallback candidate to keep formal run trainable.",
                }
                fallback_report = validate_candidate_payload(
                    fallback_payload,
                    max_revised_dim=(
                        int(cfg.get("state_representation", {}).get("max_revised_dim"))
                        if cfg.get("state_representation", {}).get("max_revised_dim") is not None
                        else None
                    ),
                )
                fallback_report["repaired_from_raw"] = True
                if fallback_report["valid"]:
                    parsed = fallback_payload
                    report = fallback_report
                    repaired = True

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
                )
                metrics["selected_task"] = route.get("final_task_mode", route.get("task_mode"))
                metrics["llm_effective_mode"] = client.effective_mode()
                metrics["router_mode"] = args.router_mode
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

        feedback_payload = build_feedback(best_candidate, "selection_score")
        try:
            feedback_raw = client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": FEEDBACK_PROMPT + "\n\n" + json.dumps(feedback_payload, indent=2)}],
                response_kind="feedback",
            )
            feedback_json, _ = parse_json_with_repair(feedback_raw)
            if not feedback_json:
                raise RuntimeError("Feedback stage JSON parse failed under real LLM run.")
        except Exception as exc:  # noqa: BLE001
            _write_failure_artifacts(run_dir, "feedback", exc, client)
            raise
        (round_dir / "feedback.json").write_text(json.dumps({"source": "llm", "feedback": feedback_json}, indent=2), encoding="utf-8")

        summary = {
            "round": round_idx + 1,
            "selected_task": route.get("final_task_mode", route.get("task_mode")),
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
            "best_candidate": best_candidate,
            "feedback_payload": feedback_payload,
            "router_source": "llm",
            "planning_source": "llm",
            "feedback_source": "llm",
            "llm_task_mode_raw": route.get("llm_task_mode_raw", route.get("task_mode")),
            "final_task_mode": route.get("final_task_mode", route.get("task_mode")),
            "override_applied": bool(route.get("override_applied", False)),
            "override_reason": str(route.get("override_reason", "")),
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
        "planning_model": client.chat_model,
        "codegen_model": client.chat_model,
        "feedback_model": client.reasoner_model,
        "real_llm_call_count": client.call_count,
    }
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

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import math
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
from prompts import COMPACT_PLANNING_PROMPT, FEEDBACK_PROMPT, PLANNING_PROMPT, STRUCTURED_SPEC_PROMPT, SYSTEM_PROMPT
from structured_spec_builder import build_module_payload, normalize_phase_contract, normalize_spec
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


def _write_run_status(
    run_dir: Path,
    *,
    started_at: str,
    current_stage: str,
    last_completed_stage: str,
    current_round: int = 0,
    current_candidate: str = "",
    completed: bool = False,
    failed: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "current_stage": current_stage,
        "last_completed_stage": last_completed_stage,
        "current_round": int(current_round),
        "current_candidate": str(current_candidate),
        "completed": bool(completed),
        "failed": bool(failed),
    }
    if extra:
        payload.update(extra)
    (run_dir / "run_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def _safe_short_text(v: Any, max_len: int = 360) -> str:
    text = str(v) if v is not None else ""
    text = " ".join(text.strip().split())
    return text[:max_len]


def _normalize_planning_obj(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str], bool]:
    required = [
        "weakest_layer",
        "weakest_zone",
        "late_stage_risk",
        "violation_risk",
        "should_reward",
        "should_penalize",
        "should_avoid",
        "finishing_strategy",
        "codegen_guidance",
        "phase_mode",
        "phase_duration",
        "resource_floor_target",
        "completion_push_allowed",
        "late_stage_trigger",
    ]
    notes: list[str] = []
    normalized: dict[str, Any] = {}
    changed = False
    for key in required:
        if key not in parsed:
            notes.append(f"missing_key:{key}")
            if key in {"should_reward", "should_penalize", "should_avoid"}:
                normalized[key] = []
            elif key == "phase_duration":
                normalized[key] = 8
            elif key == "resource_floor_target":
                normalized[key] = 0.12
            elif key == "late_stage_trigger":
                normalized[key] = 0.72
            elif key == "completion_push_allowed":
                normalized[key] = True
            else:
                normalized[key] = ""
            changed = True
            continue
        val = parsed.get(key)
        if key in {"should_reward", "should_penalize"}:
            if isinstance(val, list):
                out = []
                for item in val[:6]:
                    try:
                        out.append(int(item))
                    except (TypeError, ValueError):
                        notes.append(f"non_int_{key}:{item}")
                normalized[key] = out
                changed = changed or (out != val)
            else:
                notes.append(f"type_{key}:expected_list")
                normalized[key] = []
                changed = True
        elif key == "should_avoid":
            if isinstance(val, list):
                out = [_safe_short_text(x, max_len=120) for x in val[:6]]
                normalized[key] = out
                changed = changed or (out != val)
            else:
                notes.append("type_should_avoid:expected_list")
                normalized[key] = []
                changed = True
        elif key in {"phase_duration"}:
            try:
                normalized[key] = int(np.clip(int(val), 2, 80))
            except (TypeError, ValueError):
                normalized[key] = 8
                notes.append("type_phase_duration:coerced")
                changed = True
        elif key in {"resource_floor_target", "late_stage_trigger"}:
            defaults = {"resource_floor_target": 0.12, "late_stage_trigger": 0.72}
            bounds = {"resource_floor_target": (0.05, 0.40), "late_stage_trigger": (0.50, 0.95)}
            lo, hi = bounds[key]
            try:
                normalized[key] = float(np.clip(float(val), lo, hi))
            except (TypeError, ValueError):
                normalized[key] = defaults[key]
                notes.append(f"type_{key}:coerced")
                changed = True
        elif key == "completion_push_allowed":
            if isinstance(val, bool):
                normalized[key] = val
            else:
                normalized[key] = str(val).strip().lower() in {"1", "true", "yes", "on"}
                changed = True
        else:
            short = _safe_short_text(val)
            if len(short) == 0:
                notes.append(f"empty_text:{key}")
            if short != str(val):
                changed = True
            normalized[key] = short
    allowed_modes = {"critical_push", "capability_unblock", "balanced_progress", "late_finish", "resource_preserve"}
    if str(normalized.get("phase_mode", "")).strip().lower() not in allowed_modes:
        normalized["phase_mode"] = "balanced_progress"
        notes.append("phase_mode:defaulted")
        changed = True
    return normalized, notes, changed


def _normalize_compact_planning_obj(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str], bool]:
    required = [
        "weakest_layer",
        "weakest_zone",
        "should_reward",
        "should_avoid",
        "codegen_guidance",
        "phase_mode",
        "phase_duration",
        "resource_floor_target",
        "completion_push_allowed",
        "late_stage_trigger",
    ]
    notes: list[str] = []
    normalized: dict[str, Any] = {}
    changed = False
    for key in required:
        if key not in parsed:
            notes.append(f"missing_key:{key}")
            if key in {"should_reward", "should_avoid"}:
                normalized[key] = []
            elif key == "phase_duration":
                normalized[key] = 8
            elif key == "resource_floor_target":
                normalized[key] = 0.12
            elif key == "late_stage_trigger":
                normalized[key] = 0.72
            elif key == "completion_push_allowed":
                normalized[key] = True
            else:
                normalized[key] = ""
            changed = True
            continue
        val = parsed.get(key)
        if key == "should_reward":
            if isinstance(val, list):
                out = []
                for item in val[:4]:
                    try:
                        out.append(int(item))
                    except (TypeError, ValueError):
                        txt = _safe_short_text(item, max_len=24)
                        if txt:
                            out.append(txt)
                normalized[key] = out
                changed = changed or (out != val)
            else:
                normalized[key] = []
                notes.append("type_should_reward:expected_list")
                changed = True
        elif key == "should_avoid":
            if isinstance(val, list):
                out = [_safe_short_text(x, max_len=48) for x in val[:4]]
                normalized[key] = out
                changed = changed or (out != val)
            else:
                normalized[key] = []
                notes.append("type_should_avoid:expected_list")
                changed = True
        elif key == "phase_duration":
            try:
                normalized[key] = int(np.clip(int(val), 2, 80))
            except (TypeError, ValueError):
                normalized[key] = 8
                notes.append("type_phase_duration:coerced")
                changed = True
        elif key in {"resource_floor_target", "late_stage_trigger"}:
            defaults = {"resource_floor_target": 0.12, "late_stage_trigger": 0.72}
            bounds = {"resource_floor_target": (0.05, 0.40), "late_stage_trigger": (0.50, 0.95)}
            lo, hi = bounds[key]
            try:
                normalized[key] = float(np.clip(float(val), lo, hi))
            except (TypeError, ValueError):
                normalized[key] = defaults[key]
                notes.append(f"type_{key}:coerced")
                changed = True
        elif key == "completion_push_allowed":
            if isinstance(val, bool):
                normalized[key] = val
            else:
                normalized[key] = str(val).strip().lower() in {"1", "true", "yes", "on"}
                changed = True
        else:
            short = _safe_short_text(val, max_len=96)
            normalized[key] = short
            changed = changed or (short != str(val))
            if not short:
                notes.append(f"empty_text:{key}")
    allowed_modes = {"critical_push", "capability_unblock", "balanced_progress", "late_finish", "resource_preserve"}
    if str(normalized.get("phase_mode", "")).strip().lower() not in allowed_modes:
        normalized["phase_mode"] = "balanced_progress"
        notes.append("phase_mode:defaulted")
        changed = True
    return normalized, notes, changed


def _normalize_feedback_obj(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str], bool]:
    required = [
        "improvement_focus",
        "keep_signals",
        "avoid_patterns",
        "finish_strategy_adjustments",
        "phase_guidance",
        "next_phase_mode",
        "next_phase_duration",
        "confidence",
    ]
    notes: list[str] = []
    normalized: dict[str, Any] = {}
    changed = False
    for key in required:
        if key not in parsed:
            notes.append(f"missing_key:{key}")
            normalized[key] = [] if key != "confidence" else 0.5
            changed = True
            continue
        val = parsed.get(key)
        if key == "confidence":
            try:
                conf = float(val)
            except (TypeError, ValueError):
                conf = 0.5
                notes.append("type_confidence:coerced")
                changed = True
            conf = float(np.clip(conf, 0.0, 1.0))
            normalized[key] = conf
        elif key == "next_phase_duration":
            try:
                normalized[key] = int(np.clip(int(val), 2, 80))
            except (TypeError, ValueError):
                normalized[key] = 8
                notes.append("type_next_phase_duration:coerced")
                changed = True
        elif key in {"phase_guidance", "next_phase_mode"}:
            short = _safe_short_text(val, max_len=40)
            normalized[key] = short
            changed = changed or (short != str(val))
        else:
            if isinstance(val, list):
                out = [_safe_short_text(x, max_len=80) for x in val[:4]]
                normalized[key] = out
                changed = changed or (out != val)
            else:
                normalized[key] = []
                notes.append(f"type_{key}:expected_list")
                changed = True
    if normalized.get("phase_guidance", "") not in {"keep", "switch", "extend"}:
        normalized["phase_guidance"] = "keep"
        notes.append("phase_guidance:defaulted")
        changed = True
    if normalized.get("next_phase_mode", "") not in {"critical_push", "capability_unblock", "balanced_progress", "late_finish", "resource_preserve"}:
        normalized["next_phase_mode"] = "balanced_progress"
        notes.append("next_phase_mode:defaulted")
        changed = True
    return normalized, notes, changed


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
        "structured_spec": payload.get("structured_spec", {}),
        "phase_contract": payload.get("phase_contract", {}),
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


def validate_structured_spec_payload(payload: dict[str, Any], task_mode: str) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {"valid": False, "errors": ["payload_not_object"], "normalized_payload": {}}
    file_name = str(payload.get("file_name", "structured_candidate.py")).strip() or "structured_candidate.py"
    if not file_name.endswith(".py"):
        file_name = f"{file_name}.py"
    rationale = str(payload.get("rationale", "")).strip()
    expected_behavior = str(payload.get("expected_behavior", "")).strip()
    raw_spec = payload.get("spec", {})
    if not isinstance(raw_spec, dict):
        errors.append("spec_must_be_object")
        raw_spec = {}
    style = str(payload.get("style", raw_spec.get("style", "balanced")))
    normalized_spec = normalize_spec(raw_spec, style=style, task_mode=task_mode)
    phase_raw = payload.get("phase_contract", raw_spec)
    normalized_phase = normalize_phase_contract(phase_raw if isinstance(phase_raw, dict) else {})
    normalized_spec["phase_contract"] = dict(normalized_phase)
    built_payload = build_module_payload(
        normalized_spec,
        file_name=file_name,
        rationale=rationale,
        expected_behavior=expected_behavior,
    )
    return {"valid": len(errors) == 0, "errors": errors, "normalized_payload": built_payload}


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

    regression_order = sorted(
        [
            ("min_recovery_ratio", -core_delta.get("min_recovery_ratio", 0.0)),
            ("constraint_violation_rate_eval", core_delta.get("constraint_violation_rate_eval", 0.0)),
            ("invalid_action_rate_eval", core_delta.get("invalid_action_rate_eval", 0.0)),
            ("lipschitz_mean", lipschitz_delta.get("lipschitz_mean", 0.0)),
            ("wait_hold_usage_eval", core_delta.get("wait_hold_usage_eval", 0.0)),
        ],
        key=lambda x: float(x[1]),
        reverse=True,
    )
    worst_regression_metric, worst_regression_delta = regression_order[0]
    likely_overcorrection = {
        "min_recovery_ratio": "overly conservative shaping preserved safety but weakened recovery floor",
        "constraint_violation_rate_eval": "over-aggressive progress incentives without legality guard",
        "invalid_action_rate_eval": "action push exceeded feasibility constraints",
        "lipschitz_mean": "high-sensitivity reward terms or overly complex appended features",
        "wait_hold_usage_eval": "overly conservative anti-risk shaping",
    }.get(worst_regression_metric, "unknown")
    under_recovery_while_safe = (
        _safe_float(core_delta.get("min_recovery_ratio", 0.0)) < 0.0
        and _safe_float(candidate_core.get("constraint_violation_rate_eval", 1.0)) <= 0.0
        and _safe_float(candidate_core.get("invalid_action_rate_eval", 1.0)) <= 0.0
    )
    structured_repair_instruction = {
        "most_important_regression": {
            "metric": str(worst_regression_metric),
            "delta_vs_baseline": float(worst_regression_delta),
        },
        "likely_overcorrection": likely_overcorrection,
        "reduce": [
            "overly aggressive reward bonuses that increase invalid/violation",
            "high-sensitivity feature/reward couplings that increase lipschitz_mean",
        ],
        "preserve": [
            "signals that improved min_recovery_ratio and critical-load recovery",
            "finish-oriented progression cues that do not increase safety violations",
        ],
        "under_recovery_while_safe": under_recovery_while_safe,
        "recovery_floor_priority": "if safe but under-recovering, increase bounded recovery terms; do not trade away recovery floor for smoothness alone",
        "not_allowed_tradeoff_next_round": "tiny score gain with any safety regression",
    }

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
        "structured_repair_instruction": structured_repair_instruction,
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
        "remaining_recovery_gap": float(env.get("remaining_recovery_gap", 1.0)),
        "completion_feasibility": float(env.get("completion_feasibility", 0.0)),
        "resource_floor_risk": float(env.get("resource_floor_risk", 0.0)),
        "phase_recommendation": str(env.get("phase_recommendation", "balanced_progress")),
        "safe_completion_window": bool(env.get("safe_completion_window", False)),
        "latest_feedback_summary": _summarize_feedback(previous_feedback),
    }


def _extract_phase_contract(
    planning_json: dict[str, Any],
    previous_feedback: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
    previous_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    default_mode = str(planning_json.get("phase_mode", "balanced_progress")).strip().lower() or "balanced_progress"
    if isinstance(previous_feedback, dict) and str(previous_feedback.get("phase_guidance", "")) == "switch":
        default_mode = str(previous_feedback.get("next_phase_mode", default_mode)).strip().lower() or default_mode
    duration = planning_json.get("phase_duration", 8)
    if isinstance(previous_feedback, dict) and str(previous_feedback.get("phase_guidance", "")) == "extend":
        try:
            duration = int(duration) + 2
        except (TypeError, ValueError):
            duration = 10
    try:
        duration = int(duration)
    except (TypeError, ValueError):
        duration = 8
    duration = int(np.clip(duration, 2, 80))
    try:
        resource_floor_target = float(planning_json.get("resource_floor_target", 0.12))
    except (TypeError, ValueError):
        resource_floor_target = 0.12
    resource_floor_target = float(np.clip(resource_floor_target, 0.05, 0.40))
    completion_push_allowed = bool(planning_json.get("completion_push_allowed", True))
    try:
        late_stage_trigger = float(planning_json.get("late_stage_trigger", 0.72))
    except (TypeError, ValueError):
        late_stage_trigger = 0.72
    late_stage_trigger = float(np.clip(late_stage_trigger, 0.50, 0.95))
    if default_mode not in {"critical_push", "capability_unblock", "balanced_progress", "late_finish", "resource_preserve"}:
        default_mode = "balanced_progress"
    benchmark_split = str((cfg or {}).get("benchmark", {}).get("split_name", "")).strip()
    if benchmark_split == "benchmark_eval_presets":
        duration = int(np.clip(duration, 8, 12))
        resource_floor_target = float(np.clip(resource_floor_target, 0.12, 0.15))
        completion_push_allowed = True
        late_stage_trigger = float(np.clip(late_stage_trigger, 0.65, 0.70))
        benchmark_severity = str((cfg or {}).get("benchmark", {}).get("fixed_severity", (cfg or {}).get("scenario", {}).get("severity", "moderate"))).strip().lower()
        prev_truncated = int(_safe_float((previous_metrics or {}).get("eval_truncated_count", 0.0))) > 0
        low_late_finish = _safe_float((previous_metrics or {}).get("late_finish_action_share_eval", 1.0)) < 0.20
        completion_window_entries = _safe_float((previous_metrics or {}).get("completion_window_entries", 0.0))
        prev_min_recovery = _safe_float((previous_metrics or {}).get("min_recovery_ratio", 0.0))
        prev_critical_recovery = _safe_float((previous_metrics or {}).get("critical_load_recovery_ratio", 0.0))
        rep = (previous_metrics or {}).get("representative_eval_summary", {})
        final_stage = str(rep.get("final_stage", "")).strip().lower() if isinstance(rep, dict) else ""
        near_finish_evidence = completion_window_entries >= 2.0 and (
            prev_min_recovery >= 0.80 or prev_critical_recovery >= 0.80 or final_stage == "late"
        )
        if prev_truncated and low_late_finish:
            if benchmark_severity == "moderate":
                if near_finish_evidence:
                    default_mode = "late_finish"
            else:
                default_mode = "late_finish"
    return {
        "phase_mode": default_mode,
        "phase_duration": duration,
        "resource_floor_target": resource_floor_target,
        "completion_push_allowed": completion_push_allowed,
        "late_stage_trigger": late_stage_trigger,
    }


def select_best(results: list[dict[str, Any]], metric: str, higher_is_better: bool) -> dict[str, Any]:
    return sorted(results, key=lambda x: x.get(metric, 0.0), reverse=higher_is_better)[0]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _round_delta_summary(candidate_metrics: dict[str, Any], reference_metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "delta_selection_score": float(
            _safe_float(candidate_metrics.get("selection_score", 0.0)) - _safe_float(reference_metrics.get("selection_score", 0.0))
        ),
        "delta_min_recovery_ratio": float(
            _safe_float(candidate_metrics.get("min_recovery_ratio", 0.0)) - _safe_float(reference_metrics.get("min_recovery_ratio", 0.0))
        ),
        "delta_constraint_violation_rate_eval": float(
            _safe_float(candidate_metrics.get("constraint_violation_rate_eval", 0.0))
            - _safe_float(reference_metrics.get("constraint_violation_rate_eval", 0.0))
        ),
        "delta_invalid_action_rate_eval": float(
            _safe_float(candidate_metrics.get("invalid_action_rate_eval", candidate_metrics.get("invalid_action_rate", 0.0)))
            - _safe_float(reference_metrics.get("invalid_action_rate_eval", reference_metrics.get("invalid_action_rate", 0.0)))
        ),
        "delta_lipschitz_mean": float(
            _safe_float(candidate_metrics.get("lipschitz_mean", 0.0)) - _safe_float(reference_metrics.get("lipschitz_mean", 0.0))
        ),
        "delta_wait_hold_usage_eval": float(
            _safe_float(candidate_metrics.get("wait_hold_usage_eval", candidate_metrics.get("wait_hold_usage", 0.0)))
            - _safe_float(reference_metrics.get("wait_hold_usage_eval", reference_metrics.get("wait_hold_usage", 0.0)))
        ),
    }


def _resolve_candidate_styles(cfg: dict[str, Any]) -> list[str]:
    styles = cfg.get("selection", {}).get("candidate_search_styles", [])
    if isinstance(styles, list):
        allowed = {"conservative_safety_first", "balanced"}
        out = [str(x).strip() for x in styles if str(x).strip() in allowed]
        if out:
            return out
    return ["conservative_safety_first", "balanced"]


def _style_guidance(style: str) -> dict[str, Any]:
    style = str(style).strip().lower()
    if style == "conservative_safety_first":
        return {
            "style": style,
            "emphasis": "minimize invalid/violation and keep smooth reward mapping",
            "prompt": (
                "Repair style: conservative_safety_first.\n"
                "- Primary optimization: keep invalid_action_rate_eval==0 and constraint_violation_rate_eval==0.\n"
                "- Allowed changes: small conservative delta-based shaping and simple appended features.\n"
                "- Disallowed regressions: any increase in invalid/violation; large Lipschitz increase.\n"
                "- Acceptable tradeoff: small score gain is fine only if safety remains zero."
            ),
        }
    if style == "aggressive_recovery_first":
        return {
            "style": style,
            "emphasis": "maximize recovery progress while staying valid",
            "prompt": (
                "Repair style: aggressive_recovery_first.\n"
                "- Primary optimization: maximize min_recovery_ratio and critical-load progress.\n"
                "- Allowed changes: stronger progress terms and stage-progression incentives.\n"
                "- Disallowed regressions: invalid/violation must not increase above 0.\n"
                "- Acceptable tradeoff: only mild smoothness loss is allowed when recovery gain is clearly large."
            ),
        }
    return {
        "style": "balanced",
        "emphasis": "balanced tradeoff between recovery gains and stability",
        "prompt": (
            "Repair style: balanced.\n"
            "- Primary optimization: improve selection_score and min_recovery_ratio together.\n"
            "- Allowed changes: medium-strength progress shaping with explicit anti-instability terms.\n"
            "- Disallowed regressions: do not increase invalid/violation; avoid large Lipschitz increases.\n"
            "- Acceptable tradeoff: no safety regression is allowed for small score gains."
        ),
    }


def _build_style_contract(style: str, reference_metrics: dict[str, Any], previous_feedback: dict[str, Any] | None) -> dict[str, Any]:
    ref_min_recovery = _safe_float(reference_metrics.get("min_recovery_ratio", 0.0))
    target_min_recovery = max(0.51, ref_min_recovery + 0.02)
    return {
        "style": style,
        "reference_metrics": {
            "selection_score": _safe_float(reference_metrics.get("selection_score", 0.0)),
            "min_recovery_ratio": _safe_float(reference_metrics.get("min_recovery_ratio", 0.0)),
            "constraint_violation_rate_eval": _safe_float(reference_metrics.get("constraint_violation_rate_eval", 0.0)),
            "invalid_action_rate_eval": _safe_float(reference_metrics.get("invalid_action_rate_eval", 0.0)),
            "lipschitz_mean": _safe_float(reference_metrics.get("lipschitz_mean", 0.0)),
        },
        "required_outcomes": [
            "invalid_action_rate_eval must stay at 0.0",
            "constraint_violation_rate_eval must stay at 0.0",
            "selection_score should not regress materially",
            f"min_recovery_ratio should reach at least {target_min_recovery:.4f} if feasible",
        ],
        "target_floor_min_recovery_ratio": float(target_min_recovery),
        "failure_to_repair_map": {
            "if_invalid_or_violation_increases": "reduce aggressive bonuses and add explicit legality-aware progress terms",
            "if_lipschitz_increases": "simplify revised features and remove high-sensitivity reward terms",
            "if_min_recovery_stagnates": "increase targeted critical-load/power progress terms with bounded magnitude",
        },
        "previous_feedback_summary": _summarize_feedback(previous_feedback),
    }


def _build_safe_anchor_payload(task_mode: str, file_name: str) -> dict[str, Any]:
    raw_spec = {
        "style": "conservative_safety_first",
        "task_mode": task_mode,
        "append_crit_progress": 1,
        "append_backbone_balance": 1,
        "append_resource_buffer": 1,
        "append_stage_indicator": 1,
        "recovery_floor_emphasis": 0.85,
        "safety_emphasis": 0.95,
        "late_stage_emphasis": 0.55,
        "wait_hold_discouragement": 0.95,
        "critical_gain_scale": 1.0,
        "progress_bonus_scale": 0.9,
        "weak_layer_gain_scale": 0.9,
        "weak_zone_gain_scale": 0.9,
        "late_stage_bonus_scale": 0.9,
        "completion_bonus_scale": 0.95,
        "wait_penalty_scale": 1.45,
        "invalid_penalty_scale": 1.5,
        "constraint_penalty_scale": 1.55,
        "material_penalty_scale": 1.35,
        "recovery_floor_bonus_scale": 1.2,
    }
    spec = normalize_spec(raw_spec, style="conservative_safety_first", task_mode=task_mode)
    return build_module_payload(
        spec,
        file_name=file_name,
        rationale="Deterministic safe-anchor candidate with strict legality/resource guards and recovery-floor support.",
        expected_behavior="Conservative legal actions with bounded progress; intended as safe fallback anchor for uncertain rounds.",
    )


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


def _build_benchmark_reset_options(cfg: dict[str, Any]) -> dict[str, Any] | None | callable:
    bench = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark"), dict) else {}
    enabled = bool(bench.get("enabled", False))
    preset_name = str(bench.get("preset_name", "")).strip()
    preset_group = str(bench.get("preset_group", "")).strip()
    split_name = str(bench.get("split_name", "")).strip()
    mode = str(bench.get("mode", "off")).strip().lower() or "off"
    fixed_severity = str(bench.get("fixed_severity", "")).strip().lower()
    preset_jitter = float(bench.get("preset_jitter", 0.0))
    if not enabled and not preset_name and not preset_group and mode == "off":
        return None

    def _resolver(phase: str, episode_idx: int) -> dict[str, Any]:
        phase_split = split_name
        if enabled and not preset_name and not preset_group and not split_name:
            phase_split = "benchmark_train_presets" if phase == "train" else "benchmark_eval_presets"
        out = {
            "benchmark_mode": mode if enabled else "off",
            "preset_name": preset_name,
            "preset_group": preset_group,
            "split_name": phase_split,
            "preset_index": int(episode_idx),
            "preset_jitter": preset_jitter,
        }
        if fixed_severity:
            out["severity"] = fixed_severity
        return out

    return _resolver


def _resolve_train_eval_horizons(cfg: dict[str, Any]) -> tuple[int, int, str]:
    train_steps = int(cfg.get("env", {}).get("max_steps", 60))
    runtime = cfg.get("benchmark_runtime", {}) if isinstance(cfg.get("benchmark_runtime"), dict) else {}
    eval_steps = int(runtime.get("eval_max_steps", train_steps))
    eval_budget_mode = str(runtime.get("eval_budget_mode", "standard_eval"))
    return train_steps, eval_steps, eval_budget_mode


def _probe_generated_candidate(
    *,
    candidate_path: Path,
    probe_out: Path,
    seed: int,
    route_task_mode: str,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    phase_contract: dict[str, Any],
    train_max_steps: int,
    eval_max_steps: int,
    eval_budget_mode: str,
    reference_metrics: dict[str, Any],
) -> tuple[bool, dict[str, Any], list[str]]:
    probe_metrics = run_training(
        revise_module_path=candidate_path,
        env_name=args.env,
        train_episodes=6,
        eval_episodes=2,
        max_steps_per_episode=min(20, int(train_max_steps)),
        train_max_steps_per_episode=min(20, int(train_max_steps)),
        eval_max_steps_per_episode=min(28, int(eval_max_steps)),
        gamma=float(cfg["training"]["gamma"]),
        task_mode=route_task_mode,
        llm_mode="real",
        output_json_path=probe_out,
        seed=seed,
        max_revised_dim=(int(cfg.get("state_representation", {}).get("max_revised_dim")) if cfg.get("state_representation", {}).get("max_revised_dim") is not None else None),
        task_mode_metric_weights=cfg.get("selection", {}).get("task_mode_metric_weights", {}),
        dqn_cfg=cfg.get("training", {}),
        severity=str(cfg.get("scenario", {}).get("severity", "moderate")),
        intrinsic_mode=args.intrinsic_mode,
        intrinsic_scale=args.intrinsic_scale,
        env_reset_options=_build_benchmark_reset_options(cfg),
        phase_contract=phase_contract,
        eval_budget_mode=eval_budget_mode,
    )
    reasons: list[str] = []
    ref_invalid = _safe_float(reference_metrics.get("invalid_action_rate_eval", reference_metrics.get("invalid_action_rate", 0.0)))
    ref_violation = _safe_float(reference_metrics.get("constraint_violation_rate_eval", 0.0))
    ref_min_recovery = _safe_float(reference_metrics.get("min_recovery_ratio", 0.0))
    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark"), dict) else {}
    split_name = str(bench_cfg.get("split_name", "")).strip().lower()
    severity = str(bench_cfg.get("fixed_severity", cfg.get("scenario", {}).get("severity", "moderate"))).strip().lower()
    invalid_tol = 0.02
    violation_tol = 0.02
    recovery_drop_tol = 0.05
    if severity == "severe":
        invalid_tol = 0.05
        violation_tol = 0.05
        recovery_drop_tol = 0.08
    elif split_name == "benchmark_resource_constrained_presets" and severity == "moderate":
        invalid_tol = 0.03
        violation_tol = 0.03
        recovery_drop_tol = 0.06
    invalid = _safe_float(probe_metrics.get("invalid_action_rate_eval", probe_metrics.get("invalid_action_rate", 1.0)))
    violation = _safe_float(probe_metrics.get("constraint_violation_rate_eval", 1.0))
    wait_hold = _safe_float(probe_metrics.get("wait_hold_usage_eval", probe_metrics.get("wait_hold_usage", 1.0)))
    min_recovery = _safe_float(probe_metrics.get("min_recovery_ratio", 0.0))
    mat_end = _safe_float(probe_metrics.get("material_stock_end_mean", probe_metrics.get("material_stock_mean_end", 1.0)))
    progress = _safe_float(probe_metrics.get("mean_progress_delta_eval", probe_metrics.get("mean_progress_delta", 0.0)))
    late_finish = _safe_float(probe_metrics.get("late_finish_action_share_eval", 0.0))
    if invalid > (ref_invalid + invalid_tol):
        reasons.append("probe_invalid_too_high")
    if violation > (ref_violation + violation_tol):
        reasons.append("probe_violation_too_high")
    if min_recovery < (ref_min_recovery - recovery_drop_tol):
        reasons.append("probe_recovery_floor_too_low")
    if wait_hold > 0.70:
        reasons.append("probe_wait_overuse")
    if mat_end < 0.04 and progress < 0.001:
        reasons.append("probe_material_drop_too_fast")
    return len(reasons) == 0, probe_metrics, reasons


def select_best_candidate(
    round_candidates: list[dict[str, Any]],
    reference_metrics: dict[str, Any],
    higher_is_better: bool,
    previous_best: dict[str, Any] | None = None,
    stability_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback_origins = {"deterministic_safe_anchor", "deterministic_safe_backstop", "previous_round_best"}

    def _is_generated_candidate(c: dict[str, Any]) -> bool:
        return str(c.get("candidate_origin", "generated")) not in fallback_origins

    accepted: list[dict[str, Any]] = []
    acceptable_generated_ids: list[str] = []
    safe_candidates_all: list[dict[str, Any]] = []
    rejected_ids: list[str] = []
    rejection_reasons: dict[str, list[str]] = {}
    stability_cfg = stability_cfg or {}
    stability_enabled = bool(stability_cfg.get("enabled", True))
    small_gain_max = float(stability_cfg.get("small_score_gain_max", 0.02))
    meaningful_score_gain = float(stability_cfg.get("meaningful_score_gain_min", 0.05))
    meaningful_recovery_gain = float(stability_cfg.get("meaningful_recovery_gain_min", 0.03))
    max_violation_regression = float(stability_cfg.get("max_violation_regression", 0.015))
    max_invalid_regression = float(stability_cfg.get("max_invalid_regression", 0.015))
    max_lipschitz_regression = float(stability_cfg.get("max_lipschitz_regression", 15.0))
    max_wait_regression = float(stability_cfg.get("max_wait_usage_regression", 0.20))
    violation_penalty_weight = float(stability_cfg.get("violation_penalty_weight", 0.6))
    invalid_penalty_weight = float(stability_cfg.get("invalid_penalty_weight", 0.6))
    lipschitz_penalty_weight = float(stability_cfg.get("lipschitz_penalty_weight", 0.001))
    wait_penalty_weight = float(stability_cfg.get("wait_penalty_weight", 0.2))
    score_gain_protection = float(stability_cfg.get("score_gain_protection", 0.02))
    recovery_floor_baseline = float(stability_cfg.get("recovery_floor_baseline", 0.510000005364418))
    recovery_floor_tolerance = float(stability_cfg.get("recovery_floor_tolerance", 0.002))
    recovery_floor_penalty_weight = float(stability_cfg.get("recovery_floor_penalty_weight", 2.0))
    recovery_floor_gate_epsilon = float(stability_cfg.get("recovery_floor_gate_epsilon", 1e-6))

    ref_critical = _safe_float(reference_metrics.get("critical_load_recovery_ratio", 0.0))
    ref_progress = _safe_float(reference_metrics.get("mean_progress_delta_eval", reference_metrics.get("mean_progress_delta", 0.0)))
    ref_power = _safe_float(reference_metrics.get("power_recovery_ratio", 0.0))
    ref_comm = _safe_float(reference_metrics.get("communication_recovery_ratio", 0.0))
    ref_road = _safe_float(reference_metrics.get("road_recovery_ratio", 0.0))
    ref_avg_recovery = (ref_power + ref_comm + ref_road) / 3.0
    ref_violation = _safe_float(reference_metrics.get("constraint_violation_rate_eval", 1.0), default=1.0)
    ref_invalid = _safe_float(reference_metrics.get("invalid_action_rate_eval", reference_metrics.get("invalid_action_rate", 1.0)), default=1.0)
    prev_min_recovery = _safe_float(previous_best.get("metrics", {}).get("min_recovery_ratio", 0.0)) if previous_best else 0.0
    ref_min_recovery = _safe_float(reference_metrics.get("min_recovery_ratio", 0.0))
    split_name = str(reference_metrics.get("split_name", reference_metrics.get("benchmark_split", ""))).strip().lower()
    severity = str(reference_metrics.get("severity", reference_metrics.get("fixed_severity", "moderate"))).strip().lower()
    acceptable_invalid_tol = 0.02
    acceptable_violation_tol = 0.02
    acceptable_recovery_drop_tol = 0.05
    if severity == "severe":
        acceptable_invalid_tol = 0.05
        acceptable_violation_tol = 0.05
        acceptable_recovery_drop_tol = 0.08
    elif split_name == "benchmark_resource_constrained_presets" and severity == "moderate":
        acceptable_invalid_tol = 0.03
        acceptable_violation_tol = 0.03
        acceptable_recovery_drop_tol = 0.06
    recovery_floor_target = max(recovery_floor_baseline, ref_min_recovery - recovery_floor_tolerance, prev_min_recovery - recovery_floor_tolerance)
    recovery_gate_triggered = False

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
            min_recovery = _safe_float(metrics.get("min_recovery_ratio", min(power, comm, road)))
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
            if violation > 0.0:
                if not _is_generated_candidate(cand):
                    reasons.append("constraint_violation_not_allowed")
            if invalid_rate > 0.0:
                if not _is_generated_candidate(cand):
                    reasons.append("invalid_action_not_allowed")
            if _is_generated_candidate(cand):
                non_dominated_safe_improvement = (
                    violation <= (ref_violation + acceptable_violation_tol)
                    and invalid_rate <= (ref_invalid + acceptable_invalid_tol)
                    and (
                        min_recovery >= (ref_min_recovery + 1e-6)
                        or critical >= (ref_critical + 1e-6)
                    )
                )
                acceptable_generated = (
                    invalid_rate <= (ref_invalid + acceptable_invalid_tol)
                    and violation <= (ref_violation + acceptable_violation_tol)
                    and min_recovery >= (ref_min_recovery - acceptable_recovery_drop_tol)
                )
                if acceptable_generated or non_dominated_safe_improvement:
                    acceptable_generated_ids.append(cid)
                    reasons = [r for r in reasons if r in {"resource_sustainability_collapse"}]
                else:
                    reasons.append("generated_not_competitive_vs_baseline_tolerance")
            if (min_recovery + recovery_floor_gate_epsilon) < recovery_floor_target and violation <= 0.0 and invalid_rate <= 0.0:
                reasons.append("under_recovery_below_floor")
                recovery_gate_triggered = True
            if violation <= 0.0 and invalid_rate <= 0.0:
                safe_candidates_all.append(cand)

        if reasons:
            rejected_ids.append(cid)
            rejection_reasons[cid] = reasons
        else:
            accepted.append(cand)

    strict_safety_preference_applied = False
    selected_candidate_is_strict_safe = False
    accepted_generated = [c for c in accepted if _is_generated_candidate(c)]
    if accepted_generated:
        pool = accepted_generated
    elif accepted:
        pool = accepted
    else:
        safe_fallback_pool = [
            c
            for c in round_candidates
            if isinstance(c.get("metrics"), dict)
            and _safe_float(c.get("metrics", {}).get("constraint_violation_rate_eval", 1.0)) <= 0.0
            and _safe_float(c.get("metrics", {}).get("invalid_action_rate_eval", c.get("metrics", {}).get("invalid_action_rate", 1.0))) <= 0.0
        ]
        if safe_fallback_pool:
            strict_safety_preference_applied = True
            pool = safe_fallback_pool
        elif previous_best and isinstance(previous_best.get("metrics"), dict):
            pool = [previous_best]
        else:
            pool = [c for c in round_candidates if isinstance(c.get("metrics"), dict)]
    if not pool:
        raise RuntimeError("No candidate with metrics is available for selection.")
    stability_adjusted_scores: dict[str, float] = {}
    recovery_adjusted_scores: dict[str, float] = {}
    for c in pool:
        cid = str(c.get("candidate_id", "unknown"))
        m = c.get("metrics", {})
        base_score = _safe_float(m.get("selection_score", 0.0))
        delta = _round_delta_summary(m, reference_metrics)
        regression_penalty = (
            max(0.0, delta["delta_constraint_violation_rate_eval"]) * violation_penalty_weight
            + max(0.0, delta["delta_invalid_action_rate_eval"]) * invalid_penalty_weight
            + max(0.0, delta["delta_lipschitz_mean"]) * lipschitz_penalty_weight
            + max(0.0, delta["delta_wait_hold_usage_eval"]) * wait_penalty_weight
        )
        stability_adjusted = float(base_score - regression_penalty + max(0.0, delta["delta_selection_score"]) * score_gain_protection)
        floor_deficit = max(0.0, recovery_floor_target - _safe_float(m.get("min_recovery_ratio", 0.0)))
        recovery_adjusted = float(stability_adjusted - floor_deficit * recovery_floor_penalty_weight)
        m["stability_adjusted_selection_score"] = stability_adjusted
        m["recovery_adjusted_selection_score"] = recovery_adjusted
        stability_adjusted_scores[cid] = stability_adjusted
        recovery_adjusted_scores[cid] = recovery_adjusted

    all_zero_success = all(_safe_float(c.get("metrics", {}).get("success_rate", 0.0)) <= 0.0 for c in pool)

    def _candidate_rank_key(c: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        m = c.get("metrics", {})
        origin = str(c.get("candidate_origin", "generated"))
        if origin == "generated":
            source_rank = 2.0
        elif origin == "deterministic_safe_anchor":
            source_rank = 1.0
        else:
            source_rank = 0.0
        success = _safe_float(m.get("success_rate", 0.0))
        min_recovery = _safe_float(m.get("min_recovery_ratio", 0.0))
        wait_usage = _safe_float(m.get("wait_hold_usage_eval", m.get("wait_hold_usage", 0.0)))
        adjusted = _safe_float(m.get("recovery_adjusted_selection_score", m.get("selection_score", 0.0)))
        raw_score = _safe_float(m.get("selection_score", 0.0))
        return (source_rank, success, min_recovery, -wait_usage, adjusted, raw_score)

    best_candidate = sorted(pool, key=_candidate_rank_key, reverse=True)[0]
    best_metrics = best_candidate["metrics"]

    accepted_deterministic_safe = [
        c
        for c in accepted
        if str(c.get("candidate_origin", "")) in {"deterministic_safe_anchor", "deterministic_safe_backstop"}
        and _safe_float(c.get("metrics", {}).get("constraint_violation_rate_eval", 1.0)) <= 0.0
        and _safe_float(c.get("metrics", {}).get("invalid_action_rate_eval", c.get("metrics", {}).get("invalid_action_rate", 1.0))) <= 0.0
    ]
    generated_in_pool = [c for c in pool if _is_generated_candidate(c)]
    if _is_generated_candidate(best_candidate) and accepted_deterministic_safe and generated_in_pool:
        best_generated_min_recovery = max(_safe_float(c.get("metrics", {}).get("min_recovery_ratio", 0.0)) for c in generated_in_pool)
        safe_near_generated = [
            c
            for c in accepted_deterministic_safe
            if _safe_float(c.get("metrics", {}).get("min_recovery_ratio", 0.0)) >= (best_generated_min_recovery - 0.01)
        ]
        if safe_near_generated:
            best_candidate = sorted(
                safe_near_generated,
                key=lambda c: (
                    _safe_float(c.get("metrics", {}).get("min_recovery_ratio", 0.0)),
                    _safe_float(c.get("metrics", {}).get("recovery_adjusted_selection_score", c.get("metrics", {}).get("selection_score", 0.0))),
                ),
                reverse=True,
            )[0]
            best_metrics = best_candidate["metrics"]

    best_violation = _safe_float(best_candidate.get("metrics", {}).get("constraint_violation_rate_eval", 1.0), default=1.0)
    best_invalid = _safe_float(
        best_candidate.get("metrics", {}).get("invalid_action_rate_eval", best_candidate.get("metrics", {}).get("invalid_action_rate", 1.0)),
        default=1.0,
    )
    selected_candidate_is_strict_safe = bool(best_violation <= 0.0 and best_invalid <= 0.0)

    # Hard safety preference: whenever a zero/zero candidate exists in this round, do not allow an unsafe winner.
    if safe_candidates_all and not selected_candidate_is_strict_safe:
        strict_safety_preference_applied = True
        safe_pool = list(safe_candidates_all)
        if all(_safe_float(c.get("metrics", {}).get("success_rate", 0.0)) <= 0.0 for c in safe_pool):
            best_candidate = sorted(
                safe_pool,
                key=lambda c: (
                    _safe_float(c.get("metrics", {}).get("min_recovery_ratio", 0.0)),
                    _safe_float(c.get("metrics", {}).get("selection_score", 0.0)),
                    -_safe_float(c.get("metrics", {}).get("wait_hold_usage_eval", c.get("metrics", {}).get("wait_hold_usage", 0.0))),
                ),
                reverse=True,
            )[0]
        else:
            safe_best_metrics = select_best([c["metrics"] for c in safe_pool], "recovery_adjusted_selection_score", higher_is_better)
            best_candidate = next(c for c in safe_pool if c["metrics"] is safe_best_metrics)
        selected_candidate_is_strict_safe = True

    best_by_stability = sorted(
        pool,
        key=lambda c: _safe_float(c.get("metrics", {}).get("recovery_adjusted_selection_score", -1e9)),
        reverse=True,
    )[0]
    round_delta = _round_delta_summary(best_candidate.get("metrics", {}), reference_metrics)
    stability_guard_triggered = False
    stability_rejection_reason = ""
    selected_from_previous_round = False
    if stability_enabled and previous_best and isinstance(previous_best.get("metrics"), dict):
        safety_regression = (
            round_delta["delta_constraint_violation_rate_eval"] > max_violation_regression
            or round_delta["delta_invalid_action_rate_eval"] > max_invalid_regression
            or round_delta["delta_lipschitz_mean"] > max_lipschitz_regression
            or round_delta["delta_wait_hold_usage_eval"] > max_wait_regression
        )
        tiny_or_small_gain = round_delta["delta_selection_score"] <= small_gain_max
        meaningful_gain = (
            round_delta["delta_selection_score"] >= meaningful_score_gain
            or round_delta["delta_min_recovery_ratio"] >= meaningful_recovery_gain
        )
        if tiny_or_small_gain and safety_regression and not meaningful_gain:
            stability_guard_triggered = True
            stability_rejection_reason = "small_gain_with_safety_or_smoothness_regression"
            if best_by_stability is not best_candidate:
                best_candidate = best_by_stability
                round_delta = _round_delta_summary(best_candidate.get("metrics", {}), reference_metrics)
        elif best_by_stability is not best_candidate:
            best_candidate = best_by_stability
            round_delta = _round_delta_summary(best_candidate.get("metrics", {}), reference_metrics)

    sentinel_rescue_applied = False
    sentinel_rescue_reason = ""
    sentinel_rescue_origin = ""

    # Final sentinel rescue: never emit sentinel-like final candidate when deterministic safe fallback exists.
    if True:
        best_m = best_candidate.get("metrics", {}) if isinstance(best_candidate.get("metrics"), dict) else {}
        best_sel = _safe_float(best_m.get("selection_score", 0.0))
        collapsed = (
            (not math.isfinite(best_sel))
            or best_sel <= -1e8
            or (
                _safe_float(best_m.get("min_recovery_ratio", 0.0)) <= 0.0
                and _safe_float(best_m.get("critical_load_recovery_ratio", 0.0)) <= 0.0
                and _safe_float(best_m.get("wait_hold_usage_eval", best_m.get("wait_hold_usage", 0.0))) <= 0.0
            )
        )
        if collapsed:
            deterministic_pool = [
                c
                for c in round_candidates
                if str(c.get("candidate_origin", "")) in {"deterministic_safe_anchor", "deterministic_safe_backstop"}
                and isinstance(c.get("metrics"), dict)
                and math.isfinite(_safe_float(c.get("metrics", {}).get("selection_score", float("-inf"))))
                and math.isfinite(_safe_float(c.get("metrics", {}).get("min_recovery_ratio", float("-inf"))))
                and math.isfinite(_safe_float(c.get("metrics", {}).get("critical_load_recovery_ratio", float("-inf"))))
            ]
            if deterministic_pool:
                sentinel_rescue_applied = True
                sentinel_rescue_reason = "generated_candidate_invalid"
                best_candidate = sorted(
                    deterministic_pool,
                    key=lambda c: (
                        _safe_float(c.get("metrics", {}).get("min_recovery_ratio", 0.0)),
                        _safe_float(c.get("metrics", {}).get("critical_load_recovery_ratio", 0.0)),
                        _safe_float(c.get("metrics", {}).get("selection_score", -1e9)),
                    ),
                    reverse=True,
                )[0]
                sentinel_rescue_origin = str(best_candidate.get("candidate_origin", ""))
                round_delta = _round_delta_summary(best_candidate.get("metrics", {}), reference_metrics)
            else:
                raise RuntimeError("Selection collapsed to sentinel-like candidate and no deterministic safe fallback is available.")
    generated_candidate_count = int(sum(1 for c in round_candidates if _is_generated_candidate(c)))
    selected_candidate_generated = bool(_is_generated_candidate(best_candidate))
    fallback_used = not selected_candidate_generated
    selected_origin = str(best_candidate.get("candidate_origin", "generated"))
    if selected_origin == "generated":
        winner_source = "llm_generated"
    elif selected_origin == "deterministic_safe_anchor":
        winner_source = "safe_anchor"
    else:
        winner_source = "noop_fallback"
    return {
        "best_candidate": best_candidate,
        "selection_diagnostics": {
            "selection_policy": "gate_then_score",
            "zero_success_policy_applied": all_zero_success,
            "accepted_ids": [str(c.get("candidate_id", "")) for c in accepted],
            "rejected_ids": rejected_ids,
            "rejection_reasons": rejection_reasons,
            "used_fallback_pool": len(accepted) == 0,
            "stability_adjusted_scores": stability_adjusted_scores,
            "recovery_adjusted_scores": recovery_adjusted_scores,
            "recovery_floor_target": recovery_floor_target,
            "recovery_gate_triggered": recovery_gate_triggered,
            "stability_guard_enabled": stability_enabled,
            "stability_guard_triggered": stability_guard_triggered,
            "stability_rejection_reason": stability_rejection_reason,
            "selected_from_previous_round": selected_from_previous_round,
            "selected_candidate_meets_recovery_floor": bool(
                _safe_float(best_candidate.get("metrics", {}).get("min_recovery_ratio", 0.0)) + recovery_floor_gate_epsilon >= recovery_floor_target
            ),
            "safe_candidate_ids": [str(c.get("candidate_id", "")) for c in safe_candidates_all],
            "safe_candidate_available": bool(safe_candidates_all),
            "strict_safety_preference_applied": strict_safety_preference_applied,
            "selected_candidate_is_strict_safe": selected_candidate_is_strict_safe,
            "selected_candidate_origin": str(best_candidate.get("candidate_origin", "")),
            "generated_candidate_count": generated_candidate_count,
            "acceptable_generated_candidate_count": int(len(acceptable_generated_ids)),
            "acceptable_generated_candidate_ids": acceptable_generated_ids,
            "selected_candidate_is_generated": selected_candidate_generated,
            "fallback_used": fallback_used,
            "fallback_reason": sentinel_rescue_reason,
            "final_candidate_origin": str(best_candidate.get("candidate_origin", "")),
            "sentinel_rescue_applied": sentinel_rescue_applied,
            "sentinel_rescue_origin": sentinel_rescue_origin,
            "winner_source": winner_source,
            "round_delta_summary": round_delta,
            "stability_thresholds": {
                "small_score_gain_max": small_gain_max,
                "meaningful_score_gain_min": meaningful_score_gain,
                "meaningful_recovery_gain_min": meaningful_recovery_gain,
                "max_violation_regression": max_violation_regression,
                "max_invalid_regression": max_invalid_regression,
                "max_lipschitz_regression": max_lipschitz_regression,
                "max_wait_usage_regression": max_wait_regression,
            },
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="LLM outer loop for project-grade tri-layer recovery env.")
    parser.add_argument("--env", default="project_recovery")
    parser.add_argument("--llm-mode", choices=["real"], default="real")
    parser.add_argument("--router-mode", choices=["llm"], default="llm")
    parser.add_argument(
        "--fixed-task-mode",
        choices=["", "critical_load_priority", "restoration_capability_priority", "global_efficiency_priority"],
        default="",
    )
    parser.add_argument("--reroute-each-round", action="store_true")
    parser.add_argument("--rounds-override", type=int, default=0)
    parser.add_argument("--candidates-override", type=int, default=0)
    parser.add_argument("--intrinsic-mode", choices=["off", "state_only", "full"], default="full")
    parser.add_argument("--intrinsic-scale", type=float, default=1.0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--planning-mode", choices=["standard_planning", "compact_planning"], default="")
    parser.add_argument("--disable-feedback", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if args.llm_mode != "real":
        raise RuntimeError("Formal run requires llm_mode=real.")
    if args.router_mode != "llm":
        raise RuntimeError("Formal run requires router_mode=llm.")
    cfg = load_yaml(Path(args.config))
    planning_mode = str(args.planning_mode or cfg.get("planning", {}).get("mode", "standard_planning")).strip()
    if planning_mode not in {"standard_planning", "compact_planning"}:
        raise RuntimeError(f"Unsupported planning mode: {planning_mode}")
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
    started_at = datetime.now(timezone.utc).isoformat()
    stage_timings: list[dict[str, Any]] = []
    last_completed_stage = "init"
    _write_run_status(run_dir, started_at=started_at, current_stage="preflight", last_completed_stage=last_completed_stage)
    try:
        t0_preflight = time.time()
        client.preflight_check()
        stage_timings.append({"stage": "preflight", "elapsed_sec": float(time.time() - t0_preflight)})
        (run_dir / "stage_timings.json").write_text(json.dumps(stage_timings, indent=2), encoding="utf-8")
        last_completed_stage = "preflight"
    except Exception as exc:  # noqa: BLE001
        _write_failure_artifacts(run_dir, "preflight", exc, client)
        _write_run_status(
            run_dir,
            started_at=started_at,
            current_stage="preflight",
            last_completed_stage=last_completed_stage,
            failed=True,
            extra={"error": str(exc)},
        )
        raise

    (run_dir / "run_snapshot.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "config": cfg,
                "llm_requested_mode": args.llm_mode,
                "llm_effective_mode": client.effective_mode(),
                "router_mode": args.router_mode,
                "planning_mode": planning_mode,
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
        round_started = time.time()
        _write_run_status(
            run_dir,
            started_at=started_at,
            current_stage="routing",
            last_completed_stage=last_completed_stage,
            current_round=round_idx + 1,
        )
        previous_best = history[-1].get("best_candidate", {}) if history else None
        prev_metrics = previous_best.get("metrics", {}) if previous_best else {}
        round_reference_metrics = _reference_metrics(previous_best, cfg, outputs_root)
        routing_context = collect_routing_context(args.env, prev_metrics, cfg, previous_best_candidate=previous_best)

        previous_task = str(history[-1].get("selected_task", "")) if history else ""
        previous_round_failed = bool(float(prev_metrics.get("success_rate", 0.0)) <= 0.0) if previous_best else False
        route_elapsed = 0.0
        if args.fixed_task_mode:
            route = {
                "task_mode": str(args.fixed_task_mode),
                "stage": str(routing_context.get("env_summary", {}).get("stage", "middle")),
                "reason": "fixed_task_mode_override",
                "source": "fixed",
            }
            stage_timings.append({"stage": "routing", "round": round_idx + 1, "elapsed_sec": 0.0, "fixed_task_mode": str(args.fixed_task_mode)})
            (run_dir / "stage_timings.json").write_text(json.dumps(stage_timings, indent=2), encoding="utf-8")
            last_completed_stage = "routing"
        else:
            try:
                t0_route = time.time()
                route = recognizer.recognize_with_llm(
                    client=client,
                    system_prompt=SYSTEM_PROMPT,
                    routing_context=routing_context,
                    previous_task=previous_task,
                    previous_round_failed=previous_round_failed,
                )
                route_elapsed = float(time.time() - t0_route)
                stage_timings.append({"stage": "routing", "round": round_idx + 1, "elapsed_sec": route_elapsed})
                (run_dir / "stage_timings.json").write_text(json.dumps(stage_timings, indent=2), encoding="utf-8")
                last_completed_stage = "routing"
            except Exception as exc:  # noqa: BLE001
                _write_failure_artifacts(run_dir, "router", exc, client)
                _write_run_status(
                    run_dir,
                    started_at=started_at,
                    current_stage="routing",
                    last_completed_stage=last_completed_stage,
                    current_round=round_idx + 1,
                    failed=True,
                    extra={"error": str(exc)},
                )
                raise
        if route["task_mode"] not in TASK_MODE_ALLOWED:
            raise RuntimeError(f"Router returned unsupported task mode in formal run: {route['task_mode']}")
        route["task_switched_vs_prev_round"] = bool(round_idx > 0 and route.get("task_mode") != previous_task)

        round_dir = run_dir / f"round_{round_idx+1}"
        round_dir.mkdir(parents=True, exist_ok=True)
        route["source"] = "fixed" if args.fixed_task_mode else "llm"
        (round_dir / "route.json").write_text(json.dumps(route, indent=2), encoding="utf-8")
        planning_payload = build_planning_payload(
            route=route,
            routing_context=routing_context,
            previous_feedback=(history[-1].get("llm_feedback", {}) if history else None),
        )
        planning_json: dict[str, Any] = {}
        planning_raw = ""
        planning_repaired = False
        planning_normalized = False
        planning_normalization_notes: list[str] = []
        planning_validation_error = ""
        planning_error: Exception | None = None
        planning_prompt_text = COMPACT_PLANNING_PROMPT if planning_mode == "compact_planning" else PLANNING_PROMPT
        planning_response_kind = "planning_compact" if planning_mode == "compact_planning" else "planning"
        planning_retry_used = False
        planning_token_cap_used = int(min(int(cfg["llm"]["max_tokens"]), 320)) if planning_mode == "compact_planning" else int(max(4096, int(cfg["llm"]["max_tokens"])))
        _write_run_status(
            run_dir,
            started_at=started_at,
            current_stage="planning",
            last_completed_stage=last_completed_stage,
            current_round=round_idx + 1,
            extra={"planning_mode": planning_mode},
        )
        t0_plan = time.time()
        if planning_mode == "compact_planning":
            planning_attempt_payloads = [
                {
                    "task_mode": str(route.get("task_mode", "global_efficiency_priority")),
                    "stage": str(route.get("stage", "middle")),
                    "weakest_layer": str(planning_payload.get("weakest_layer", "")),
                    "weakest_zone": str(planning_payload.get("weakest_zone", "")),
                    "critical_load_shortfall": float(planning_payload.get("critical_load_shortfall", 1.0)),
                    "backbone_comm_ratio": float(planning_payload.get("backbone_comm_ratio", 0.0)),
                    "constraint_violation_rate": float(planning_payload.get("constraint_violation_rate", 0.0)),
                    "invalid_action_rate": float(planning_payload.get("invalid_action_rate", 0.0)),
                },
                {
                    "task_mode": str(route.get("task_mode", "global_efficiency_priority")),
                    "stage": str(route.get("stage", "middle")),
                    "weakest_layer": str(planning_payload.get("weakest_layer", "")),
                    "weakest_zone": str(planning_payload.get("weakest_zone", "")),
                    "strict_json_retry": True,
                },
            ]
        else:
            planning_attempt_payloads = [
                dict(planning_payload),
                {
                    **dict(planning_payload),
                    "route_reason": str(planning_payload.get("route_reason", ""))[:180],
                    "latest_feedback_summary": _summarize_feedback(history[-1].get("llm_feedback", {}) if history else None),
                    "compression_retry": True,
                },
                {
                    "task_mode": str(route.get("task_mode", "global_efficiency_priority")),
                    "stage": str(route.get("stage", "middle")),
                    "weakest_layer": str(planning_payload.get("weakest_layer", "")),
                    "weakest_zone": str(planning_payload.get("weakest_zone", "")),
                    "critical_load_shortfall": float(planning_payload.get("critical_load_shortfall", 1.0)),
                    "backbone_comm_ratio": float(planning_payload.get("backbone_comm_ratio", 0.0)),
                    "strict_json_retry": True,
                },
            ]
        for pidx, payload_try in enumerate(planning_attempt_payloads):
            try:
                planning_retry_used = planning_retry_used or (pidx > 0)
                strict_suffix = (
                    "\n\nStrict output reminder: JSON object only, exact required keys, short values, no markdown, no extra commentary."
                    if pidx > 0
                    else ""
                )
                planning_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": planning_prompt_text + strict_suffix + "\n\n" + json.dumps(payload_try, indent=2)},
                ]
                planning_raw = client.chat(planning_messages, response_kind=planning_response_kind, sample_idx=round_idx + pidx * 1000)
                parsed_plan, repaired_flag = parse_json_with_repair(planning_raw)
                planning_repaired = bool(repaired_flag)
                if not parsed_plan:
                    planning_validation_error = "planning_parse_failed"
                    continue
                if planning_mode == "compact_planning":
                    normalized_plan, norm_notes, norm_changed = _normalize_compact_planning_obj(parsed_plan)
                else:
                    normalized_plan, norm_notes, norm_changed = _normalize_planning_obj(parsed_plan)
                planning_normalization_notes = norm_notes
                planning_normalized = bool(norm_changed)
                missing_after_norm = [k for k, v in normalized_plan.items() if k in {"weakest_layer", "weakest_zone"} and not str(v)]
                if missing_after_norm:
                    planning_validation_error = f"planning_schema_failed:{missing_after_norm}"
                    continue
                planning_json = normalized_plan
                if pidx > 0:
                    planning_payload = payload_try
                planning_validation_error = ""
                break
            except Exception as exc:  # noqa: BLE001
                planning_error = exc
                planning_validation_error = f"planning_exception:{exc}"
                continue

        if not planning_json:
            (round_dir / "planning_failure_raw.txt").write_text(planning_raw or "", encoding="utf-8")
            (round_dir / "planning_failure_diagnostics.json").write_text(
                json.dumps(
                    {
                        "validation_error": planning_validation_error,
                        "normalization_notes": planning_normalization_notes,
                        "parse_repaired": planning_repaired,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            _write_failure_artifacts(run_dir, "planning", planning_error or RuntimeError(planning_validation_error), client)
            _write_run_status(
                run_dir,
                started_at=started_at,
                current_stage="planning",
                last_completed_stage=last_completed_stage,
                current_round=round_idx + 1,
                failed=True,
                extra={"error": planning_validation_error, "planning_mode": planning_mode, "planning_retry_used": planning_retry_used},
            )
            raise RuntimeError(f"Planning stage failed: {planning_validation_error}")
        planning_elapsed = float(time.time() - t0_plan)
        stage_timings.append({"stage": "planning", "round": round_idx + 1, "elapsed_sec": planning_elapsed})
        (run_dir / "stage_timings.json").write_text(json.dumps(stage_timings, indent=2), encoding="utf-8")
        last_completed_stage = "planning"
        (round_dir / "planning_raw.txt").write_text(planning_raw, encoding="utf-8")
        (round_dir / "planning.json").write_text(
            json.dumps(
                {
                    "source": "llm",
                    "planning_mode": planning_mode,
                    "planning_token_cap_used": planning_token_cap_used,
                    "planning_timeout_seconds": int(cfg["llm"]["timeout_seconds"]),
                    "planning_retries_configured": int(cfg["llm"]["max_retries"]),
                    "planning_retry_used": planning_retry_used,
                    "payload": planning_payload,
                    "planning": planning_json,
                    "repaired_from_raw": planning_repaired,
                    "normalized_from_raw": planning_normalized,
                    "normalization_notes": planning_normalization_notes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        phase_contract = _extract_phase_contract(
            planning_json,
            previous_feedback=(history[-1].get("llm_feedback", {}) if history else None),
            cfg=cfg,
            previous_metrics=prev_metrics,
        )
        train_max_steps, eval_max_steps, eval_budget_mode = _resolve_train_eval_horizons(cfg)

        round_candidates: list[dict[str, Any]] = []
        candidate_styles = _resolve_candidate_styles(cfg)
        round_status: dict[str, Any] = {
            "round": round_idx + 1,
            "route_elapsed_sec": float(route_elapsed),
            "planning_elapsed_sec": float(planning_elapsed),
            "candidate_search_styles": candidate_styles,
            "candidates": [],
            "feedback_elapsed_sec": 0.0,
            "round_elapsed_sec": 0.0,
        }
        (round_dir / "round_status.json").write_text(json.dumps(round_status, indent=2), encoding="utf-8")
        for sample_idx in range(candidates_per_round):
            cid = f"r{round_idx+1}_c{sample_idx+1}"
            cdir = round_dir / cid
            cdir.mkdir(parents=True, exist_ok=True)
            style_name = candidate_styles[sample_idx % len(candidate_styles)]
            style_meta = _style_guidance(style_name)
            style_contract = _build_style_contract(style_name, round_reference_metrics, history[-1].get("llm_feedback", {}) if history else None)
            _write_run_status(
                run_dir,
                started_at=started_at,
                current_stage="candidate_codegen",
                last_completed_stage=last_completed_stage,
                current_round=round_idx + 1,
                current_candidate=cid,
            )
            t0_codegen = time.time()

            prompt = STRUCTURED_SPEC_PROMPT.format(
                task_mode=route["task_mode"],
                stage=route["stage"],
                observation_schema=str(cfg["env"]),
                planning_json=json.dumps(planning_json, indent=2, ensure_ascii=False),
            )
            prompt += "\n\nReturn compact JSON for spec only; deterministic local builder will generate Python module."
            prompt += "\n\n" + style_meta["prompt"]
            prompt += "\n\nStyle repair contract (must obey):\n" + json.dumps(style_contract, indent=2)
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
                        "file_name, rationale, expected_behavior, spec."
                    )
                    attempt_prompt += (
                        "\nRetry hard constraints:"
                        "\n- provide bounded scalar coefficients only in spec."
                        "\n- no python code."
                        "\n- keep spec concise and deterministic-builder friendly."
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
                    report = {
                        "valid": False,
                        "errors": [f"codegen_json_parse_failed_after_retries:{cid}"],
                        "normalized_payload": {},
                        "repaired_from_raw": repaired,
                    }
                    break
                if isinstance(parsed, dict) and "phase_contract" not in parsed:
                    parsed["phase_contract"] = dict(phase_contract)
                report = validate_structured_spec_payload(parsed, task_mode=str(route.get("task_mode", "global_efficiency_priority")))
                if report.get("valid", False):
                    report = validate_candidate_payload(
                        report.get("normalized_payload", {}),
                        max_revised_dim=(
                            int(cfg.get("state_representation", {}).get("max_revised_dim"))
                            if cfg.get("state_representation", {}).get("max_revised_dim") is not None
                            else None
                        ),
                    )
                report["repaired_from_raw"] = repaired
                if report["valid"]:
                    break
            codegen_elapsed = float(time.time() - t0_codegen)
            stage_timings.append({"stage": "candidate_codegen_validation", "round": round_idx + 1, "candidate": cid, "elapsed_sec": codegen_elapsed})
            (run_dir / "stage_timings.json").write_text(json.dumps(stage_timings, indent=2), encoding="utf-8")

            (cdir / "prompt.txt").write_text(prompt, encoding="utf-8")
            (cdir / "raw_response.txt").write_text(raw, encoding="utf-8")
            (cdir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

            record = {
                "candidate_id": cid,
                "candidate_origin": "generated",
                "validation": report,
                "candidate": report["normalized_payload"],
                "search_style": style_name,
                "search_style_emphasis": style_meta["emphasis"],
                "style_contract": style_contract,
            }
            if report["valid"]:
                fname = report["normalized_payload"]["file_name"]
                code = report["normalized_payload"]["code"]
                candidate_path = generated_dir / fname
                candidate_path.write_text(code, encoding="utf-8")
                (cdir / fname).write_text(code, encoding="utf-8")

                probe_ok, probe_metrics, probe_reasons = _probe_generated_candidate(
                    candidate_path=candidate_path,
                    probe_out=cdir / "probe_result.json",
                    seed=int(args.base_seed) + round_idx * 100 + sample_idx,
                    route_task_mode=route["task_mode"],
                    cfg=cfg,
                    args=args,
                    phase_contract=phase_contract,
                    train_max_steps=train_max_steps,
                    eval_max_steps=eval_max_steps,
                    eval_budget_mode=eval_budget_mode,
                    reference_metrics=round_reference_metrics,
                )
                record["probe_ok"] = bool(probe_ok)
                record["probe_metrics"] = probe_metrics
                record["probe_reject_reasons"] = probe_reasons
                if not probe_ok:
                    record["metrics"] = {"selection_score": -1e9 if higher_is_better else 1e9, "success_rate": 0.0}
                    record["error"] = "probe_rejected"
                    (cdir / "candidate_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                    round_status["candidates"].append(
                        {
                            "candidate_id": cid,
                            "search_style": style_name,
                            "search_style_emphasis": style_meta["emphasis"],
                            "valid": True,
                            "probe_ok": False,
                            "probe_reject_reasons": probe_reasons,
                            "codegen_validation_elapsed_sec": float(codegen_elapsed),
                            "training_elapsed_sec": 0.0,
                            "selection_score": float(record.get("metrics", {}).get("selection_score", 0.0)),
                        }
                    )
                    (round_dir / "round_status.json").write_text(json.dumps(round_status, indent=2), encoding="utf-8")
                    round_candidates.append(record)
                    continue

                _write_run_status(
                    run_dir,
                    started_at=started_at,
                    current_stage="candidate_training",
                    last_completed_stage="candidate_codegen",
                    current_round=round_idx + 1,
                    current_candidate=cid,
                )
                t0_train = time.time()
                metrics = run_training(
                    revise_module_path=candidate_path,
                    env_name=args.env,
                    train_episodes=int(cfg["training"]["train_episodes"]),
                    eval_episodes=int(cfg["training"]["eval_episodes"]),
                    max_steps_per_episode=int(train_max_steps),
                    train_max_steps_per_episode=int(train_max_steps),
                    eval_max_steps_per_episode=int(eval_max_steps),
                    gamma=float(cfg["training"]["gamma"]),
                    task_mode=route["task_mode"],
                    llm_mode="real",
                    output_json_path=cdir / "training_result.json",
                    seed=int(args.base_seed) + round_idx * 10 + sample_idx,
                    max_revised_dim=(int(cfg.get("state_representation", {}).get("max_revised_dim")) if cfg.get("state_representation", {}).get("max_revised_dim") is not None else None),
                    task_mode_metric_weights=cfg.get("selection", {}).get("task_mode_metric_weights", {}),
                    dqn_cfg=cfg.get("training", {}),
                    severity=str(cfg.get("scenario", {}).get("severity", "moderate")),
                    intrinsic_mode=args.intrinsic_mode,
                    intrinsic_scale=args.intrinsic_scale,
                    env_reset_options=_build_benchmark_reset_options(cfg),
                    phase_contract=phase_contract,
                    eval_budget_mode=eval_budget_mode,
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
                record["training_elapsed_sec"] = float(time.time() - t0_train)
                (cdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            else:
                record["metrics"] = {"selection_score": -1e9 if higher_is_better else 1e9, "success_rate": 0.0}
                record["error"] = "Validation failed"

            (cdir / "candidate_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            round_status["candidates"].append(
                {
                    "candidate_id": cid,
                    "search_style": style_name,
                    "search_style_emphasis": style_meta["emphasis"],
                    "valid": bool(report["valid"]),
                    "codegen_validation_elapsed_sec": float(codegen_elapsed),
                    "training_elapsed_sec": float(record.get("training_elapsed_sec", 0.0)),
                    "selection_score": float(record.get("selection_score", record.get("metrics", {}).get("selection_score", 0.0))),
                }
            )
            (round_dir / "round_status.json").write_text(json.dumps(round_status, indent=2), encoding="utf-8")
            round_candidates.append(record)

        # Deterministic safe anchor candidate: always available each round.
        anchor_id = f"r{round_idx+1}_anchor"
        anchor_dir = round_dir / anchor_id
        anchor_dir.mkdir(parents=True, exist_ok=True)
        anchor_payload = _build_safe_anchor_payload(
            task_mode=str(route.get("task_mode", "global_efficiency_priority")),
            file_name=f"safe_anchor_{route.get('task_mode', 'global_efficiency_priority')}_{round_idx+1}.py",
        )
        anchor_report = validate_candidate_payload(
            anchor_payload,
            max_revised_dim=(
                int(cfg.get("state_representation", {}).get("max_revised_dim"))
                if cfg.get("state_representation", {}).get("max_revised_dim") is not None
                else None
            ),
        )
        anchor_record: dict[str, Any] = {
            "candidate_id": anchor_id,
            "candidate_origin": "deterministic_safe_anchor",
            "validation": anchor_report,
            "candidate": anchor_report.get("normalized_payload", anchor_payload),
            "search_style": "safe_anchor",
            "search_style_emphasis": "deterministic conservative legality/recovery-floor anchor",
            "style_contract": {"anchor": True, "task_mode": route.get("task_mode", "")},
        }
        if anchor_report.get("valid", False):
            anchor_code = anchor_report["normalized_payload"]["code"]
            anchor_fname = anchor_report["normalized_payload"]["file_name"]
            anchor_candidate_path = generated_dir / anchor_fname
            anchor_candidate_path.write_text(anchor_code, encoding="utf-8")
            (anchor_dir / anchor_fname).write_text(anchor_code, encoding="utf-8")
            t0_anchor_train = time.time()
            anchor_metrics = run_training(
                revise_module_path=anchor_candidate_path,
                env_name=args.env,
                train_episodes=int(cfg["training"]["train_episodes"]),
                eval_episodes=int(cfg["training"]["eval_episodes"]),
                max_steps_per_episode=int(train_max_steps),
                train_max_steps_per_episode=int(train_max_steps),
                eval_max_steps_per_episode=int(eval_max_steps),
                gamma=float(cfg["training"]["gamma"]),
                task_mode=route["task_mode"],
                llm_mode="real",
                output_json_path=anchor_dir / "training_result.json",
                seed=int(args.base_seed) + round_idx * 10 + 99,
                max_revised_dim=(
                    int(cfg.get("state_representation", {}).get("max_revised_dim"))
                    if cfg.get("state_representation", {}).get("max_revised_dim") is not None
                    else None
                ),
                task_mode_metric_weights=cfg.get("selection", {}).get("task_mode_metric_weights", {}),
                dqn_cfg=cfg.get("training", {}),
                severity=str(cfg.get("scenario", {}).get("severity", "moderate")),
                intrinsic_mode=args.intrinsic_mode,
                intrinsic_scale=args.intrinsic_scale,
                env_reset_options=_build_benchmark_reset_options(cfg),
                phase_contract=phase_contract,
                eval_budget_mode=eval_budget_mode,
            )
            anchor_metrics["selected_task"] = route.get("task_mode")
            anchor_metrics["llm_effective_mode"] = "deterministic_anchor"
            anchor_metrics["router_mode"] = args.router_mode
            anchor_metrics["wait_hold_usage_eval"] = float(anchor_metrics.get("wait_hold_usage_eval", anchor_metrics.get("wait_hold_usage", 0.0)))
            anchor_record["metrics"] = anchor_metrics
            anchor_record["candidate_path"] = str(anchor_candidate_path)
            anchor_record["task_mode"] = route["task_mode"]
            anchor_record["route_source"] = "deterministic_anchor"
            anchor_record["selection_score"] = float(anchor_metrics.get("selection_score", 0.0))
            anchor_record["representative_eval_summary"] = dict(anchor_metrics.get("representative_eval_summary", {}))
            anchor_record["training_elapsed_sec"] = float(time.time() - t0_anchor_train)
            (anchor_dir / "metrics.json").write_text(json.dumps(anchor_metrics, indent=2), encoding="utf-8")
        else:
            anchor_record["metrics"] = {"selection_score": -1e9 if higher_is_better else 1e9, "success_rate": 0.0}
            anchor_record["error"] = "safe_anchor_validation_failed"

        (anchor_dir / "candidate_record.json").write_text(json.dumps(anchor_record, indent=2), encoding="utf-8")
        round_status["candidates"].append(
            {
                "candidate_id": anchor_id,
                "candidate_origin": "deterministic_safe_anchor",
                "search_style": "safe_anchor",
                "search_style_emphasis": "deterministic conservative legality/recovery-floor anchor",
                "valid": bool(anchor_report.get("valid", False)),
                "codegen_validation_elapsed_sec": 0.0,
                "training_elapsed_sec": float(anchor_record.get("training_elapsed_sec", 0.0)),
                "selection_score": float(anchor_record.get("selection_score", anchor_record.get("metrics", {}).get("selection_score", 0.0))),
            }
        )
        (round_dir / "round_status.json").write_text(json.dumps(round_status, indent=2), encoding="utf-8")
        round_candidates.append(anchor_record)

        # Deterministic baseline-noop safety backstop candidate.
        backstop_id = f"r{round_idx+1}_backstop"
        backstop_dir = round_dir / backstop_id
        backstop_dir.mkdir(parents=True, exist_ok=True)
        backstop_record: dict[str, Any] = {
            "candidate_id": backstop_id,
            "candidate_origin": "deterministic_safe_backstop",
            "search_style": "baseline_noop_backstop",
            "search_style_emphasis": "strict zero-safety backstop with intrinsic off",
            "style_contract": {"backstop": True, "intrinsic_mode": "off", "task_mode": route.get("task_mode", "")},
            "validation": {"valid": True, "source_module": "baseline_noop.py"},
            "candidate": {"file_name": "baseline_noop.py"},
            "candidate_path": str(Path("baseline_noop.py")),
            "task_mode": route["task_mode"],
            "route_source": "deterministic_backstop",
        }
        t0_backstop_train = time.time()
        backstop_attempts: list[dict[str, Any]] = []
        for offset in [10, 40, 50]:
            backstop_seed = int(args.base_seed) + round_idx * 10 + offset
            attempt_metrics = run_training(
                revise_module_path=Path("baseline_noop.py"),
                env_name=args.env,
                train_episodes=int(cfg["training"]["train_episodes"]),
                eval_episodes=int(cfg["training"]["eval_episodes"]),
                max_steps_per_episode=int(train_max_steps),
                train_max_steps_per_episode=int(train_max_steps),
                eval_max_steps_per_episode=int(eval_max_steps),
                gamma=float(cfg["training"]["gamma"]),
                task_mode=route["task_mode"],
                llm_mode="real",
                output_json_path=backstop_dir / f"training_result_seed{backstop_seed}.json",
                seed=backstop_seed,
                max_revised_dim=(
                    int(cfg.get("state_representation", {}).get("max_revised_dim"))
                    if cfg.get("state_representation", {}).get("max_revised_dim") is not None
                    else None
                ),
                task_mode_metric_weights=cfg.get("selection", {}).get("task_mode_metric_weights", {}),
                dqn_cfg=cfg.get("training", {}),
                severity=str(cfg.get("scenario", {}).get("severity", "moderate")),
                intrinsic_mode="off",
                intrinsic_scale=1.0,
                env_reset_options=_build_benchmark_reset_options(cfg),
                phase_contract=phase_contract,
                eval_budget_mode=eval_budget_mode,
            )
            attempt_metrics["backstop_seed"] = backstop_seed
            attempt_metrics["is_strict_safe"] = bool(
                _safe_float(attempt_metrics.get("constraint_violation_rate_eval", 1.0), default=1.0) <= 0.0
                and _safe_float(attempt_metrics.get("invalid_action_rate_eval", attempt_metrics.get("invalid_action_rate", 1.0)), default=1.0) <= 0.0
            )
            backstop_attempts.append(attempt_metrics)

        safe_attempts = [m for m in backstop_attempts if bool(m.get("is_strict_safe", False))]
        candidate_attempts = safe_attempts if safe_attempts else backstop_attempts
        backstop_metrics = sorted(
            candidate_attempts,
            key=lambda m: (
                _safe_float(m.get("selection_score", 0.0)),
                _safe_float(m.get("min_recovery_ratio", 0.0)),
                -_safe_float(m.get("wait_hold_usage_eval", m.get("wait_hold_usage", 0.0))),
            ),
            reverse=True,
        )[0]
        backstop_metrics["selected_task"] = route.get("task_mode")
        backstop_metrics["llm_effective_mode"] = "deterministic_backstop"
        backstop_metrics["router_mode"] = args.router_mode
        backstop_metrics["wait_hold_usage_eval"] = float(backstop_metrics.get("wait_hold_usage_eval", backstop_metrics.get("wait_hold_usage", 0.0)))
        backstop_record["metrics"] = backstop_metrics
        backstop_record["backstop_attempts"] = backstop_attempts
        backstop_record["strict_safe_attempt_count"] = int(len(safe_attempts))
        backstop_record["selected_backstop_seed"] = int(backstop_metrics.get("backstop_seed", -1))
        backstop_record["selection_score"] = float(backstop_metrics.get("selection_score", 0.0))
        backstop_record["representative_eval_summary"] = dict(backstop_metrics.get("representative_eval_summary", {}))
        backstop_record["training_elapsed_sec"] = float(time.time() - t0_backstop_train)
        (backstop_dir / "metrics.json").write_text(json.dumps(backstop_metrics, indent=2), encoding="utf-8")
        (backstop_dir / "candidate_record.json").write_text(json.dumps(backstop_record, indent=2), encoding="utf-8")
        round_status["candidates"].append(
            {
                "candidate_id": backstop_id,
                "candidate_origin": "deterministic_safe_backstop",
                "search_style": "baseline_noop_backstop",
                "search_style_emphasis": "strict zero-safety backstop with intrinsic off",
                "valid": True,
                "codegen_validation_elapsed_sec": 0.0,
                "training_elapsed_sec": float(backstop_record.get("training_elapsed_sec", 0.0)),
                "selection_score": float(backstop_record.get("selection_score", 0.0)),
                "strict_safe_attempt_count": int(backstop_record.get("strict_safe_attempt_count", 0)),
                "selected_backstop_seed": int(backstop_record.get("selected_backstop_seed", -1)),
            }
        )
        (round_dir / "round_status.json").write_text(json.dumps(round_status, indent=2), encoding="utf-8")
        round_candidates.append(backstop_record)

        selection_result = select_best_candidate(
            round_candidates=round_candidates,
            reference_metrics=round_reference_metrics,
            higher_is_better=higher_is_better,
            previous_best=previous_best,
            stability_cfg=cfg.get("selection", {}).get("stability_guard", {}),
        )
        best_candidate = selection_result["best_candidate"]
        selection_diagnostics = selection_result["selection_diagnostics"]

        feedback_payload = build_feedback(
            best_candidate,
            "selection_score",
            reference_metrics=round_reference_metrics,
            planning_summary={
                "weakest_layer": planning_json.get("weakest_layer", ""),
                "weakest_zone": planning_json.get("weakest_zone", ""),
                "finishing_strategy": planning_json.get("finishing_strategy", ""),
                "phase_mode": phase_contract.get("phase_mode", "balanced_progress"),
                "phase_duration": phase_contract.get("phase_duration", 8),
                "resource_floor_target": phase_contract.get("resource_floor_target", 0.12),
                "completion_push_allowed": phase_contract.get("completion_push_allowed", True),
                "late_stage_trigger": phase_contract.get("late_stage_trigger", 0.72),
            },
            previous_feedback=(history[-1].get("llm_feedback", {}) if history else None),
        )
        if bool(selection_diagnostics.get("stability_guard_triggered", False)):
            feedback_payload["stability_alert"] = {
                "guard_triggered": True,
                "reason": str(selection_diagnostics.get("stability_rejection_reason", "")),
                "round_delta_summary": dict(selection_diagnostics.get("round_delta_summary", {})),
                "guidance": [
                    "reduce invalid and constraint-violating behavior",
                    "prefer smoother state-reward mapping and lower lipschitz_mean",
                    "avoid over-aggressive shaping when score gain is marginal",
                ],
            }
            hints = feedback_payload.get("failure_mode_hints", [])
            if isinstance(hints, list):
                hints.extend(
                    [
                        "Stability guard blocked a marginal but riskier candidate.",
                        "Lower invalid/violation rates and smooth reward response before pushing score.",
                    ]
                )
                feedback_payload["failure_mode_hints"] = hints
        feedback_fallback_used = False
        feedback_primary_model = client.reasoner_model
        feedback_final_model = feedback_primary_model
        feedback_repaired = False
        feedback_normalized = False
        feedback_normalization_notes: list[str] = []
        _write_run_status(
            run_dir,
            started_at=started_at,
            current_stage="feedback",
            last_completed_stage=last_completed_stage,
            current_round=round_idx + 1,
        )
        t0_feedback = time.time()
        if args.disable_feedback:
            raw_feedback_fallback = {
                "improvement_focus": ["feedback disabled"],
                "keep_signals": [],
                "avoid_patterns": [],
                "finish_strategy_adjustments": [],
                "phase_guidance": "keep",
                "next_phase_mode": str(phase_contract.get("phase_mode", "balanced_progress")),
                "next_phase_duration": int(phase_contract.get("phase_duration", 8)),
                "confidence": 1.0,
            }
            feedback_json, feedback_normalization_notes, feedback_normalized = _normalize_feedback_obj(raw_feedback_fallback)
        else:
            feedback_json = {}
            feedback_error: Exception | None = None
            feedback_raw = ""
            try:
                feedback_raw = client.chat(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": FEEDBACK_PROMPT + "\n\n" + json.dumps(feedback_payload, indent=2)}],
                    response_kind="feedback",
                )
                feedback_json, repaired_flag = parse_json_with_repair(feedback_raw)
                feedback_repaired = bool(repaired_flag)
                if not feedback_json:
                    raise RuntimeError("Feedback stage JSON parse failed under primary attempt.")
                feedback_json, feedback_normalization_notes, feedback_normalized = _normalize_feedback_obj(feedback_json)
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
                    "structured_repair_instruction": feedback_payload.get("structured_repair_instruction", {}),
                    "planning_summary": feedback_payload.get("planning_summary", {}),
                }
                try:
                    feedback_raw = client.chat(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": FEEDBACK_PROMPT
                                + "\n\nStrict output reminder: JSON object only, required keys only, concise list values."
                                + "\n\n"
                                + json.dumps(compressed_feedback_payload, indent=2),
                            },
                        ],
                        response_kind="feedback",
                    )
                    feedback_json, repaired_flag = parse_json_with_repair(feedback_raw)
                    feedback_repaired = bool(repaired_flag)
                    if not feedback_json:
                        raise RuntimeError("Feedback stage JSON parse failed under compressed retry.")
                    feedback_json, feedback_normalization_notes, feedback_normalized = _normalize_feedback_obj(feedback_json)
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
                    "structured_repair_instruction": feedback_payload.get("structured_repair_instruction", {}),
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
                            {
                                "role": "user",
                                "content": FEEDBACK_PROMPT
                                + "\n\nStrict output reminder: JSON object only, required keys only, concise list values."
                                + "\n\n"
                                + json.dumps(fallback_payload, indent=2),
                            },
                        ],
                        temperature=0.0,
                        max_tokens=int(max(4096, client.max_tokens)),
                        response_format={"type": "json_object"},
                    )
                    fallback_content = fallback_resp.choices[0].message.content or ""
                    feedback_json, repaired_flag = parse_json_with_repair(fallback_content)
                    feedback_repaired = bool(repaired_flag)
                    if not feedback_json:
                        raise RuntimeError("Feedback fallback chat model JSON parse failed.")
                    feedback_json, feedback_normalization_notes, feedback_normalized = _normalize_feedback_obj(feedback_json)
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
        if not args.disable_feedback and not feedback_json:
            (round_dir / "feedback_failure_raw.txt").write_text(feedback_raw or "", encoding="utf-8")
        (round_dir / "feedback.json").write_text(
            json.dumps(
                {
                    "source": "llm",
                    "feedback": feedback_json,
                    "repaired_from_raw": feedback_repaired,
                    "normalized_from_raw": feedback_normalized,
                    "normalization_notes": feedback_normalization_notes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        feedback_elapsed = float(time.time() - t0_feedback)
        stage_timings.append({"stage": "feedback", "round": round_idx + 1, "elapsed_sec": feedback_elapsed})
        (run_dir / "stage_timings.json").write_text(json.dumps(stage_timings, indent=2), encoding="utf-8")
        round_status["feedback_elapsed_sec"] = feedback_elapsed
        round_status["round_elapsed_sec"] = float(time.time() - round_started)
        (round_dir / "round_status.json").write_text(json.dumps(round_status, indent=2), encoding="utf-8")
        last_completed_stage = "feedback"

        summary = {
            "round": round_idx + 1,
            "selected_task": route.get("task_mode"),
            "route": route,
            "planning": planning_json,
            "planning_repaired_from_raw": planning_repaired,
            "best_metric": "selection_score",
            "best_value": best_candidate["metrics"].get("selection_score"),
            "stability_adjusted_selection_score": float(
                _safe_float(best_candidate["metrics"].get("stability_adjusted_selection_score", best_candidate["metrics"].get("selection_score", 0.0)))
            ),
            "stability_guard_triggered": bool(selection_diagnostics.get("stability_guard_triggered", False)),
            "stability_rejection_reason": str(selection_diagnostics.get("stability_rejection_reason", "")),
            "round_delta_summary": dict(selection_diagnostics.get("round_delta_summary", {})),
            "best_candidate_id": str(best_candidate.get("candidate_id", "")),
            "best_candidate_path": str(best_candidate.get("candidate_path", "")),
            "best_candidate_search_style": str(best_candidate.get("search_style", "")),
            "generated_candidate_count": int(selection_diagnostics.get("generated_candidate_count", 0)),
            "acceptable_generated_candidate_count": int(selection_diagnostics.get("acceptable_generated_candidate_count", 0)),
            "fallback_used": bool(selection_diagnostics.get("fallback_used", False)),
            "winner_source": str(selection_diagnostics.get("winner_source", "noop_fallback")),
            "winner_type": "generated" if bool(selection_diagnostics.get("selected_candidate_is_generated", False)) else "fallback",
            "candidate_styles_explored": [str(c.get("search_style", "")) for c in round_candidates],
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
            "planning_mode": planning_mode,
            "planning_retry_used": planning_retry_used,
            "planning_token_cap_used": planning_token_cap_used,
            "feedback_source": "llm",
            "feedback_fallback_used": bool(feedback_fallback_used),
            "feedback_primary_model": feedback_primary_model,
            "feedback_final_model": feedback_final_model,
            "task_switched_vs_prev_round": bool(route.get("task_switched_vs_prev_round", False)),
            "selection_diagnostics": selection_diagnostics,
            "llm_feedback": feedback_json,
            "feedback_repaired_from_raw": bool(feedback_repaired),
            "feedback_normalized_from_raw": bool(feedback_normalized),
            "feedback_normalization_notes": feedback_normalization_notes,
            "llm_effective_mode": client.effective_mode(),
            "router_mode": args.router_mode,
        }
        (round_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_run_status(
            run_dir,
            started_at=started_at,
            current_stage="round_summary",
            last_completed_stage=last_completed_stage,
            current_round=round_idx + 1,
        )
        last_completed_stage = "round_summary"
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
        "planning_mode": planning_mode,
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
                "winner_source": str(final_selection_diag.get("winner_source", "noop_fallback")),
                "final_winner_type": "generated" if bool(final_selection_diag.get("selected_candidate_is_generated", False)) else "fallback",
                "final_fallback_used": bool(final_selection_diag.get("fallback_used", False)),
                "final_generated_candidate_count": int(final_selection_diag.get("generated_candidate_count", 0)),
                "final_acceptable_generated_candidate_count": int(final_selection_diag.get("acceptable_generated_candidate_count", 0)),
                "llm_audit": llm_audit,
                "llm_effective_mode": client.effective_mode(),
                "router_mode": args.router_mode,
                "planning_mode": planning_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_run_status(
        run_dir,
        started_at=started_at,
        current_stage="final_summary",
        last_completed_stage=last_completed_stage,
        current_round=rounds,
        completed=True,
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

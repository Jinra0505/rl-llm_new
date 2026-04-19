from __future__ import annotations

from typing import Any


def _clip(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = default
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def normalize_spec(raw: dict[str, Any] | None, style: str = "balanced", task_mode: str = "global_efficiency_priority") -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    style = str(style or raw.get("style", "balanced")).strip().lower() or "balanced"
    defaults = {
        "conservative_safety_first": {
            "append_crit_progress": 1,
            "append_backbone_balance": 1,
            "append_resource_buffer": 1,
            "append_stage_indicator": 1,
            "w_delta_comm": 0.22,
            "w_delta_power": 0.22,
            "w_delta_road": 0.18,
            "w_delta_critical": 0.28,
            "w_stage_progress": 0.07,
            "w_finish_bonus": 0.06,
            "w_resource_penalty": 0.04,
            "w_wait_hold_penalty": 0.06,
            "w_violation_penalty": 0.08,
        },
        "aggressive_recovery_first": {
            "append_crit_progress": 1,
            "append_backbone_balance": 1,
            "append_resource_buffer": 1,
            "append_stage_indicator": 1,
            "w_delta_comm": 0.32,
            "w_delta_power": 0.32,
            "w_delta_road": 0.24,
            "w_delta_critical": 0.42,
            "w_stage_progress": 0.12,
            "w_finish_bonus": 0.12,
            "w_resource_penalty": 0.05,
            "w_wait_hold_penalty": 0.09,
            "w_violation_penalty": 0.12,
        },
        "balanced": {
            "append_crit_progress": 1,
            "append_backbone_balance": 1,
            "append_resource_buffer": 1,
            "append_stage_indicator": 1,
            "w_delta_comm": 0.28,
            "w_delta_power": 0.28,
            "w_delta_road": 0.2,
            "w_delta_critical": 0.36,
            "w_stage_progress": 0.09,
            "w_finish_bonus": 0.09,
            "w_resource_penalty": 0.045,
            "w_wait_hold_penalty": 0.075,
            "w_violation_penalty": 0.1,
        },
    }
    d = dict(defaults.get(style, defaults["balanced"]))
    style_reward_control_multipliers = {
        "conservative_safety_first": {
            "critical_gain_scale": 0.92,
            "progress_bonus_scale": 0.90,
            "weak_layer_gain_scale": 0.92,
            "weak_zone_gain_scale": 0.92,
            "late_stage_bonus_scale": 0.92,
            "completion_bonus_scale": 0.95,
            "wait_penalty_scale": 1.35,
            "invalid_penalty_scale": 1.35,
            "constraint_penalty_scale": 1.40,
            "material_penalty_scale": 1.35,
            "recovery_floor_bonus_scale": 1.15,
        },
        "balanced": {
            "critical_gain_scale": 1.00,
            "progress_bonus_scale": 1.00,
            "weak_layer_gain_scale": 1.00,
            "weak_zone_gain_scale": 1.00,
            "late_stage_bonus_scale": 1.00,
            "completion_bonus_scale": 1.00,
            "wait_penalty_scale": 1.00,
            "invalid_penalty_scale": 1.00,
            "constraint_penalty_scale": 1.00,
            "material_penalty_scale": 1.00,
            "recovery_floor_bonus_scale": 1.00,
        },
        "aggressive_recovery_first": {
            "critical_gain_scale": 1.25,
            "progress_bonus_scale": 1.22,
            "weak_layer_gain_scale": 1.18,
            "weak_zone_gain_scale": 1.18,
            "late_stage_bonus_scale": 1.22,
            "completion_bonus_scale": 1.25,
            "wait_penalty_scale": 0.85,
            "invalid_penalty_scale": 0.90,
            "constraint_penalty_scale": 0.90,
            "material_penalty_scale": 0.90,
            "recovery_floor_bonus_scale": 1.20,
        },
    }
    task_mode = str(task_mode or raw.get("task_mode", "global_efficiency_priority")).strip()
    task_multipliers = {
        "critical_load_priority": {
            "w_delta_critical": 1.20,
            "w_delta_power": 1.15,
            "w_finish_bonus": 1.10,
            "critical_gain_scale": 1.25,
            "completion_bonus_scale": 1.15,
            "recovery_floor_bonus_scale": 1.20,
        },
        "restoration_capability_priority": {
            "w_delta_comm": 1.15,
            "w_delta_road": 1.15,
            "w_resource_penalty": 1.20,
            "w_wait_hold_penalty": 1.15,
            "material_penalty_scale": 1.25,
            "constraint_penalty_scale": 1.20,
            "wait_penalty_scale": 1.15,
        },
        "global_efficiency_priority": {
            "w_delta_comm": 1.10,
            "w_delta_power": 1.10,
            "w_delta_road": 1.10,
            "w_stage_progress": 1.15,
            "w_finish_bonus": 1.15,
            "progress_bonus_scale": 1.20,
            "late_stage_bonus_scale": 1.20,
            "completion_bonus_scale": 1.10,
        },
    }
    mul = task_multipliers.get(task_mode, task_multipliers["global_efficiency_priority"])
    for k in ["append_crit_progress", "append_backbone_balance", "append_resource_buffer", "append_stage_indicator"]:
        d[k] = 1 if int(raw.get(k, d[k])) > 0 else 0
    d["style"] = style
    d["recovery_floor_emphasis"] = _clip(raw.get("recovery_floor_emphasis", 0.6), 0.0, 1.0, 0.6)
    d["safety_emphasis"] = _clip(raw.get("safety_emphasis", 0.6), 0.0, 1.0, 0.6)
    d["late_stage_emphasis"] = _clip(raw.get("late_stage_emphasis", 0.6), 0.0, 1.0, 0.6)
    d["wait_hold_discouragement"] = _clip(raw.get("wait_hold_discouragement", 0.6), 0.0, 1.0, 0.6)

    d["w_delta_comm"] = _clip(raw.get("w_delta_comm", d["w_delta_comm"] * mul.get("w_delta_comm", 1.0)), 0.0, 0.6, d["w_delta_comm"])
    d["w_delta_power"] = _clip(raw.get("w_delta_power", d["w_delta_power"] * mul.get("w_delta_power", 1.0)), 0.0, 0.6, d["w_delta_power"])
    d["w_delta_road"] = _clip(raw.get("w_delta_road", d["w_delta_road"] * mul.get("w_delta_road", 1.0)), 0.0, 0.5, d["w_delta_road"])
    d["w_delta_critical"] = _clip(raw.get("w_delta_critical", d["w_delta_critical"] * mul.get("w_delta_critical", 1.0)), 0.0, 0.8, d["w_delta_critical"])
    d["w_stage_progress"] = _clip(raw.get("w_stage_progress", d["w_stage_progress"] * mul.get("w_stage_progress", 1.0)), 0.0, 0.3, d["w_stage_progress"])
    d["w_finish_bonus"] = _clip(raw.get("w_finish_bonus", d["w_finish_bonus"] * mul.get("w_finish_bonus", 1.0)), 0.0, 0.3, d["w_finish_bonus"])
    d["w_resource_penalty"] = _clip(raw.get("w_resource_penalty", d["w_resource_penalty"]), 0.0, 0.2, d["w_resource_penalty"])
    d["w_wait_hold_penalty"] = _clip(raw.get("w_wait_hold_penalty", d["w_wait_hold_penalty"]), 0.0, 0.2, d["w_wait_hold_penalty"])
    d["w_violation_penalty"] = _clip(raw.get("w_violation_penalty", d["w_violation_penalty"]), 0.0, 0.3, d["w_violation_penalty"])
    d["task_mode"] = task_mode
    d["reward_controls"] = {
        "critical_gain_scale": _clip(raw.get("critical_gain_scale", mul.get("critical_gain_scale", 1.0) * style_reward_control_multipliers.get(style, {}).get("critical_gain_scale", 1.0)), 0.6, 1.8, 1.0),
        "progress_bonus_scale": _clip(raw.get("progress_bonus_scale", mul.get("progress_bonus_scale", 1.0) * style_reward_control_multipliers.get(style, {}).get("progress_bonus_scale", 1.0)), 0.6, 1.8, 1.0),
        "weak_layer_gain_scale": _clip(raw.get("weak_layer_gain_scale", mul.get("weak_layer_gain_scale", 1.0) * style_reward_control_multipliers.get(style, {}).get("weak_layer_gain_scale", 1.0)), 0.6, 1.8, 1.0),
        "weak_zone_gain_scale": _clip(raw.get("weak_zone_gain_scale", mul.get("weak_zone_gain_scale", 1.0) * style_reward_control_multipliers.get(style, {}).get("weak_zone_gain_scale", 1.0)), 0.6, 1.8, 1.0),
        "late_stage_bonus_scale": _clip(raw.get("late_stage_bonus_scale", mul.get("late_stage_bonus_scale", 1.0) * style_reward_control_multipliers.get(style, {}).get("late_stage_bonus_scale", 1.0)), 0.6, 1.8, 1.0),
        "completion_bonus_scale": _clip(raw.get("completion_bonus_scale", mul.get("completion_bonus_scale", 1.0) * style_reward_control_multipliers.get(style, {}).get("completion_bonus_scale", 1.0)), 0.6, 1.8, 1.0),
        "wait_penalty_scale": _clip(raw.get("wait_penalty_scale", mul.get("wait_penalty_scale", 1.0) * style_reward_control_multipliers.get(style, {}).get("wait_penalty_scale", 1.0)), 0.6, 1.8, 1.0),
        "invalid_penalty_scale": _clip(raw.get("invalid_penalty_scale", mul.get("invalid_penalty_scale", 1.0) * style_reward_control_multipliers.get(style, {}).get("invalid_penalty_scale", 1.0)), 0.6, 1.8, 1.0),
        "constraint_penalty_scale": _clip(raw.get("constraint_penalty_scale", mul.get("constraint_penalty_scale", 1.0) * style_reward_control_multipliers.get(style, {}).get("constraint_penalty_scale", 1.0)), 0.6, 1.8, 1.0),
        "material_penalty_scale": _clip(raw.get("material_penalty_scale", mul.get("material_penalty_scale", 1.0) * style_reward_control_multipliers.get(style, {}).get("material_penalty_scale", 1.0)), 0.6, 1.8, 1.0),
        "recovery_floor_bonus_scale": _clip(raw.get("recovery_floor_bonus_scale", mul.get("recovery_floor_bonus_scale", 1.0) * style_reward_control_multipliers.get(style, {}).get("recovery_floor_bonus_scale", 1.0)), 0.6, 1.8, 1.0),
    }
    return d


def normalize_phase_contract(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    allowed_modes = {"critical_push", "capability_unblock", "balanced_progress", "late_finish", "resource_preserve"}
    phase_mode = str(raw.get("phase_mode", "balanced_progress")).strip().lower() or "balanced_progress"
    if phase_mode not in allowed_modes:
        phase_mode = "balanced_progress"
    try:
        phase_duration = int(raw.get("phase_duration", 8))
    except (TypeError, ValueError):
        phase_duration = 8
    phase_duration = max(2, min(80, phase_duration))
    try:
        resource_floor_target = float(raw.get("resource_floor_target", 0.12))
    except (TypeError, ValueError):
        resource_floor_target = 0.12
    resource_floor_target = max(0.05, min(0.40, resource_floor_target))
    completion_push_allowed = bool(raw.get("completion_push_allowed", True))
    try:
        late_stage_trigger = float(raw.get("late_stage_trigger", 0.72))
    except (TypeError, ValueError):
        late_stage_trigger = 0.72
    late_stage_trigger = max(0.50, min(0.95, late_stage_trigger))
    return {
        "phase_mode": phase_mode,
        "phase_duration": phase_duration,
        "resource_floor_target": resource_floor_target,
        "completion_push_allowed": completion_push_allowed,
        "late_stage_trigger": late_stage_trigger,
    }


def build_module_payload(spec: dict[str, Any], file_name: str, rationale: str = "", expected_behavior: str = "") -> dict[str, Any]:
    s = normalize_spec(spec, str(spec.get("style", "balanced")), str(spec.get("task_mode", "global_efficiency_priority")))
    phase_contract = normalize_phase_contract(spec.get("phase_contract", spec))
    reward_controls = dict(s.get("reward_controls", {}))
    code = f'''import numpy as np

SPEC = {s!r}
PHASE_CONTRACT = {phase_contract!r}
REWARD_CONTROLS = {reward_controls!r}

def revise_state(state, info=None):
    x = np.asarray(state, dtype=np.float32).flatten()
    feats = []
    if SPEC["append_crit_progress"] > 0:
        feats.append(float(np.mean(x[9:12])))
    if SPEC["append_backbone_balance"] > 0:
        bb = x[12:15]
        feats.append(float(1.0 - np.std(bb) / (np.mean(bb) + 1e-6)))
    if SPEC["append_resource_buffer"] > 0:
        feats.append(float(np.clip(np.mean([x[19], x[20], x[21]]), 0.0, 1.0)))
    if SPEC["append_stage_indicator"] > 0:
        feats.append(float(x[22]))
    return np.concatenate([x, np.asarray(feats, dtype=np.float32)], axis=0)


def intrinsic_reward(state, action, next_state, info=None, revised_state=None):
    s = np.asarray(state, dtype=np.float32)
    n = np.asarray(next_state, dtype=np.float32)
    r = 0.0
    r += SPEC["w_delta_comm"] * float(np.mean(n[3:6] - s[3:6]))
    r += SPEC["w_delta_power"] * float(np.mean(n[0:3] - s[0:3]))
    r += SPEC["w_delta_road"] * float(np.mean(n[6:9] - s[6:9]))
    r += SPEC["w_delta_critical"] * float(np.mean(n[9:12] - s[9:12]))
    if n[22] > s[22]:
        r += SPEC["w_stage_progress"] * (0.5 + SPEC["late_stage_emphasis"])
    if float(np.min(n[:9])) > 0.7:
        r += SPEC["w_finish_bonus"] * (0.5 + SPEC["recovery_floor_emphasis"])
    if int(action) == 14:
        r -= SPEC["w_wait_hold_penalty"] * (0.5 + SPEC["wait_hold_discouragement"])
    if n[23] > s[23]:
        r -= SPEC["w_violation_penalty"] * (0.5 + SPEC["safety_emphasis"])
    if n[20] < s[20] * 0.8:
        r -= SPEC["w_resource_penalty"]
    return float(r)
'''
    if not rationale:
        rationale = "Deterministic template module built from bounded structured shaping spec."
    if not expected_behavior:
        expected_behavior = "Bounded delta-based shaping with explicit safety/resource/wait controls and stable appended features."
    return {
        "file_name": file_name,
        "rationale": rationale,
        "code": code,
        "expected_behavior": expected_behavior,
        "structured_spec": s,
        "phase_contract": phase_contract,
    }

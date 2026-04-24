from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

LLM_METHODS = {"single_shot_llm", "full_outer_loop", "ablation_fixed_global"}


@dataclass
class ValidationResult:
    scenario: str
    method: str
    seed: int
    path: str
    completed: bool
    failed: bool
    has_sentinel: bool
    has_zero_recovery_anomaly: bool
    valid_for_paper: bool
    selection_score: float
    min_recovery_ratio: float
    critical_load_recovery_ratio: float
    constraint_violation_rate_eval: float
    invalid_action_rate_eval: float
    wait_hold_usage_eval: float
    eval_success_rate: float


METRICS = [
    "selection_score",
    "min_recovery_ratio",
    "critical_load_recovery_ratio",
    "communication_recovery_ratio",
    "power_recovery_ratio",
    "road_recovery_ratio",
    "constraint_violation_rate_eval",
    "invalid_action_rate_eval",
    "wait_hold_usage_eval",
    "mean_progress_delta_eval",
    "eval_success_rate",
    "safety_capacity_index",
]


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def compute_safety_capacity_index(row: dict[str, Any]) -> float:
    critical = _f(row.get("critical_load_recovery_ratio", 0.0))
    min_rec = _f(row.get("min_recovery_ratio", 0.0))
    invalid = _f(row.get("invalid_action_rate_eval", row.get("invalid_action_rate", 0.0)))
    vio = _f(row.get("constraint_violation_rate_eval", 0.0))
    return float(0.35 * critical + 0.35 * min_rec + 0.15 * (1.0 - invalid) + 0.15 * (1.0 - vio))


def detect_sentinel_invalid(result: dict[str, Any], method: str) -> tuple[bool, bool]:
    selection = _f(result.get("selection_score", 0.0))
    min_rec = _f(result.get("min_recovery_ratio", 0.0))
    critical = _f(result.get("critical_load_recovery_ratio", 0.0))
    completed = bool(result.get("completed", True))
    failed = bool(result.get("failed", False))

    sentinel = (not math.isfinite(selection)) or selection <= -1e8
    zero_recovery = method in LLM_METHODS and min_rec == 0.0 and critical == 0.0
    completed_but_failed_like = completed and (sentinel or zero_recovery) and not failed
    return bool(sentinel or completed_but_failed_like), bool(zero_recovery)


def validate_result(path: Path, scenario: str, method: str, seed: int) -> ValidationResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    has_sentinel, zero_recovery = detect_sentinel_invalid(data, method)
    completed = bool(data.get("completed", True))
    failed = bool(data.get("failed", False))
    valid_for_paper = completed and (not failed) and (not has_sentinel)
    return ValidationResult(
        scenario=scenario,
        method=method,
        seed=seed,
        path=str(path),
        completed=completed,
        failed=failed,
        has_sentinel=has_sentinel,
        has_zero_recovery_anomaly=zero_recovery,
        valid_for_paper=valid_for_paper,
        selection_score=_f(data.get("selection_score", 0.0)),
        min_recovery_ratio=_f(data.get("min_recovery_ratio", 0.0)),
        critical_load_recovery_ratio=_f(data.get("critical_load_recovery_ratio", 0.0)),
        constraint_violation_rate_eval=_f(data.get("constraint_violation_rate_eval", 0.0)),
        invalid_action_rate_eval=_f(data.get("invalid_action_rate_eval", data.get("invalid_action_rate", 0.0))),
        wait_hold_usage_eval=_f(data.get("wait_hold_usage_eval", data.get("wait_hold_usage", 0.0))),
        eval_success_rate=_f(data.get("eval_success_rate", data.get("success_rate", 0.0))),
    )


def aggregate_valid(runs: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in runs if r.get("valid_for_paper")]
    out: dict[str, Any] = {"n_total": len(runs), "n_valid": len(valid), "metrics": {}}
    for metric in METRICS:
        vals = [_f(r.get(metric, 0.0)) for r in valid]
        if not vals:
            continue
        out["metrics"][metric] = {
            "mean": mean(vals),
            "std": pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return out


def as_dict(v: ValidationResult) -> dict[str, Any]:
    return asdict(v)

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from llm_client import LLMClient


TASKS = [
    "road_opening_priority",
    "critical_power_priority",
    "backbone_comm_priority",
    "coordinated_restoration",
    "stabilization_priority",
]


def summarize_trajectory(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    if not trajectory:
        return {"mean_progress_delta": 0.0, "invalid_action_rate": 0.0, "constraint_violation_rate": 0.0, "stage_distribution": {}, "length": 0}
    n = len(trajectory)
    progress = [float(t.get("info", {}).get("progress_delta", 0.0)) for t in trajectory]
    invalid = [1.0 if t.get("info", {}).get("invalid_action", False) else 0.0 for t in trajectory]
    violate = [1.0 if t.get("info", {}).get("constraint_violation", False) else 0.0 for t in trajectory]
    stages = [t.get("info", {}).get("stage", "unknown") for t in trajectory]
    return {
        "mean_progress_delta": sum(progress) / n,
        "invalid_action_rate": sum(invalid) / n,
        "constraint_violation_rate": sum(violate) / n,
        "stage_distribution": {k: v / n for k, v in Counter(stages).items()},
        "length": n,
    }


def route_rule(routing_context: dict[str, Any]) -> dict[str, Any]:
    env = routing_context.get("env_summary", {})
    traj = routing_context.get("trajectory_summary", {})

    comm = float(env.get("communication_recovery_ratio", 0.0))
    power = float(env.get("power_recovery_ratio", 0.0))
    road = float(env.get("road_recovery_ratio", 0.0))
    shortfall = float(env.get("critical_load_shortfall", 1.0))
    backbone_comm = float(env.get("backbone_comm_ratio", comm))
    weakest_zone = str(env.get("weakest_zone", "A"))
    weakest_layer_idx = str(env.get("weakest_layer", "0"))

    if traj.get("constraint_violation_rate", 0.0) > 0.25:
        return {"task_mode": "stabilization_priority", "confidence": 0.86, "reason": "Frequent violations observed.", "stage": "late"}

    avg = (comm + power + road) / 3.0
    stage = "early" if avg < 0.35 else "middle" if avg < 0.75 else "late"

    if stage == "early" and road < min(comm, power):
        return {"task_mode": "road_opening_priority", "confidence": 0.82, "reason": f"Road is weakest in early stage (zone {weakest_zone}).", "stage": stage}

    if shortfall > 0.35 or weakest_layer_idx == "0":
        return {"task_mode": "critical_power_priority", "confidence": 0.84, "reason": "Critical-load shortfall/high power weakness.", "stage": stage}

    if weakest_layer_idx == "1" and min(comm, backbone_comm) < 0.65:
        return {"task_mode": "backbone_comm_priority", "confidence": 0.8, "reason": "Communication layer is weakest.", "stage": stage}

    if stage == "middle":
        return {"task_mode": "coordinated_restoration", "confidence": 0.8, "reason": "Middle stage favors coordinated zonal restoration.", "stage": stage}

    return {"task_mode": "stabilization_priority", "confidence": 0.78, "reason": "Late stage stabilization and constraint reduction.", "stage": stage}


def route_llm(client: LLMClient, system_prompt: str, router_prompt: str, routing_context: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": router_prompt + "\n\nContext:\n" + json.dumps(routing_context, indent=2)},
    ]
    route = client.chat_json(messages, response_kind="router")
    for key in ["task_mode", "confidence", "reason", "stage"]:
        if key not in route:
            raise ValueError(f"Router response missing key: {key}")
    if route["task_mode"] not in TASKS:
        raise ValueError(f"Router returned invalid task_mode: {route['task_mode']}")
    route["source"] = "llm"
    route["model"] = client.reasoner_model
    route["llm_task_mode_raw"] = route["task_mode"]
    route["final_task_mode"] = route["task_mode"]
    route["override_applied"] = False
    route["override_reason"] = ""
    return route

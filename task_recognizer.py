from __future__ import annotations

import json
from collections import Counter
from typing import Any

from llm_client import LLMClient
from task_recognition_prompt import TASK_SET, build_task_recognition_prompt


def summarize_trajectory(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    if not trajectory:
        return {
            "mean_progress_delta": 0.0,
            "invalid_action_rate": 0.0,
            "constraint_violation_rate": 0.0,
            "stage_distribution": {},
            "length": 0,
        }
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


class ScenarioTaskRecognizer:
    """Three-task recognizer for recovery scenarios."""

    def recognize_rule(self, routing_context: dict[str, Any]) -> dict[str, Any]:
        env = routing_context.get("env_summary", {})
        traj = routing_context.get("trajectory_summary", {})

        comm = float(env.get("communication_recovery_ratio", 0.0))
        power = float(env.get("power_recovery_ratio", 0.0))
        road = float(env.get("road_recovery_ratio", 0.0))
        shortfall = float(env.get("critical_load_shortfall", 1.0))
        material = float(env.get("material_stock", 1.0))
        wait_usage = float(traj.get("action_category_distribution", {}).get("wait", 0.0))
        mean_progress = float(traj.get("mean_progress_delta", 0.0))
        violation = float(traj.get("constraint_violation_rate", 0.0))

        avg = (comm + power + road) / 3.0
        stage = "early" if avg < 0.35 else "middle" if avg < 0.75 else "late"

        if violation > 0.28 or material < 0.16 or min(comm, road) < 0.50:
            return {
                "task_mode": "restoration_capability_priority",
                "confidence": 0.84,
                "reason": "Capability and feasibility constraints dominate the scenario.",
                "stage": stage,
                "source": "rule",
            }
        if shortfall > 0.42 and power < 0.58:
            return {
                "task_mode": "critical_load_priority",
                "confidence": 0.86,
                "reason": "Critical-load shortfall remains the dominant completion blocker.",
                "stage": stage,
                "source": "rule",
            }
        if wait_usage > 0.38 and mean_progress < 0.008:
            return {
                "task_mode": "global_efficiency_priority",
                "confidence": 0.82,
                "reason": "Wait-overuse with low progress suggests a global finishing bottleneck.",
                "stage": stage,
                "source": "rule",
            }
        return {
            "task_mode": "global_efficiency_priority",
            "confidence": 0.78,
            "reason": "Balanced but incomplete recovery favors global efficiency finishing.",
            "stage": stage,
            "source": "rule",
        }

    def recognize_with_llm(
        self,
        client: LLMClient,
        system_prompt: str,
        routing_context: dict[str, Any],
        previous_task: str = "",
        previous_round_failed: bool = False,
    ) -> dict[str, Any]:
        user_prompt = build_task_recognition_prompt(
            routing_context=routing_context,
            previous_task=previous_task,
            previous_round_failed=previous_round_failed,
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        route = client.chat_json(messages, response_kind="router")
        for key in ["task_mode", "confidence", "reason", "stage"]:
            if key not in route:
                raise ValueError(f"Recognizer response missing key: {key}")
        if route["task_mode"] not in TASK_SET:
            raise ValueError(f"Recognizer returned invalid task_mode: {route['task_mode']}")
        route["source"] = "llm"
        route["model"] = client._select_model("router")
        return route

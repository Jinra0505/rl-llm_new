from __future__ import annotations

import json
from typing import Any

TASK_SET = [
    "critical_load_priority",
    "restoration_capability_priority",
    "global_efficiency_priority",
]


def build_task_recognition_prompt(
    decision_features: dict[str, Any],
    previous_task: str = "",
    previous_round_failed: bool = False,
) -> str:
    """Build strict, feature-first routing prompt."""
    return (
        "You are a strict tri-task recognizer.\n"
        "Use ONLY the structured feature table below.\n"
        "Select exactly one task_mode from:\n"
        "- critical_load_priority\n"
        "- restoration_capability_priority\n"
        "- global_efficiency_priority\n\n"
        "Task boundaries:\n"
        "A) critical_load_priority: choose only when critical-load gap itself is dominant.\n"
        "   Do not choose it just because text mentions critical/key load.\n"
        "   If unmet critical load is mainly caused by weak capability/resources, do not choose critical first.\n"
        "B) restoration_capability_priority: choose when restoration capability is dominant bottleneck\n"
        "   (material/backbone/path/connectivity/feasible-action constraints).\n"
        "   Even with unmet critical load, choose restoration if capability bottleneck is primary.\n"
        "C) global_efficiency_priority: choose when critical gap and capability bottleneck are no longer dominant\n"
        "   and main issue is finishing coordination/global efficiency.\n\n"
        "Short exemplars:\n"
        "- material+backbone constrained, critical still unmet -> restoration_capability_priority\n"
        "- capability sufficient, critical shortfall still dominant -> critical_load_priority\n"
        "- near-finish, cross-layer coordination inefficiency -> global_efficiency_priority\n\n"
        "Output JSON only (no markdown, no extra text).\n"
        "Required keys only:\n"
        "{\"task_mode\":\"...\",\"confidence\":0.0,\"dominant_signal\":\"...\",\"competing_signal\":\"...\",\"reason\":\"one sentence\"}\n\n"
        f"Previous task: {previous_task or 'none'}\n"
        f"Previous round failed: {str(previous_round_failed).lower()}\n"
        "Structured feature table JSON:\n"
        f"{json.dumps(decision_features, ensure_ascii=False, indent=2)}"
    )

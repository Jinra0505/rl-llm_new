"""Offline LLM decision proxy for downstream validation (no API call)."""

from typing import Any


def recognize_with_offline_proxy(routing_context: dict[str, Any], oracle_task: str | None = None) -> dict[str, Any]:
    cue = str(routing_context.get("semantic_cue", "")).lower()
    profile = str(routing_context.get("definition_profile", "default"))

    if oracle_task:
        task = oracle_task
    elif "backbone" in cue or "mobility" in cue or "material bottleneck" in cue:
        task = "restoration_capability_priority"
    elif "finishing" in cue or "closeout" in cue or "coordination" in cue:
        task = "global_efficiency_priority"
    elif "critical" in cue:
        task = "critical_load_priority"
    elif profile == "shifted_finish_coordination":
        task = "global_efficiency_priority"
    else:
        task = "critical_load_priority"

    return {
        "task_mode": task,
        "confidence": 0.78,
        "dominant_signal": "offline_llm_semantic_proxy",
        "competing_signal": "rule_boundary_overlap",
        "reason": "Offline LLM decision proxy for downstream validation (no API).",
        "source": "offline_llm_proxy",
    }

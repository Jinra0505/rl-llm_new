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
    definition_profile: str = "default",
) -> str:
    """Build strict, feature-first routing prompt."""
    definition_guidance = ""
    if definition_profile == "shifted_finish_coordination":
        definition_guidance = (
            "Definition-shift guidance:\n"
            "- global_efficiency_priority emphasizes cross-layer finishing coordination and system-level closeout.\n"
            "- restoration_capability_priority emphasizes backbone/mobility/material bottlenecks that block feasible actions.\n"
            "- if numeric scores are close, use scenario_semantic_cue as tie-breaker.\n"
            "- if scenario_semantic_cue explicitly mentions backbone/mobility/material bottleneck, prefer restoration_capability_priority.\n"
            "- if scenario_semantic_cue explicitly mentions coordinated closeout/finishing choreography, prefer global_efficiency_priority.\n\n"
        )
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
        "Confidence calibration guidance:\n"
        "- clear dominance (score_margin >= 0.12): confidence >= 0.72\n"
        "- moderate boundary (0.06 <= score_margin < 0.12): confidence 0.58~0.72\n"
        "- ambiguous (score_margin < 0.06): confidence <= 0.58\n\n"
        f"{definition_guidance}"
        f"Previous task: {previous_task or 'none'}\n"
        f"Previous round failed: {str(previous_round_failed).lower()}\n"
        "Structured feature table JSON:\n"
        f"{json.dumps(decision_features, ensure_ascii=False, indent=2)}"
    )

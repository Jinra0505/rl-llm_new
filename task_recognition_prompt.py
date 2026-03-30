from __future__ import annotations

import json
from typing import Any

TASK_SET = [
    "critical_load_priority",
    "restoration_capability_priority",
    "global_efficiency_priority",
]


def build_task_recognition_prompt(
    routing_context: dict[str, Any],
    previous_task: str = "",
    previous_round_failed: bool = False,
) -> str:
    """Builds the recognition prompt for strict 3-task scenario classification."""
    return (
        "You are a strict scenario recognizer for a tri-layer recovery environment.\n"
        "Select exactly one task_mode from:\n"
        "- critical_load_priority\n"
        "- restoration_capability_priority\n"
        "- global_efficiency_priority\n\n"
        "Step-by-step decision rules:\n"
        "1) Check dominant bottleneck first.\n"
        "2) Compare strongest signal vs competing signal.\n"
        "3) Output only compact JSON.\n\n"
        "Task boundary definitions:\n"
        "A) critical_load_priority:\n"
        "- dominant contradiction is unmet critical load (high shortfall / key nodes not restored).\n"
        "- even if other layers are imperfect, choose this when critical-load gap is clearly largest.\n"
        "- if capability constraints are the root cause of unmet load, do not choose critical first.\n"
        "B) restoration_capability_priority:\n"
        "- dominant contradiction is insufficient capability to continue restoration.\n"
        "- examples: weak material, low connectivity, poor feasible-action capability, backbone/resource/path bottlenecks.\n"
        "- focus is 'can we continue restoring now', not final finishing target.\n"
        "- if unmet critical load is mostly caused by these capability bottlenecks, choose restoration.\n"
        "C) global_efficiency_priority:\n"
        "- critical load and core capability are no longer dominant bottlenecks.\n"
        "- dominant goal is finishing-stage coordination and global efficiency optimization.\n\n"
        "Prohibitions:\n"
        "- Do NOT choose critical_load_priority just because text mentions 'critical'.\n"
        "- If critical load is already mostly recovered and remaining issue is finishing coordination, prefer global_efficiency_priority.\n"
        "- If the main issue is resource/connectivity/capability shortage, prefer restoration_capability_priority.\n"
        "- In conflicting signals, avoid defaulting to critical_load_priority without clear dominant evidence.\n\n"
        "Very short exemplars:\n"
        "- Ex1: critical shortfall very high; power/comm moderate -> critical_load_priority\n"
        "- Ex2: material/backbone/feasible actions severely constrained -> restoration_capability_priority\n"
        "- Ex3: critical mostly recovered, system near-finish, cross-layer coordination inefficiency -> global_efficiency_priority\n"
        "- Ex4 (boundary): backlog exists but materials/backbone/actions are insufficient -> restoration_capability_priority\n"
        "- Ex5 (boundary): backlog exists, materials/backbone/actions are sufficient, issue is finishing coordination -> global_efficiency_priority\n\n"
        "Return JSON only. No markdown. No extra text.\n"
        "Schema (fixed keys only):\n"
        "{\n"
        '  "task_mode": "...",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "1-2 short sentences",\n'
        '  "dominant_signal": "short phrase",\n'
        '  "competing_signal": "short phrase"\n'
        "}\n\n"
        f"Previous task: {previous_task or 'none'}\n"
        f"Previous round failed: {str(previous_round_failed).lower()}\n"
        "Context JSON:\n"
        f"{json.dumps(routing_context, indent=2)}"
    )

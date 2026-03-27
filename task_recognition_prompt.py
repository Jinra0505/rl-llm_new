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
        "You are a scenario recognizer for a tri-layer recovery environment.\n"
        "You MUST select exactly one task_mode from:\n"
        "- critical_load_priority\n"
        "- restoration_capability_priority\n"
        "- global_efficiency_priority\n\n"
        "Recognition objective:\n"
        "- identify the dominant bottleneck for the next outer-loop round\n"
        "- avoid conservative stagnation\n"
        "- choose a task that improves completion-oriented progress\n\n"
        "Switching rule:\n"
        "- if previous round underperformed (low success/progress), prefer switching tasks unless strong evidence supports staying\n"
        "- if wait_hold overuse + low progress is present, avoid repeating the same stalled mode\n\n"
        "Return STRICT JSON with keys:\n"
        "task_mode, confidence, reason, stage\n\n"
        f"Previous task: {previous_task or 'none'}\n"
        f"Previous round failed: {str(previous_round_failed).lower()}\n"
        "Context JSON:\n"
        f"{json.dumps(routing_context, indent=2)}"
    )

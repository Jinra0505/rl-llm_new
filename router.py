from __future__ import annotations

"""Backward-compatible wrappers around the explicit task recognizer modules."""

from typing import Any

from llm_client import LLMClient
from task_recognizer import ScenarioTaskRecognizer, summarize_trajectory

TASKS = [
    "critical_load_priority",
    "restoration_capability_priority",
    "global_efficiency_priority",
]

_RECOGNIZER = ScenarioTaskRecognizer()


def route_rule(routing_context: dict[str, Any]) -> dict[str, Any]:
    return _RECOGNIZER.recognize_rule(routing_context)


def route_llm(client: LLMClient, system_prompt: str, router_prompt: str, routing_context: dict[str, Any]) -> dict[str, Any]:
    _ = router_prompt  # Prompt now constructed by task_recognition_prompt.build_task_recognition_prompt.
    return _RECOGNIZER.recognize_with_llm(client=client, system_prompt=system_prompt, routing_context=routing_context)

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_client import LLMClient
from prompts import SYSTEM_PROMPT
from task_recognizer import ScenarioTaskRecognizer


DEFAULT_CONTEXT: dict[str, Any] = {
    "env_summary": {
        "communication_recovery_ratio": 0.45,
        "power_recovery_ratio": 0.62,
        "road_recovery_ratio": 0.40,
        "critical_load_shortfall": 0.38,
        "material_stock": 0.22,
    },
    "trajectory_summary": {
        "mean_progress_delta": 0.002,
        "constraint_violation_rate": 0.03,
        "action_category_distribution": {"wait": 0.35},
    },
}


def load_context(path: str) -> dict[str, Any]:
    if not path:
        return dict(DEFAULT_CONTEXT)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-task recognition-only runner.")
    parser.add_argument("--mode", choices=["rule", "llm"], default="rule")
    parser.add_argument("--input-json", default="", help="Path to a JSON file with env_summary and trajectory_summary.")
    parser.add_argument("--output-json", default="", help="Optional path to write recognition output JSON.")
    parser.add_argument("--llm-mode", choices=["real"], default="real")
    args = parser.parse_args()

    context = load_context(args.input_json)
    recognizer = ScenarioTaskRecognizer()

    if args.mode == "rule":
        result = recognizer.recognize_rule(context)
    else:
        client = LLMClient(mode=args.llm_mode)
        client.preflight_check()
        result = recognizer.recognize_with_llm(
            client=client,
            system_prompt=SYSTEM_PROMPT,
            routing_context=context,
            previous_task="",
            previous_round_failed=False,
        )

    print(json.dumps(result, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

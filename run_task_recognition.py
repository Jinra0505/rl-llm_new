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

TASKS = ["critical_load_priority", "restoration_capability_priority", "global_efficiency_priority"]


def load_context(path: str) -> dict[str, Any]:
    if not path:
        return dict(DEFAULT_CONTEXT)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mk_critical(i: int) -> dict[str, Any]:
    return {
        "label": "critical_load_priority",
        "env_summary": {
            "communication_recovery_ratio": 0.54 + 0.02 * (i % 3),
            "power_recovery_ratio": 0.46 + 0.02 * (i % 2),
            "road_recovery_ratio": 0.55 + 0.02 * (i % 3),
            "critical_load_shortfall": 0.45 + 0.12 * (i % 5) / 5,
            "material_stock": 0.24 + 0.10 * (i % 4) / 4,
        },
        "trajectory_summary": {
            "mean_progress_delta": 0.004 + 0.002 * (i % 3) / 3,
            "constraint_violation_rate": 0.08 + 0.08 * (i % 4) / 4,
            "action_category_distribution": {"wait": 0.18 + 0.08 * (i % 3) / 3},
        },
    }


def _mk_restoration(i: int) -> dict[str, Any]:
    return {
        "label": "restoration_capability_priority",
        "env_summary": {
            "communication_recovery_ratio": 0.40 + 0.08 * (i % 5) / 5,
            "power_recovery_ratio": 0.56 + 0.04 * (i % 4) / 4,
            "road_recovery_ratio": 0.41 + 0.08 * (i % 5) / 5,
            "critical_load_shortfall": 0.30 + 0.10 * (i % 4) / 4,
            "material_stock": 0.10 + 0.10 * (i % 5) / 5,
        },
        "trajectory_summary": {
            "mean_progress_delta": 0.0035 + 0.0015 * (i % 3) / 3,
            "constraint_violation_rate": 0.18 + 0.14 * (i % 5) / 5,
            "action_category_distribution": {"wait": 0.14 + 0.10 * (i % 4) / 4},
        },
    }


def _mk_global(i: int, hard: bool = False) -> dict[str, Any]:
    return {
        "label": "global_efficiency_priority",
        "env_summary": {
            "communication_recovery_ratio": (0.66 if not hard else 0.62) + 0.08 * (i % 5) / 5,
            "power_recovery_ratio": (0.66 if not hard else 0.62) + 0.08 * (i % 5) / 5,
            "road_recovery_ratio": (0.64 if not hard else 0.60) + 0.08 * (i % 5) / 5,
            "critical_load_shortfall": 0.20 + 0.12 * (i % 5) / 5,
            "material_stock": (0.24 if not hard else 0.20) + 0.14 * (i % 5) / 5,
        },
        "trajectory_summary": {
            "mean_progress_delta": (0.0026 if not hard else 0.0020) + 0.0012 * (i % 4) / 4,
            "constraint_violation_rate": 0.05 + 0.08 * (i % 4) / 4,
            "action_category_distribution": {"wait": 0.32 + 0.16 * (i % 5) / 5},
        },
    }


def build_eval_sets() -> dict[str, list[dict[str, Any]]]:
    internal = [_mk_critical(i) for i in range(6)] + [_mk_restoration(i) for i in range(6)] + [_mk_global(i) for i in range(6)]
    independent = [_mk_critical(i) for i in range(15)] + [_mk_restoration(i) for i in range(15)] + [_mk_global(i) for i in range(15)]
    hard = [_mk_critical(i + 20) for i in range(15)] + [_mk_restoration(i + 20) for i in range(15)] + [_mk_global(i + 20, hard=True) for i in range(15)]
    # add light noisy/missing-conflict cases into hard set
    hard.append({"label": "restoration_capability_priority", "env_summary": {"critical_load_shortfall": 0.36, "material_stock": 0.11}, "trajectory_summary": {"mean_progress_delta": 0.003, "constraint_violation_rate": 0.28, "action_category_distribution": {"wait": 0.27}}})
    hard.append({"label": "global_efficiency_priority", "env_summary": {"communication_recovery_ratio": 0.72, "power_recovery_ratio": 0.70, "road_recovery_ratio": 0.68, "critical_load_shortfall": 0.24, "material_stock": 0.30}, "trajectory_summary": {"mean_progress_delta": 0.0018, "constraint_violation_rate": 0.05, "action_category_distribution": {"wait": 0.48}}})
    hard.append({"label": "critical_load_priority", "env_summary": {"communication_recovery_ratio": 0.57, "power_recovery_ratio": 0.49, "road_recovery_ratio": 0.56, "critical_load_shortfall": 0.59, "material_stock": 0.26}, "trajectory_summary": {"mean_progress_delta": 0.0035, "constraint_violation_rate": 0.12, "action_category_distribution": {"wait": 0.19}}})
    return {"internal": internal, "independent": independent, "hard": hard}


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    out = []
    for c in TASKS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        out.append((2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0)
    return float(sum(out) / len(out))


def _confusion(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    m = {c: {d: 0 for d in TASKS} for c in TASKS}
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


def _recall(y_true: list[str], y_pred: list[str], label: str) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    return tp / (tp + fn) if (tp + fn) else 0.0


def _precision(y_true: list[str], y_pred: list[str], label: str) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    return tp / (tp + fp) if (tp + fp) else 0.0


def eval_set(samples: list[dict[str, Any]], mode: str, recognizer: ScenarioTaskRecognizer, client: LLMClient | None = None) -> dict[str, Any]:
    y_true = [str(s["label"]) for s in samples]
    y_pred: list[str] = []
    second_pass_used = 0
    for s in samples:
        ctx = {"env_summary": s["env_summary"], "trajectory_summary": s["trajectory_summary"]}
        if mode == "rule":
            r = recognizer.recognize_rule(ctx)
        else:
            if client is None:
                raise RuntimeError("LLM mode requires initialized client")
            r = recognizer.recognize_with_llm(client=client, system_prompt=SYSTEM_PROMPT, routing_context=ctx)
            second_pass_used += int(bool(r.get("second_pass_used", False)))
        y_pred.append(str(r["task_mode"]))

    confusion = _confusion(y_true, y_pred)
    rest_to_critical = sum(1 for t, p in zip(y_true, y_pred) if t == "restoration_capability_priority" and p == "critical_load_priority")
    rest_global_confusion = sum(
        1
        for t, p in zip(y_true, y_pred)
        if (t == "restoration_capability_priority" and p == "global_efficiency_priority")
        or (t == "global_efficiency_priority" and p == "restoration_capability_priority")
    )

    return {
        "accuracy": sum(int(t == p) for t, p in zip(y_true, y_pred)) / float(len(y_true)),
        "macro_f1": _macro_f1(y_true, y_pred),
        "confusion_matrix": confusion,
        "restoration_recall": _recall(y_true, y_pred, "restoration_capability_priority"),
        "global_recall": _recall(y_true, y_pred, "global_efficiency_priority"),
        "critical_precision": _precision(y_true, y_pred, "critical_load_priority"),
        "restoration_to_critical_errors": rest_to_critical,
        "restoration_global_confusions": rest_global_confusion,
        "second_pass_used": second_pass_used,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-task recognition-only runner.")
    parser.add_argument("--mode", choices=["rule", "llm", "eval"], default="rule")
    parser.add_argument("--eval-set", choices=["internal", "independent", "hard", "all"], default="all")
    parser.add_argument("--input-json", default="", help="Path to a JSON file with env_summary and trajectory_summary.")
    parser.add_argument("--output-json", default="", help="Optional path to write recognition output JSON.")
    parser.add_argument("--llm-mode", choices=["real"], default="real")
    args = parser.parse_args()

    recognizer = ScenarioTaskRecognizer()

    if args.mode in {"rule", "llm"}:
        context = load_context(args.input_json)
        if args.mode == "rule":
            result = recognizer.recognize_rule(context)
        else:
            client = LLMClient(mode=args.llm_mode)
            client.preflight_chat_model()
            result = recognizer.recognize_with_llm(
                client=client,
                system_prompt=SYSTEM_PROMPT,
                routing_context=context,
                previous_task="",
                previous_round_failed=False,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        sets = build_eval_sets()
        target_sets = sets.keys() if args.eval_set == "all" else [args.eval_set]
        client = LLMClient(mode=args.llm_mode)
        client.preflight_check()
        result = {s: {"rule": eval_set(sets[s], "rule", recognizer), "llm": eval_set(sets[s], "llm", recognizer, client)} for s in target_sets}
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

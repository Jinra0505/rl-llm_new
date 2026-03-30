from __future__ import annotations

import argparse
import json
import random
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
    paraphrase = build_paraphrase_set()
    shuffled = build_shuffled_feature_order_set()
    conflict_sparse = build_conflict_sparse_set()
    uncertain = build_uncertain_set()
    ood_shifted = build_ood_shifted_set()
    definition_shift = build_definition_shift_set()
    return {
        "internal": internal,
        "independent": independent,
        "hard": hard,
        "paraphrase": paraphrase,
        "shuffled_feature_order": shuffled,
        "conflict_sparse": conflict_sparse,
        "uncertain": uncertain,
        "ood_shifted": ood_shifted,
        "definition_shift": definition_shift,
    }


def _with_note(sample: dict[str, Any], note: str) -> dict[str, Any]:
    s = json.loads(json.dumps(sample))
    s["note"] = note
    return s


def build_paraphrase_set() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for i in range(10):
        samples.append(_with_note(_mk_critical(i + 40), "critical gap remains primary even if network gets partial uplift"))
        samples.append(_with_note(_mk_restoration(i + 40), "capability bottleneck dominates; restoring means and access should come first"))
        samples.append(_with_note(_mk_global(i + 40), "main issue is endgame coordination and finishing efficiency"))
    return samples


def build_shuffled_feature_order_set() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for i in range(10):
        s1 = _mk_critical(i + 60)
        s1["feature_order_mode"] = "shuffled"
        s1["feature_order_seed"] = 100 + i
        samples.append(s1)
        s2 = _mk_restoration(i + 60)
        s2["feature_order_mode"] = "shuffled"
        s2["feature_order_seed"] = 200 + i
        samples.append(s2)
        s3 = _mk_global(i + 60)
        s3["feature_order_mode"] = "shuffled"
        s3["feature_order_seed"] = 300 + i
        samples.append(s3)
    return samples


def build_conflict_sparse_set() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    rnd = random.Random(7)
    for i in range(10):
        c = _mk_critical(i + 80)
        # sparse fields + weak conflict noise
        c["env_summary"].pop("communication_recovery_ratio", None)
        c["trajectory_summary"]["constraint_violation_rate"] = 0.05 + 0.02 * rnd.random()
        samples.append(c)

        r = _mk_restoration(i + 80)
        r["env_summary"].pop("road_recovery_ratio", None)
        r["env_summary"]["critical_load_shortfall"] = 0.28 + 0.06 * rnd.random()
        samples.append(r)

        g = _mk_global(i + 80, hard=True)
        g["trajectory_summary"]["mean_progress_delta"] = 0.0018 + 0.0008 * rnd.random()
        g["env_summary"]["material_stock"] = 0.24 + 0.08 * rnd.random()
        samples.append(g)
    return samples


def build_uncertain_set() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for i in range(10):
        # critical vs restoration ambiguous: rely on semantic cue
        c = {
            "label": "critical_load_priority",
            "env_summary": {
                "communication_recovery_ratio": 0.51 + 0.02 * (i % 2),
                "power_recovery_ratio": 0.53 + 0.02 * ((i + 1) % 2),
                "road_recovery_ratio": 0.52 + 0.01 * (i % 3),
                "critical_load_shortfall": 0.40 + 0.03 * (i % 3),
                "material_stock": 0.20 + 0.02 * (i % 3),
            },
            "trajectory_summary": {
                "mean_progress_delta": 0.0030 + 0.0004 * (i % 2),
                "constraint_violation_rate": 0.11 + 0.01 * (i % 3),
                "action_category_distribution": {"wait": 0.22 + 0.03 * (i % 2)},
            },
            "semantic_cue": "despite capability strain, operators state critical-load gap is first-order objective",
        }
        samples.append(c)
        r = dict(c)
        r["label"] = "restoration_capability_priority"
        r["semantic_cue"] = "dispatch notes show backbone and material bottleneck must be cleared before load closure"
        samples.append(r)

        g = {
            "label": "global_efficiency_priority",
            "env_summary": {
                "communication_recovery_ratio": 0.60 + 0.02 * (i % 2),
                "power_recovery_ratio": 0.60 + 0.02 * ((i + 1) % 2),
                "road_recovery_ratio": 0.59 + 0.02 * (i % 2),
                "critical_load_shortfall": 0.28 + 0.03 * (i % 3),
                "material_stock": 0.23 + 0.02 * (i % 2),
            },
            "trajectory_summary": {
                "mean_progress_delta": 0.0020 + 0.0004 * (i % 3),
                "constraint_violation_rate": 0.08 + 0.02 * (i % 2),
                "action_category_distribution": {"wait": 0.32 + 0.03 * (i % 2)},
            },
            "semantic_cue": "field command emphasizes cross-layer finishing choreography rather than capability rescue",
        }
        samples.append(g)
    return samples


def build_ood_shifted_set() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for i in range(10):
        # non-template mixes; still same task intent
        samples.append(
            {
                "label": "critical_load_priority",
                "env_summary": {
                    "communication_recovery_ratio": 0.70 - 0.05 * (i % 2),
                    "power_recovery_ratio": 0.44 + 0.01 * (i % 3),
                    "road_recovery_ratio": 0.66 - 0.03 * (i % 2),
                    "critical_load_shortfall": 0.52 - 0.04 * (i % 3),
                    "material_stock": 0.33 - 0.04 * (i % 2),
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0038 + 0.0005 * (i % 2),
                    "constraint_violation_rate": 0.06 + 0.02 * (i % 2),
                    "action_category_distribution": {"wait": 0.20 + 0.04 * (i % 2)},
                },
                "semantic_cue": "high comm does not close critical feeders; shortfall is operational blocker",
            }
        )
        samples.append(
            {
                "label": "restoration_capability_priority",
                "env_summary": {
                    "communication_recovery_ratio": 0.58 + 0.04 * (i % 2),
                    "power_recovery_ratio": 0.61 + 0.03 * (i % 2),
                    "road_recovery_ratio": 0.43 + 0.02 * (i % 3),
                    "critical_load_shortfall": 0.27 + 0.04 * (i % 2),
                    "material_stock": 0.11 + 0.03 * (i % 3),
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0025 + 0.0004 * (i % 2),
                    "constraint_violation_rate": 0.19 + 0.03 * (i % 2),
                    "action_category_distribution": {"wait": 0.19 + 0.03 * (i % 2)},
                },
                "semantic_cue": "mobility corridor and backbone patching are the limiting factors",
            }
        )
        samples.append(
            {
                "label": "global_efficiency_priority",
                "env_summary": {
                    "communication_recovery_ratio": 0.68 + 0.03 * (i % 2),
                    "power_recovery_ratio": 0.70 + 0.02 * (i % 2),
                    "road_recovery_ratio": 0.66 + 0.02 * (i % 2),
                    "critical_load_shortfall": 0.21 + 0.03 * (i % 2),
                    "material_stock": 0.29 + 0.02 * (i % 2),
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0018 + 0.0004 * (i % 2),
                    "constraint_violation_rate": 0.06 + 0.01 * (i % 2),
                    "action_category_distribution": {"wait": 0.40 + 0.05 * (i % 2)},
                },
                "semantic_cue": "capability baseline is sufficient; issue is cross-layer finish orchestration",
            }
        )
    return samples


def build_definition_shift_set() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for i in range(10):
        samples.append(
            {
                "label": "critical_load_priority",
                "definition_profile": "shifted_finish_coordination",
                "env_summary": {
                    "communication_recovery_ratio": 0.58 + 0.01 * (i % 2),
                    "power_recovery_ratio": 0.51 + 0.02 * (i % 2),
                    "road_recovery_ratio": 0.57 + 0.01 * (i % 2),
                    "critical_load_shortfall": 0.47 + 0.03 * (i % 2),
                    "material_stock": 0.27 + 0.02 * (i % 2),
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0031 + 0.0003 * (i % 2),
                    "constraint_violation_rate": 0.09 + 0.01 * (i % 2),
                    "action_category_distribution": {"wait": 0.24 + 0.02 * (i % 2)},
                },
                "semantic_cue": "definition-shift note: prioritize unresolved critical service obligations first",
            }
        )
        samples.append(
            {
                "label": "restoration_capability_priority",
                "definition_profile": "shifted_finish_coordination",
                "env_summary": {
                    "communication_recovery_ratio": 0.58 + 0.02 * (i % 2),
                    "power_recovery_ratio": 0.60 + 0.02 * (i % 2),
                    "road_recovery_ratio": 0.53 + 0.02 * (i % 2),
                    "critical_load_shortfall": 0.24 + 0.03 * (i % 2),
                    "material_stock": 0.12 + 0.02 * (i % 2),
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0022 + 0.0003 * (i % 2),
                    "constraint_violation_rate": 0.16 + 0.02 * (i % 2),
                    "action_category_distribution": {"wait": 0.28 + 0.02 * (i % 2)},
                },
                "semantic_cue": "definition-shift note: backbone mobility/material bottleneck is explicit priority",
            }
        )
        samples.append(
            {
                "label": "global_efficiency_priority",
                "definition_profile": "shifted_finish_coordination",
                "env_summary": {
                    "communication_recovery_ratio": 0.67 + 0.02 * (i % 2),
                    "power_recovery_ratio": 0.68 + 0.02 * (i % 2),
                    "road_recovery_ratio": 0.66 + 0.02 * (i % 2),
                    "critical_load_shortfall": 0.24 + 0.03 * (i % 2),
                    "material_stock": 0.28 + 0.02 * (i % 2),
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0019 + 0.0003 * (i % 2),
                    "constraint_violation_rate": 0.06 + 0.01 * (i % 2),
                    "action_category_distribution": {"wait": 0.42 + 0.02 * (i % 2)},
                },
                "semantic_cue": "definition-shift note: emphasize coordinated cross-layer closeout and finish quality",
            }
        )
    return samples


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
    hybrid_used_llm = 0
    for s in samples:
        ctx = {
            "env_summary": s["env_summary"],
            "trajectory_summary": s["trajectory_summary"],
            "semantic_cue": s.get("semantic_cue", ""),
            "definition_profile": s.get("definition_profile", "default"),
        }
        if mode == "rule":
            r = recognizer.recognize_rule(ctx)
        elif mode == "hybrid":
            if client is None:
                raise RuntimeError("Hybrid mode requires initialized client")
            r = recognizer.recognize_hybrid(
                client=client,
                system_prompt=SYSTEM_PROMPT,
                routing_context=ctx,
                definition_profile=str(s.get("definition_profile", "default")),
            )
            hybrid_used_llm += int(bool(r.get("hybrid_used_llm", False)))
        else:
            if client is None:
                raise RuntimeError("LLM mode requires initialized client")
            r = recognizer.recognize_with_llm(
                client=client,
                system_prompt=SYSTEM_PROMPT,
                routing_context=ctx,
                feature_order_mode=str(s.get("feature_order_mode", "stable")),
                feature_order_seed=int(s.get("feature_order_seed", 0)),
                definition_profile=str(s.get("definition_profile", "default")),
            )
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
        "hybrid_used_llm_count": hybrid_used_llm,
        "hybrid_used_llm_ratio": (hybrid_used_llm / float(len(y_true))) if y_true else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-task recognition-only runner.")
    parser.add_argument("--mode", choices=["rule", "llm", "eval"], default="rule")
    parser.add_argument(
        "--eval-set",
        choices=[
            "internal",
            "independent",
            "hard",
            "paraphrase",
            "shuffled_feature_order",
            "conflict_sparse",
            "uncertain",
            "ood_shifted",
            "definition_shift",
            "all",
        ],
        default="all",
    )
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
        result = {
            s: {
                "rule": eval_set(sets[s], "rule", recognizer),
                "llm": eval_set(sets[s], "llm", recognizer, client),
                "hybrid": eval_set(sets[s], "hybrid", recognizer, client),
            }
            for s in target_sets
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

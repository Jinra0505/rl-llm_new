from __future__ import annotations

import json
import random
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
            "action_category_distribution": {},
        }
    n = len(trajectory)
    progress = [float(t.get("info", {}).get("progress_delta", 0.0)) for t in trajectory]
    invalid = [1.0 if t.get("info", {}).get("invalid_action", False) else 0.0 for t in trajectory]
    violate = [1.0 if t.get("info", {}).get("constraint_violation", False) else 0.0 for t in trajectory]
    stages = [t.get("info", {}).get("stage", "unknown") for t in trajectory]
    actions = [int(t.get("action", -1)) for t in trajectory]
    wait_rate = sum(1 for a in actions if a == 14) / float(max(1, n))
    return {
        "mean_progress_delta": sum(progress) / n,
        "invalid_action_rate": sum(invalid) / n,
        "constraint_violation_rate": sum(violate) / n,
        "stage_distribution": {k: v / n for k, v in Counter(stages).items()},
        "action_category_distribution": {"wait": wait_rate},
        "length": n,
    }


class ScenarioTaskRecognizer:
    """Three-task recognizer for recovery scenarios."""

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def _infer_stage(env_summary: dict[str, Any]) -> str:
        comm = float(env_summary.get("communication_recovery_ratio", 0.0))
        power = float(env_summary.get("power_recovery_ratio", 0.0))
        road = float(env_summary.get("road_recovery_ratio", 0.0))
        avg = (comm + power + road) / 3.0
        return "early" if avg < 0.35 else "middle" if avg < 0.75 else "late"

    def extract_decision_features(self, routing_context: dict[str, Any]) -> dict[str, Any]:
        env = routing_context.get("env_summary", {})
        traj = routing_context.get("trajectory_summary", {})

        comm = float(env.get("communication_recovery_ratio", 0.0))
        power = float(env.get("power_recovery_ratio", 0.0))
        road = float(env.get("road_recovery_ratio", 0.0))
        shortfall = float(env.get("critical_load_shortfall", 1.0))
        material = float(env.get("material_stock", 0.0))

        backbone_comm = float(env.get("backbone_comm_ratio", comm))
        backbone_power = float(env.get("backbone_power_ratio", power))
        backbone_road = float(env.get("backbone_road_ratio", road))
        backbone_ratio = (backbone_comm + backbone_power + backbone_road) / 3.0

        wait_usage = float(traj.get("action_category_distribution", {}).get("wait", 0.0))
        mean_progress = float(traj.get("mean_progress_delta", 0.0))
        violation = float(traj.get("constraint_violation_rate", 0.0))

        min_layer = min(comm, power, road)
        weakest_layer = min({"comm": comm, "power": power, "road": road}.items(), key=lambda kv: kv[1])[0]
        weakest_zone = str(env.get("weakest_zone", "unknown"))
        stage = self._infer_stage(env)

        material_pressure = self._clip01((0.24 - material) / 0.24)
        backbone_pressure = self._clip01((0.62 - backbone_ratio) / 0.62)
        progress_pressure = self._clip01((0.008 - mean_progress) / 0.008)

        critical_gap_score = self._clip01(0.65 * shortfall + 0.20 * self._clip01((0.55 - power) / 0.55) + 0.15 * progress_pressure)
        capability_bottleneck_score = self._clip01(
            0.36 * material_pressure
            + 0.28 * backbone_pressure
            + 0.20 * self._clip01((0.55 - min_layer) / 0.55)
            + 0.16 * self._clip01(violation / 0.30)
        )
        global_finishing_score = self._clip01(
            0.30 * self._clip01((min_layer - 0.55) / 0.45)
            + 0.30 * self._clip01((0.45 - shortfall) / 0.45)
            + 0.20 * self._clip01(wait_usage / 0.50)
            + 0.20 * progress_pressure
        )

        score_map = {
            "critical_load_priority": critical_gap_score,
            "restoration_capability_priority": capability_bottleneck_score,
            "global_efficiency_priority": global_finishing_score,
        }
        sorted_tasks = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        top2 = [sorted_tasks[0][0], sorted_tasks[1][0]]
        score_margin = float(sorted_tasks[0][1] - sorted_tasks[1][1])

        return {
            "critical_load_shortfall": round(shortfall, 4),
            "critical_gap_score": round(critical_gap_score, 4),
            "capability_bottleneck_score": round(capability_bottleneck_score, 4),
            "global_finishing_score": round(global_finishing_score, 4),
            "min_layer_recovery": round(min_layer, 4),
            "backbone_ratio": round(backbone_ratio, 4),
            "backbone_summary": {
                "backbone_comm": round(backbone_comm, 4),
                "backbone_power": round(backbone_power, 4),
                "backbone_road": round(backbone_road, 4),
            },
            "material_stock": round(material, 4),
            "material_pressure": round(material_pressure, 4),
            "wait_usage": round(wait_usage, 4),
            "mean_progress_delta": round(mean_progress, 6),
            "violation_pressure": round(self._clip01(violation / 0.30), 4),
            "weakest_layer": weakest_layer,
            "weakest_zone": weakest_zone,
            "stage": stage,
            "top2_candidate_tasks": top2,
            "score_margin": round(score_margin, 4),
            "coarse_scores": {k: round(v, 4) for k, v in score_map.items()},
        }

    def recognize_rule(self, routing_context: dict[str, Any]) -> dict[str, Any]:
        feat = self.extract_decision_features(routing_context)
        top1 = feat["top2_candidate_tasks"][0]
        return {
            "task_mode": top1,
            "confidence": 0.70,
            "reason": "Feature-score top1 from rule proxy.",
            "dominant_signal": "coarse_score_top1",
            "competing_signal": "coarse_score_top2",
            "stage": feat["stage"],
            "source": "rule",
            "features": feat,
        }

    def _parse_llm_core(self, route: dict[str, Any]) -> dict[str, Any]:
        raw_task = str(route.get("task_mode", "")).strip().lower()
        normalized_task = raw_task
        aliases = {
            "critical_load_priority": "critical_load_priority",
            "critical_load_riority": "critical_load_priority",
            "critical_priority": "critical_load_priority",
            "restoration_capability_priority": "restoration_capability_priority",
            "restoration_priority": "restoration_capability_priority",
            "global_efficiency_priority": "global_efficiency_priority",
            "global_priority": "global_efficiency_priority",
        }
        if raw_task in aliases:
            normalized_task = aliases[raw_task]
        elif raw_task.startswith("critical_load"):
            normalized_task = "critical_load_priority"
        elif raw_task.startswith("restoration"):
            normalized_task = "restoration_capability_priority"
        elif raw_task.startswith("global"):
            normalized_task = "global_efficiency_priority"
        core = {
            "task_mode": normalized_task,
            "confidence": float(route.get("confidence", 0.0)),
            "dominant_signal": str(route.get("dominant_signal", "")),
            "competing_signal": str(route.get("competing_signal", "")),
            "reason": str(route.get("reason", "")),
        }
        return core

    def _should_second_pass(self, first_core: dict[str, Any], features: dict[str, Any]) -> tuple[bool, str]:
        conf = float(first_core.get("confidence", 0.0))
        margin = float(features.get("score_margin", 1.0))
        top2 = features.get("top2_candidate_tasks", [])
        top1 = top2[0] if top2 else ""
        predicted = str(first_core.get("task_mode", ""))
        comp = str(first_core.get("competing_signal", "")).lower()
        hints_top2 = any(t.replace("_priority", "").split("_")[0] in comp for t in top2)
        clear_case = margin >= 0.12 and predicted == top1 and conf >= 0.50
        moderate_clear = margin >= 0.08 and predicted == top1 and conf >= 0.62 and not hints_top2
        if clear_case or moderate_clear:
            return False, ""
        if conf < 0.48:
            return True, "low_confidence"
        if margin < 0.02:
            return True, "small_score_margin"
        if hints_top2 and conf < 0.62:
            return True, "competing_signal_points_to_top2"
        return False, ""

    @staticmethod
    def _reorder_features(features: dict[str, Any], seed: int) -> dict[str, Any]:
        keys = list(features.keys())
        rnd = random.Random(seed)
        rnd.shuffle(keys)
        return {k: features[k] for k in keys}

    def recognize_with_llm(
        self,
        client: LLMClient,
        system_prompt: str,
        routing_context: dict[str, Any],
        previous_task: str = "",
        previous_round_failed: bool = False,
        feature_order_mode: str = "stable",
        feature_order_seed: int = 0,
    ) -> dict[str, Any]:
        features = self.extract_decision_features(routing_context)
        prompt_features = (
            self._reorder_features(features, feature_order_seed)
            if feature_order_mode == "shuffled"
            else features
        )
        user_prompt = build_task_recognition_prompt(
            decision_features=prompt_features,
            previous_task=previous_task,
            previous_round_failed=previous_round_failed,
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        first_raw = client.chat_json(messages, response_kind="router")
        first_core = self._parse_llm_core(first_raw)

        for key in ["task_mode", "confidence", "dominant_signal", "competing_signal", "reason"]:
            if key not in first_core:
                raise ValueError(f"Recognizer response missing key: {key}")
        if first_core["task_mode"] not in TASK_SET:
            raise ValueError(f"Recognizer returned invalid task_mode: {first_core['task_mode']}")

        second_used, second_reason = self._should_second_pass(first_core, features)
        final_core = dict(first_core)

        if second_used:
            top2 = features["top2_candidate_tasks"]
            clarify_prompt = (
                "Second-pass top2 clarification.\n"
                f"Only choose one from: {top2[0]} OR {top2[1]}.\n"
                "Use only the feature table and first-pass signals. Keep output JSON compact.\n"
                "Required keys: task_mode, confidence, dominant_signal, competing_signal, reason.\n"
                f"Feature table JSON:\n{json.dumps(features, ensure_ascii=False, indent=2)}\n"
                f"First-pass JSON:\n{json.dumps(first_core, ensure_ascii=False, indent=2)}"
            )
            second_raw = client.chat_json(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": clarify_prompt}],
                response_kind="router",
            )
            second_core = self._parse_llm_core(second_raw)
            if second_core["task_mode"] not in top2:
                raise ValueError(f"Recognizer second-pass returned task outside top2: {second_core['task_mode']} not in {top2}")
            final_core = second_core

        result = {
            **final_core,
            "stage": features["stage"],
            "source": "llm",
            "model": client._select_model("router"),
            "features": features,
            "top2_candidate_tasks": features["top2_candidate_tasks"],
            "score_margin": features["score_margin"],
            "second_pass_used": bool(second_used),
            "first_pass_task": first_core["task_mode"],
            "final_task": final_core["task_mode"],
            "second_pass_reason": second_reason,
        }
        return result

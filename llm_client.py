from __future__ import annotations

"""Formal-run LLM client (real-only, no automatic mock fallback)."""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any


class LLMClient:
    """DeepSeek-compatible OpenAI client for formal runs.

    Notes:
    - Formal path is strictly real-only.
    - `_mock_response` is kept as test-only helper and is unreachable from formal CLI flows.
    """

    def __init__(
        self,
        mode: str = "real",
        timeout_seconds: int = 45,
        max_retries: int = 2,
        temperature: float = 0.3,
        max_tokens: int = 1400,
    ) -> None:
        self.mode = str(mode).strip().lower()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.api_provider = "deepseek-compatible-openai"
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
        legacy_model = os.getenv("DEEPSEEK_MODEL", "").strip()
        self.chat_model = os.getenv("DEEPSEEK_MODEL_CHAT", legacy_model).strip()
        self.reasoner_model = os.getenv("DEEPSEEK_MODEL_REASONER", self.chat_model).strip()

        self.call_count = 0
        self.call_history: list[dict[str, Any]] = []
        self.last_error: str = ""
        self.preflight_chat_ok = False
        self.preflight_reasoner_ok = False
        self._formal_real_available = False

        self._validate_formal_configuration()

    def _validate_formal_configuration(self) -> None:
        if self.mode != "real":
            raise RuntimeError(f"Formal run requires llm_mode=real, got: {self.mode}")
        if not self.api_key or not self.base_url or not self.chat_model or not self.reasoner_model:
            self._formal_real_available = False
            return
        try:
            from openai import OpenAI  # noqa: F401
        except ModuleNotFoundError as exc:
            _ = exc
            self._formal_real_available = False
            return
        self._formal_real_available = True

    @property
    def using_mock(self) -> bool:
        return not self._formal_real_available

    def effective_mode(self) -> str:
        return "real" if self._formal_real_available else "mock"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _record_call(
        self,
        *,
        response_kind: str,
        model: str,
        success: bool,
        latency_sec: float,
        error: str = "",
    ) -> None:
        self.call_history.append(
            {
                "timestamp": self._now(),
                "response_kind": response_kind,
                "model": model,
                "success": bool(success),
                "latency_sec": float(latency_sec),
                "error": error,
            }
        )
        self.call_count += 1
        if error:
            self.last_error = error

    def chat(self, messages: list[dict[str, str]], response_kind: str = "chat", sample_idx: int = 0) -> str:
        if self.using_mock:
            t0 = time.time()
            content = self._mock_response(response_kind=response_kind, sample_idx=sample_idx)
            self._record_call(
                response_kind=response_kind,
                model=self._mock_model_name(response_kind),
                success=True,
                latency_sec=time.time() - t0,
                error="",
            )
            return content
        return self._real_chat(messages, response_kind=response_kind)

    def chat_json(self, messages: list[dict[str, str]], response_kind: str = "chat", sample_idx: int = 0) -> dict[str, Any]:
        raw = self.chat(messages, response_kind=response_kind, sample_idx=sample_idx)
        candidates: list[str] = [raw.strip()]
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            candidates.append("\n".join(lines).strip())
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(raw[start : end + 1])

        for text in candidates:
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj

        snippet = raw[:400].replace("\n", "\\n")
        raise RuntimeError(f"{response_kind} JSON parse failed under real LLM run. Raw snippet: {snippet}")

    def _normalize_base_url(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"

    def _select_model(self, response_kind: str) -> str:
        if response_kind in {"router", "feedback"}:
            return self.reasoner_model or self.chat_model
        return self.chat_model

    def _mock_model_name(self, response_kind: str) -> str:
        return "local-mock-reasoner" if response_kind in {"router", "feedback", "planning"} else "local-mock-chat"

    def _max_tokens_for_kind(self, response_kind: str) -> int:
        if response_kind in {"router", "planning", "feedback"}:
            return int(min(self.max_tokens, 600))
        if response_kind == "codegen":
            return int(min(self.max_tokens, 800))
        return int(self.max_tokens)

    def _temperature_for_kind(self, response_kind: str) -> float:
        if response_kind in {"router", "planning", "feedback"}:
            return float(min(self.temperature, 0.2))
        return float(self.temperature)

    def preflight_chat_model(self) -> None:
        preflight_messages = [
            {"role": "system", "content": "You are a strict JSON assistant."},
            {"role": "user", "content": 'Return exactly this JSON and nothing else: {"ok": true, "kind": "chat"}'},
        ]
        obj = self.chat_json(preflight_messages, response_kind="chat")
        if obj.get("ok") is not True or obj.get("kind") != "chat":
            raise RuntimeError(f"LLM preflight check failed for chat model: {obj}")
        self.preflight_chat_ok = True

    def preflight_reasoner_model(self) -> None:
        preflight_messages = [
            {"role": "system", "content": "You are a strict JSON assistant."},
            {"role": "user", "content": 'Return exactly this JSON and nothing else: {"ok": true, "kind": "reasoner"}'},
        ]
        obj = self.chat_json(preflight_messages, response_kind="planning")
        if obj.get("ok") is not True or obj.get("kind") != "reasoner":
            raise RuntimeError(f"LLM preflight check failed for reasoner model: {obj}")
        self.preflight_reasoner_ok = True

    def preflight_check(self) -> None:
        if self.using_mock:
            self.preflight_chat_ok = True
            self.preflight_reasoner_ok = True
            return
        try:
            self.preflight_chat_model()
            self.preflight_reasoner_model()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM preflight check failed. {exc}") from exc

    # test-only helper
    def _mock_response(self, response_kind: str, sample_idx: int) -> str:
        if response_kind == "router":
            choices = ["critical_load_priority", "restoration_capability_priority", "global_efficiency_priority"]
            task = choices[sample_idx % len(choices)]
            return json.dumps(
                {
                    "task_mode": task,
                    "confidence": 0.73,
                    "reason": "Mock router selected a valid simplified task.",
                    "stage": "middle",
                }
            )
        if response_kind == "planning":
            return json.dumps(
                {
                    "weakest_layer": "0",
                    "weakest_zone": "A",
                    "late_stage_risk": "constraint_violation",
                    "violation_risk": "moderate",
                    "should_reward": ["critical_load_progress", "balanced_recovery", "task_aligned_actions"],
                    "should_penalize": ["invalid_action", "constraint_violation", "late_unfocused_actions"],
                    "should_avoid": ["blind_mes_overuse", "coordinated_overuse_without_prereqs"],
                    "finishing_strategy": "late-stage targeted power/comm actions on weakest layer-zone pair",
                    "codegen_guidance": "Keep revise_state compact and keep intrinsic reward bounded.",
                }
            )
        if response_kind == "feedback":
            return json.dumps(
                {
                    "improvement_focus": ["raise critical-load gains near termination", "lower violation rate"],
                    "keep_signals": ["progress_delta", "weakest-layer targeting"],
                    "avoid_patterns": ["late coordinated spam", "invalid mes dispatch"],
                }
            )
        code = """from __future__ import annotations
import numpy as np

def revise_state(state, info=None):
    x = np.asarray(state, dtype=np.float32)
    info = info or {}
    stage = 0.0 if str(info.get('stage', 'middle')) == 'early' else (1.0 if str(info.get('stage', 'middle')) == 'late' else 0.5)
    weak_layer = float(info.get('weakest_layer', 0))
    return np.concatenate([x, np.array([stage, weak_layer], dtype=np.float32)], axis=0)

def intrinsic_reward(state, action, next_state, info=None, revised_state=None):
    s = np.asarray(state, dtype=np.float32); ns = np.asarray(next_state, dtype=np.float32); info = info or {}
    d_load = float(np.mean(ns[9:12] - s[9:12])); d_power = float(np.mean(ns[0:3] - s[0:3])); d_comm = float(np.mean(ns[3:6] - s[3:6])); d_road = float(np.mean(ns[6:9] - s[6:9]))
    invalid = 1.0 if bool(info.get('invalid_action', False)) else 0.0; violate = 1.0 if bool(info.get('constraint_violation', False)) else 0.0
    late = 1.0 if str(info.get('stage', 'middle')) == 'late' else 0.0; targeted = 1.0 if action in {3,4,5,6,7,8} else 0.0
    return float(0.40*d_load + 0.22*d_power + 0.18*d_comm + 0.12*d_road + 0.05*late*targeted - 0.12*invalid - 0.15*violate)
"""
        return json.dumps(
            {
                "file_name": f"mock_candidate_{sample_idx}.py",
                "rationale": "Mock codegen candidate with bounded shaping terms for simplified coupled recovery.",
                "code": code,
                "expected_behavior": "Improves critical-load and balanced tri-layer progress while reducing violations.",
            }
        )

    def _real_chat(self, messages: list[dict[str, str]], response_kind: str = "chat") -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self._normalize_base_url(), timeout=self.timeout_seconds, max_retries=0)
        preferred_model = self._select_model(response_kind=response_kind)
        model_candidates = [preferred_model]
        if self.chat_model and self.chat_model != preferred_model:
            model_candidates.append(self.chat_model)
        if self.reasoner_model and self.reasoner_model not in model_candidates:
            model_candidates.append(self.reasoner_model)

        last_exc: Exception | None = None
        for model in model_candidates:
            for attempt in range(self.max_retries + 1):
                t0 = time.time()
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self._temperature_for_kind(response_kind),
                        max_tokens=self._max_tokens_for_kind(response_kind),
                    )
                    content = resp.choices[0].message.content or ""
                    if not content.strip():
                        raise RuntimeError(f"Empty content from LLM for response_kind={response_kind}, model={model}")
                    self._record_call(
                        response_kind=response_kind,
                        model=model,
                        success=True,
                        latency_sec=time.time() - t0,
                        error="",
                    )
                    return content
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    err_msg = str(exc)
                    self._record_call(
                        response_kind=response_kind,
                        model=model,
                        success=False,
                        latency_sec=time.time() - t0,
                        error=err_msg,
                    )
                    if attempt < self.max_retries:
                        time.sleep(1.2 * (attempt + 1))
        raise RuntimeError(f"DeepSeek API call failed after retries ({response_kind}, model={preferred_model}): {last_exc}")

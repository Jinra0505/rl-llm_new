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

        self._validate_formal_configuration()

    def _validate_formal_configuration(self) -> None:
        if self.mode != "real":
            raise RuntimeError(f"Formal run requires llm_mode=real, got: {self.mode}")
        if not self.api_key:
            raise RuntimeError("Formal run requires real LLM, but DEEPSEEK_API_KEY is not set.")
        if not self.base_url:
            raise RuntimeError("Formal run requires non-empty DEEPSEEK_BASE_URL.")
        if not self.chat_model:
            raise RuntimeError("Formal run requires non-empty DEEPSEEK_MODEL_CHAT (or DEEPSEEK_MODEL).")
        if not self.reasoner_model:
            raise RuntimeError("Formal run requires non-empty DEEPSEEK_MODEL_REASONER.")
        try:
            from openai import OpenAI  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Formal run requires openai dependency: install requirements.txt (or `pip install openai>=1.30.0`)."
            ) from exc

    @property
    def using_mock(self) -> bool:
        # Kept for backward compatibility with existing code paths; formal mode is always real.
        return False

    def effective_mode(self) -> str:
        return "real"

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
        _ = sample_idx  # test-only placeholder; formal path never uses mock sampling.
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
        try:
            self.preflight_chat_model()
            self.preflight_reasoner_model()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM preflight check failed. {exc}") from exc

    # test-only helper
    def _mock_response(self, response_kind: str, sample_idx: int) -> str:
        _ = response_kind
        _ = sample_idx
        return "{}"

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

"""Google GenAI wrapper with request pacing and token usage logging."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import random

from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USAGE_LOG_PATH = PROJECT_ROOT / "logs" / "usage.jsonl"
INTERACTIONS_LOG_PATH = PROJECT_ROOT / "logs" / "interactions.jsonl"
DEFAULT_MODEL = "gemini-3.5-flash"
_call_lock = threading.Lock()
_last_call_started: float | None = None
_usage_log_lock = threading.Lock()


def _minimum_interval() -> float:
    try:
        interval = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "0"))
    except ValueError as error:
        raise ValueError("LLM_MIN_INTERVAL_SECONDS must be a number") from error
    if interval < 0:
        raise ValueError("LLM_MIN_INTERVAL_SECONDS cannot be negative")
    return interval


def _wait_for_call_slot() -> None:
    global _last_call_started
    with _call_lock:
        interval = _minimum_interval()
        if _last_call_started is not None:
            delay = interval - (time.monotonic() - _last_call_started)
            if delay > 0:
                time.sleep(delay)
        _last_call_started = time.monotonic()


def _token_count(metadata: Any, *names: str) -> int | None:
    for name in names:
        value = metadata.get(name) if isinstance(metadata, dict) else getattr(metadata, name, None)
        if value is not None:
            return int(value)
    return None


def _usage_record(model: str, interaction_type: str, response: Any = None) -> dict[str, object]:
    metadata = getattr(response, "usage_metadata", None) if response is not None else None
    record: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "interaction_type": interaction_type,
        "input_tokens": _token_count(metadata, "prompt_token_count") or 0,
        "output_tokens": _token_count(metadata, "candidates_token_count") or 0,
    }
    thinking_tokens = _token_count(metadata, "thoughts_token_count", "thinking_token_count")
    if thinking_tokens is not None:
        record["thinking_tokens"] = thinking_tokens
    return record


def _append_usage(record: dict[str, object]) -> None:
    USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _usage_log_lock:
        with USAGE_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def finalize_usage_records(
    records: list[dict[str, object]], interaction_type: str
) -> None:
    """Replace provisional per-call types once the router classifies the question."""
    timestamps = {record.get("timestamp") for record in records}
    if not timestamps or not USAGE_LOG_PATH.exists():
        return
    with _usage_log_lock:
        lines = USAGE_LOG_PATH.read_text(encoding="utf-8").splitlines()
        updated_lines = []
        for line in lines:
            record = json.loads(line)
            if record.get("timestamp") in timestamps:
                record["interaction_type"] = interaction_type
                line = json.dumps(record, ensure_ascii=False)
            updated_lines.append(line)
        USAGE_LOG_PATH.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

RETRYABLE_STATUS_CODES = {429, 500, 503}
MAX_ATTEMPTS = 5


def _status_code(error: Exception) -> int | None:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    return code or getattr(response, "status_code", None)


def _is_rate_limit(error: Exception) -> bool:
    return _status_code(error) == 429


def _is_retryable(error: Exception) -> bool:
    return _status_code(error) in RETRYABLE_STATUS_CODES


def _build_config(model: str, tools: Any, system_instruction: str | None) -> dict[str, object]:
    config: dict[str, object] = {}
    if tools is not None:
        config["tools"] = tools
    if system_instruction is not None:
        config["system_instruction"] = system_instruction
    if not model.lower().startswith("gemma"):
        thinking_level = os.getenv("LLM_THINKING_LEVEL", "low").upper()
        config["thinking_config"] = {"thinking_level": thinking_level}
    return config


def _call_with_retries(client, model, messages, config, interaction_type):
    for attempt in range(MAX_ATTEMPTS):
        _wait_for_call_slot()
        try:
            response = client.models.generate_content(
                model=model,
                contents=messages,
                config=config or None,
            )
        except Exception as error:
            _append_usage(_usage_record(model, interaction_type))
            if not _is_retryable(error) or attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** (attempt + 1) + random.uniform(0, 1))
            continue
        usage = _usage_record(model, interaction_type, response)
        _append_usage(usage)
        return response, usage
    raise RuntimeError("Generation failed after retries")


def generate(
    messages: Any,
    tools: Any = None,
    system_instruction: str | None = None,
    interaction_type: str = "unknown",
) -> tuple[Any, dict[str, object]]:
    """Generate a response and return it with the recorded token usage."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    fallback = os.getenv("LLM_FALLBACK_MODEL")
    client = genai.Client(api_key=api_key)
    try:
        return _call_with_retries(
            client, model, messages,
            _build_config(model, tools, system_instruction), interaction_type,
        )
    except Exception as error:
        if fallback and fallback != model and _is_retryable(error):
            return _call_with_retries(
                client, fallback, messages,
                _build_config(fallback, tools, system_instruction), interaction_type,
            )
        raise



def summarize_usage(
    path: str | Path = INTERACTIONS_LOG_PATH,
) -> dict[str, dict[str, float | int]]:
    """Return question counts and average token usage grouped by interaction type."""
    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "thinking_calls": 0}
    )
    with Path(path).open(encoding="utf-8") as log_file:
        for line in log_file:
            if not line.strip():
                continue
            record = json.loads(line)
            interaction_type = record.get("type", record.get("interaction_type", "unknown"))
            summary = totals[str(interaction_type)]
            summary["calls"] += 1
            summary["input_tokens"] += int(record.get("input_tokens", 0))
            summary["output_tokens"] += int(record.get("output_tokens", 0))
            if "thinking_tokens" in record:
                summary["thinking_tokens"] += int(record["thinking_tokens"])
                summary["thinking_calls"] += 1

    return {
        interaction_type: {
            "calls": int(summary["calls"]),
            "average_input_tokens": summary["input_tokens"] / summary["calls"],
            "average_output_tokens": summary["output_tokens"] / summary["calls"],
            "average_thinking_tokens": (
                summary["thinking_tokens"] / summary["thinking_calls"]
                if summary["thinking_calls"]
                else 0.0
            ),
        }
        for interaction_type, summary in totals.items()
    }
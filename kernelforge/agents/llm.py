"""LLM helpers shared by agent implementations."""

from __future__ import annotations

import json
import os
import re
from typing import Any


def call_anthropic(
    system: str,
    user: str,
    model: str,
    max_tokens: int = 16384,
    temperature: float = 0.2,
    timeout_sec: int = 180,
) -> str:
    """Call Anthropic Messages API and return the concatenated text response."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Run:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    import requests

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=timeout_sec,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Anthropic API error {response.status_code}: {response.text[:500]}"
        )

    payload = response.json()
    parts: list[str] = []
    for block in payload.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def parse_json_response(raw: str) -> dict[str, Any]:
    """Extract JSON from plain text or markdown-fenced LLM output."""
    fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:500]}")

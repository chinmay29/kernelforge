"""Focused kernel generator agent."""

from __future__ import annotations

from typing import Any

from kernelforge.agents.base import Agent, AgentResult
from kernelforge.agents.llm import call_anthropic, parse_json_response


_SYSTEM_PROMPT = """You are a GPU kernel engineer writing Triton kernels for NVIDIA B200.
Generate valid Python source for `kernel.py`.

Hard rules:
1. Keep `def kernel(...)` as the callable entrypoint.
2. Use imports: torch, triton, triton.language as tl.
3. FP8 dequantization must happen before tl.dot.
4. Avoid type promotion bugs: fp16*fp32 and bf16*fp32 produce fp32; cast result back.
5. Do not use tl.atomic_add for fp32 accumulation.
6. Any `tl.load` / `tl.store` using `offs_n`/`offs_bn` must guard N-tail (`offs_n < N`) or use `% N`.
7. CUDA async errors often surface late; treat tail-mask correctness as mandatory.
8. In sorted-token kernels, use `offs_token` (token ids) only for original token tensors; use `offs_token_id` for intermediate tensors sized by `num_tokens`.

Return JSON only with keys: reasoning, mutation_name, source.
"""


class GeneratorAgent(Agent):
    """LLM-backed source generator with compact prompts and retry context."""

    name = "generator"

    def __init__(self, model: str = "claude-sonnet-4-5-20250929") -> None:
        self.model = model

    def run(self, inputs: dict[str, Any]) -> AgentResult:
        source = str(inputs.get("source", ""))
        patterns = inputs.get("patterns", []) or []
        attempt = int(inputs.get("attempt", 0))
        task_instruction = str(inputs.get("task_instruction", "Optimize one high-impact kernel bottleneck."))
        error = str(inputs.get("error", ""))
        feedback = inputs.get("feedback", {}) or {}
        previous_generated = str(inputs.get("previous_generated", ""))

        user_prompt = self._build_user_prompt(
            source=source,
            patterns=patterns,
            attempt=attempt,
            task_instruction=task_instruction,
            error=error,
            feedback=feedback,
            previous_generated=previous_generated,
        )

        try:
            raw = call_anthropic(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                model=self.model,
                temperature=0.2 if attempt == 0 else 0.05,
            )
        except Exception as exc:
            return AgentResult(
                ok=False,
                error=f"Generator API call failed: {exc}",
                data={
                    "system_prompt": _SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                },
            )

        try:
            parsed = parse_json_response(raw)
        except Exception as exc:
            return AgentResult(
                ok=False,
                error=f"Generator response parse failed: {exc}",
                data={
                    "system_prompt": _SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                    "raw_response": raw,
                },
            )

        generated_source = str(parsed.get("source", ""))
        if not generated_source.strip():
            return AgentResult(
                ok=False,
                error="Generator returned empty source field.",
                data={
                    "system_prompt": _SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                    "raw_response": raw,
                    "parsed": parsed,
                },
            )

        return AgentResult(
            ok=True,
            data={
                "reasoning": str(parsed.get("reasoning", "")),
                "mutation_name": str(parsed.get("mutation_name", f"multi_agent_attempt_{attempt}")),
                "source": generated_source,
                "system_prompt": _SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "raw_response": raw,
                "parsed": parsed,
            },
        )

    @staticmethod
    def _build_user_prompt(
        source: str,
        patterns: list[dict[str, Any]],
        attempt: int,
        task_instruction: str,
        error: str,
        feedback: dict[str, Any],
        previous_generated: str,
    ) -> str:
        patterns_text = GeneratorAgent._format_patterns(patterns[:3])
        runtime_guards = GeneratorAgent._runtime_safety_guards()

        if attempt == 0:
            return (
                "Generate an optimized Triton kernel mutation.\n\n"
                f"## Current Source ({len(source.splitlines())} lines)\n"
                "```python\n"
                f"{source}\n"
                "```\n\n"
                "## Top RAG Patterns\n"
                f"{patterns_text}\n\n"
                "## Runtime Safety Guards\n"
                f"{runtime_guards}\n\n"
                "## Task\n"
                f"{task_instruction}\n\n"
                "Return JSON only."
            )

        return (
            f"RETRY ATTEMPT {attempt}/2\n\n"
            "Your previous output failed validation/constraints.\n\n"
            "## Error\n"
            f"{error or 'Unknown error'}\n\n"
            "## Targeted Feedback\n"
            f"Analysis: {feedback.get('analysis', '')}\n"
            f"Fix instructions: {feedback.get('fix_instructions', '')}\n"
            f"Code example:\n{feedback.get('code_example', '')}\n\n"
            "## Previous Generated Source (failed)\n"
            "```python\n"
            f"{previous_generated[:8000]}\n"
            "```\n\n"
            "## Current Baseline Source\n"
            "```python\n"
            f"{source}\n"
            "```\n\n"
            "## Top RAG Patterns\n"
            f"{patterns_text}\n\n"
            "## Runtime Safety Guards\n"
            f"{runtime_guards}\n\n"
            "Focus only on fixing the failure first, then keep the optimization direction.\n"
            "Return JSON only."
        )

    @staticmethod
    def _format_patterns(patterns: list[dict[str, Any]]) -> str:
        if not patterns:
            return "- No patterns available."

        rows: list[str] = []
        for idx, pattern in enumerate(patterns, start=1):
            rows.append(f"{idx}. {pattern.get('id', 'unknown')} ({pattern.get('category', 'pattern')})")
            desc = str(pattern.get("description", "")).strip()
            hint = str(pattern.get("code_hint", "")).strip()
            constraint = str(pattern.get("constraint", "")).strip()
            if desc:
                rows.append(f"   - Description: {desc[:220]}")
            if hint:
                rows.append(f"   - Code hint: `{hint[:180]}`")
            if constraint:
                rows.append(f"   - Constraint: `{constraint[:160]}`")

        return "\n".join(rows)

    @staticmethod
    def _runtime_safety_guards() -> str:
        return (
            "- Guard all N-tail loads/stores: if pointer math uses `offs_n` or `offs_bn`, include `offs_n < N` in mask.\n"
            "- For FP8 block scales, dequantize before dot and keep dot operands non-fp32.\n"
            "- Prefer `% N` only when wraparound semantics are intended; otherwise explicit N mask is safer.\n"
            "- For sorted token dispatch: token-id indexing for hidden_states/routing; pair-index (`offs_token_id`) indexing for num_tokens intermediates."
        )

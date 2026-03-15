"""Focused kernel generator agent."""

from __future__ import annotations

from typing import Any

from kernelforge.agents.base import Agent, AgentResult
from kernelforge.agents.llm import call_anthropic, parse_json_response
from kernelforge.triton_templates import (
    GEMM1_ONLY_NOTES,
    GEMM1_ONLY_TEMPLATE,
    GEMM1_TEMPLATE_KERNEL_NAME,
)


_SYSTEM_PROMPT = """You are a GPU kernel engineer writing high-performance Triton kernels for NVIDIA B200.
Generate valid Python source for `kernel.py`.

## Architecture target
The kernel implements a DeepSeek-V3/R1 MoE layer: routing → GEMM1 (gate+up proj) → SwiGLU → GEMM2.
The goal is 2-4× speedup over the baseline through two structural changes:
(A) Sort tokens by expert ID and run a single grouped GEMM instead of a per-expert loop.
(B) Use native FP8 tl.dot (float8e4nv operands) with block-scale correction in the accumulator
    instead of pre-dequantizing all expert weights upfront.

## Hard rules
1. Keep `def kernel(...)` as the callable entrypoint and preserve the exact baseline argument order.
2. Use imports: torch, triton, triton.language as tl.
3. **FP8 native GEMM**: On B200, `tl.dot` accepts `tl.float8e4nv` operands directly (maps to
   tcgen05.mma). Do NOT pre-dequantize all expert weights — load FP8, call tl.dot, apply block
   scales to the FP32 accumulator every BLOCK_K=128 step.
4. Avoid type promotion bugs: fp16*fp32 and bf16*fp32 produce fp32; cast result back.
5. Do not use tl.atomic_add for fp32 accumulation; write to a flat output buffer indexed by
   sorted-pair position, then let PyTorch scatter_add_ / index_add_ accumulate back.
6. Any `tl.load` / `tl.store` using `offs_n`/`offs_bn` must guard N-tail (`offs_n < N`) or use `% N`.
7. CUDA async errors surface late; treat tail-mask correctness as mandatory.
8. In sorted-token kernels: `tok_id` (loaded from sorted_ids) indexes original-token tensors
   (hidden_states, routing_weights); `pos` (0..num_pairs-1) indexes intermediate flat buffers.
9. Preserve the current return style.
10. Tile sizes: prefer BLOCK_M=128, BLOCK_N=128 or 256, BLOCK_K=64. With FP8 (1 byte/elem) and
    3 stages, 128×256×64 costs 3*(128*64 + 256*64)*1 = 73KB — well within B200's 228KB.
11. Persistent grouped GEMM: use `grid=(NUM_SMS,)` where NUM_SMS=148 for B200. Each CTA loops
    over tiles across all experts via a tile-to-expert mapping (cumulative tile offsets per expert).
12. Block-scale application: apply `acc *= scale_a[:, None] * scale_b[None, :]` inside the K loop
    every BLOCK_K=128 steps (DeepSeek-V3 block size is 128).
13. If you attempt the grouped FP8 architecture, use the template helper names exactly:
    `NUM_SMS = 148`, `_sort_tokens`, `_grouped_gemm1_swiglu_kernel`, `_grouped_gemm2_kernel`,
    `sorted_ids`, `expert_starts`, `counts`, `tile_starts`.
14. Do NOT emit the legacy sorted-token architecture names or patterns:
    `gemm1_kernel`, `gemm2_kernel`, `expert_ids_per_block`, `token_expert_pairs`.

## Output mode
- Preferred for all attempts: return a patch against the provided baseline source.
- JSON keys: reasoning, mutation_name, patch.
- Fallback allowed only when patching is impossible: return full file `source`.
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
        patch = str(parsed.get("patch", ""))
        if attempt == 0 and not patch.strip() and not generated_source.strip():
            return AgentResult(
                ok=False,
                error="Generator returned neither patch nor source.",
                data={
                    "system_prompt": _SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                    "raw_response": raw,
                    "parsed": parsed,
                },
            )
        if attempt > 0 and not patch.strip() and not generated_source.strip():
            return AgentResult(
                ok=False,
                error="Generator returned neither patch nor source on retry.",
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
                "patch": patch,
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
        seed_template_context = GeneratorAgent._seed_template_context(source)

        if attempt == 0:
            error_block = ""
            if error.strip():
                error_block = (
                    "## Last Failure (must fix first)\n"
                    f"{error.strip()[:1200]}\n\n"
                    "Do not do tuning-only edits (tile/warps/stages) until this failure is resolved.\n\n"
                )
            return (
                "Generate an optimized Triton kernel mutation via PATCH-ONLY editing.\n"
                "Start from the baseline source and make minimal local edits.\n\n"
                f"## Current Source ({len(source.splitlines())} lines)\n"
                "```python\n"
                f"{source}\n"
                "```\n\n"
                f"{error_block}"
                "## Allowed Edit Scope (seed-first)\n"
                "- preserve the exact `kernel(...)` signature, parameter order, and current return style\n"
                "- preserve geometry constants (`H`, `I`, `E_GLOBAL`, `E_LOCAL`, `BLOCK`, `TOP_K`) unless the task explicitly requires a change\n"
                "- tile sizes / num_warps / num_stages\n"
                "- vector widths and load/store policies\n"
                "- shared-memory layout and predicates\n"
                "- epilogue fusion logic\n"
                "- Do not rewrite unrelated routing/control-flow blocks.\n\n"
                "## Top RAG Patterns\n"
                f"{patterns_text}\n\n"
                "## Runtime Safety Guards\n"
                f"{runtime_guards}\n\n"
                f"{seed_template_context}"
                "## Task\n"
                f"{task_instruction}\n\n"
                "Patch format (can include multiple blocks):\n"
                "<<<<<<< SEARCH\n"
                "<exact old text>\n"
                "=======\n"
                "<new text>\n"
                ">>>>>>> REPLACE\n\n"
                "Return JSON only with keys: reasoning, mutation_name, patch."
            )

        return (
            f"RETRY ATTEMPT {attempt}/2\n\n"
            "Your previous output failed validation/constraints.\n"
            "Return a patch against the current baseline source. Do NOT return a full-file rewrite.\n\n"
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
            f"{seed_template_context}"
            "Focus only on fixing the failure first, then keep the optimization direction.\n"
            "Patch format (can include multiple blocks):\n"
            "<<<<<<< SEARCH\n"
            "<exact old text>\n"
            "=======\n"
            "<new text>\n"
            ">>>>>>> REPLACE\n\n"
            "Return JSON only with keys: reasoning, mutation_name, patch."
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
            "- FP8 native GEMM: cast tensors to tl.float8e4nv before tl.dot; accumulate in fp32; apply block scales every 128 K-steps.\n"
            "- Block scales layout for hidden_states: scale[k_blk, tok_id] where k_blk = k_start // 128. Load with tl.load(scale_ptr + k_blk * T + tok_ids).\n"
            "- Block scales layout for weights W13: scale[e, row_blk, col_blk]. Load per K-tile.\n"
            "- Prefer `% N` only when wraparound semantics are intended; otherwise explicit N mask is safer.\n"
            "- Sorted token dispatch: `tok_id = tl.load(sorted_ids_ptr + pos)` gives the original token index;\n"
            "  use tok_id to index hidden_states/routing_weights; use pos (flat sort position) to index intermediate flat buffers.\n"
            "- Do not use tl.atomic_add; write GEMM2 output to a flat buffer [N_pairs, H], then PyTorch index_add_ accumulates."
        )

    @staticmethod
    def _seed_template_context(source: str) -> str:
        # Seed-stage (pure PyTorch, no @triton.jit yet): show the GEMM1-only template.
        if "def _gemm1_subpath(" in source and "@triton.jit" not in source:
            return (
                "## Vetted Triton Micro-Kernel Template (GEMM1 only — seed stage)\n"
                "Integration contract:\n"
                f"{GEMM1_ONLY_NOTES}\n\n"
                "```python\n"
                f"{GEMM1_ONLY_TEMPLATE}\n"
                "```\n\n"
                f"The only new Triton kernel should be `{GEMM1_TEMPLATE_KERNEL_NAME}`.\n\n"
            )
        # Post-seed: baseline already has Triton JIT kernels. Show the full optimization target.
        from kernelforge.triton_templates import GROUPED_GEMM_FP8_NOTES, GROUPED_GEMM_FP8_TEMPLATE
        return (
            "## Optimization Target: Grouped FP8 GEMM Architecture\n"
            "The baseline still has a per-expert Python loop and pre-dequantizes all weights.\n"
            "Replace both with the sort-then-grouped-GEMM pattern shown below.\n"
            "Use the helper/function names exactly as written in the template and keep the "
            "persistent launch shape `grid=(NUM_SMS,)`.\n"
            "Required names for grouped attempts: `NUM_SMS = 148`, `_sort_tokens`, "
            "`_grouped_gemm1_swiglu_kernel`, `_grouped_gemm2_kernel`, `sorted_ids`, "
            "`expert_starts`, `counts`, `tile_starts`.\n"
            "Do not fall back to legacy `gemm1_kernel`/`gemm2_kernel` or "
            "`expert_ids_per_block` dispatch.\n\n"
            f"{GROUPED_GEMM_FP8_NOTES}\n\n"
            "```python\n"
            f"{GROUPED_GEMM_FP8_TEMPLATE}\n"
            "```\n\n"
        )

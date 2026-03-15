"""Targeted feedback agent for retry instructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kernelforge.agents.base import Agent, AgentResult


@dataclass(frozen=True)
class Feedback:
    analysis: str
    fix_instructions: str
    code_example: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "fix_instructions": self.fix_instructions,
            "code_example": self.code_example,
        }


class FeedbackAgent(Agent):
    """Translate failures into explicit, line-level retry instructions."""

    name = "feedback"

    def run(self, inputs: dict[str, Any]) -> AgentResult:
        stage = str(inputs.get("stage", "validation"))
        error = str(inputs.get("error", ""))
        attempt = int(inputs.get("attempt", 0))
        report = inputs.get("report", {}) or {}
        feedback = self.generate(stage=stage, error=error, attempt=attempt, report=report)
        return AgentResult(ok=True, data=feedback.to_dict())

    def generate(
        self,
        stage: str,
        error: str,
        attempt: int,
        report: dict[str, Any] | None = None,
    ) -> Feedback:
        report = report or {}

        explicitness = "high" if attempt >= 1 else "medium"

        if "type promotion" in error.lower() or "fp16" in error.lower() and "fp32" in error.lower():
            return Feedback(
                analysis=(
                    "The failure indicates mixed precision promotion during FP8 dequantization. "
                    "A cast to fp16/bf16 happened before scaling, but the multiply promoted the result to fp32."
                ),
                fix_instructions=(
                    "Replace the dequantization assignment with an explicit post-multiply cast, or cast the scale first. "
                    "Do not feed promoted fp32 tensors into tl.dot operands. "
                    f"Retry mode: {explicitness}."
                ),
                code_example=(
                    "a_fp16 = (a.to(tl.float16) * a_scale[:, None]).to(tl.float16)\n"
                    "# or\n"
                    "a_scale_fp16 = a_scale.to(tl.float16)\n"
                    "a_fp16 = a.to(tl.float16) * a_scale_fp16[:, None]"
                ),
            )

        if "shared memory" in error.lower() or "smem" in error.lower():
            return Feedback(
                analysis="The generated tile/stage configuration exceeds per-block shared memory budget on B200.",
                fix_instructions=(
                    "Reduce BLOCK_M/BLOCK_N/BLOCK_K or NUM_STAGES so estimated SMEM stays <= 227KB. "
                    "Prioritize lowering BLOCK_K or NUM_STAGES first to preserve occupancy."
                ),
                code_example="BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 64\nNUM_STAGES = 3",
            )

        if "atomic_add" in error.lower():
            return Feedback(
                analysis="The kernel uses tl.atomic_add path that is unsafe for this fp32 accumulation workflow.",
                fix_instructions=(
                    "Remove tl.atomic_add from Triton path and write per-token/per-expert outputs, then combine with "
                    "PyTorch index_add_ outside the kernel."
                ),
                code_example="output.index_add_(0, token_ids, partial_values)",
            )

        if (
            "operation not supported on global/shared address space" in error.lower()
            or "acceleratorerror" in error.lower()
        ):
            return Feedback(
                analysis=(
                    "This CUDA error is often asynchronous and commonly indicates an earlier illegal memory access "
                    "in kernel loads/stores."
                ),
                fix_instructions=(
                    "Audit every tl.load/tl.store using offs_n/offs_bn tail indices. Add `(offs_n < N)` to masks "
                    "or `% N` in pointer arithmetic for safe wraparound. Fix this before any further optimization."
                ),
                code_example=(
                    "w = tl.load(\n"
                    "    w_ptrs,\n"
                    "    mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N),\n"
                    "    other=0.0,\n"
                    ")"
                ),
            )

        if "seed_stage_numerics" in error.lower():
            return Feedback(
                analysis=(
                    "The seed-stage GEMM1 helper already receives dequantized float32 "
                    "activations and weights, so the previous attempt lost accuracy by "
                    "downcasting the GEMM operands too early."
                ),
                fix_instructions=(
                    "Keep `_gemm1_tile_kernel` operands in fp32 for the first correctness "
                    "pass. Do not cast `a` or `w` to bf16/fp16 before the dot. Use "
                    "`tl.dot(a, w, input_precision=\"ieee\")` or equivalent fp32 accumulation."
                ),
                code_example=(
                    "a = tl.load(a_ptrs, mask=..., other=0.0)\n"
                    "w = tl.load(w_ptrs, mask=..., other=0.0)\n"
                    "acc += tl.dot(a, w, input_precision=\"ieee\")"
                ),
            )

        if "grouped_gemm_template_alignment" in error.lower():
            return Feedback(
                analysis=(
                    "The grouped FP8 attempt drifted away from the vetted architecture and "
                    "reintroduced the older sorted-token Triton path."
                ),
                fix_instructions=(
                    "Stay close to the grouped template. Use exactly these helper names: "
                    "`NUM_SMS = 148`, `_sort_tokens`, `_grouped_gemm1_swiglu_kernel`, "
                    "`_grouped_gemm2_kernel`, `sorted_ids`, `expert_starts`, `counts`, "
                    "`tile_starts`. Do not emit `gemm1_kernel`, `gemm2_kernel`, "
                    "`expert_ids_per_block`, or Python `token_expert_pairs` lists."
                ),
                code_example=(
                    "sorted_ids, expert_starts, counts, tile_starts = _sort_tokens(\n"
                    "    topk_idx, local_start, E_LOCAL, device\n"
                    ")\n"
                    "_grouped_gemm1_swiglu_kernel[(NUM_SMS,)](...)\n"
                    "_grouped_gemm2_kernel[(NUM_SMS,)](...)\n"
                    "accum.index_add_(0, sorted_ids.long(), flat_out * rw_sorted[:, None])"
                ),
            )

        if "incorrect_numerical" in error.lower() or "max_abs_error" in error.lower():
            return Feedback(
                analysis=(
                    "The kernel executes but produces wrong values. This is commonly caused by semantic indexing "
                    "mismatch in sorted dispatch (token-id vs pair-index), or incorrect scale application location."
                ),
                fix_instructions=(
                    "Audit sorted-token indexing: use `offs_token` only for original hidden/routing tensors, and "
                    "use `offs_token_id` for intermediate/output tensors shaped by `num_tokens`. Keep FP8 "
                    "dequantization before tl.dot."
                ),
                code_example=(
                    "offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)\n"
                    "offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)\n"
                    "a_hidden_ptrs = hidden_ptr + offs_token[:, None] * stride_am\n"
                    "c_ptrs = c_ptr + offs_token_id[:, None] * stride_cm"
                ),
            )

        if "seed_stage_scope" in error.lower() or "_gemm1_tile_kernel" in error:
            return Feedback(
                analysis=(
                    "The seed-stage Tritonization scope was violated. The first Triton step "
                    "must only replace the GEMM1 helper with the vetted micro-kernel template."
                ),
                fix_instructions=(
                    "Use exactly one `@triton.jit` kernel named `_gemm1_tile_kernel`, "
                    "launch it from `_gemm1_subpath(...)`, and keep routing, SwiGLU, GEMM2, "
                    "and `accum.index_add_` unchanged in PyTorch."
                ),
                code_example=(
                    "@triton.jit\n"
                    "def _gemm1_tile_kernel(...):\n"
                    "    ...\n\n"
                    "def _gemm1_subpath(A_e, W13_e):\n"
                    "    out = torch.empty(...)\n"
                    "    _gemm1_tile_kernel[grid](...)\n"
                    "    return out"
                ),
            )

        if stage == "generator":
            return Feedback(
                analysis="The generator response format or parse failed, so no valid source was produced.",
                fix_instructions=(
                    "Return strict JSON with keys reasoning, mutation_name, patch. "
                    "Use SEARCH/REPLACE blocks in `patch`, do not wrap JSON in prose, "
                    "and fall back to full-file `source` only if patching is impossible."
                ),
                code_example=(
                    '{"reasoning": "...", "mutation_name": "retry_fix", '
                    '"patch": "<<<<<<< SEARCH\\nold\\n=======\\nnew\\n>>>>>>> REPLACE"}'
                ),
            )

        first_failure = report.get("first_failure") or {}
        where = first_failure.get("name", "validation")
        msg = first_failure.get("message", error)

        return Feedback(
            analysis=f"The previous attempt failed in {stage} at check `{where}`.",
            fix_instructions=(
                f"Address this error directly before further optimization: {msg}. "
                "Keep other logic unchanged and make only minimal targeted edits."
            ),
            code_example="# Apply minimal local patch around failing line only",
        )

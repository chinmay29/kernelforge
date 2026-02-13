"""Focused retrieval agent for mutation guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kernelforge.agents.base import Agent, AgentResult
from kernelforge.rag import RagStore


@dataclass(frozen=True)
class RetrievedPattern:
    id: str
    category: str
    relevance_score: float
    description: str
    code_hint: str
    constraint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "relevance_score": self.relevance_score,
            "description": self.description,
            "code_hint": self.code_hint,
            "constraint": self.constraint,
        }


class RagAgent(Agent):
    """Retrieve high-signal RAG entries for the current attempt."""

    name = "rag"

    def __init__(self, rag_store: RagStore, max_patterns: int = 5) -> None:
        self.rag_store = rag_store
        self.max_patterns = max_patterns

    def run(self, inputs: dict[str, Any]) -> AgentResult:
        source = str(inputs.get("source", ""))
        last_error = str(inputs.get("last_error", "")).strip()
        mutation_goal = str(inputs.get("mutation_goal", "")).strip()
        retrieved_hint = inputs.get("retrieved_hint", []) or []

        query = " ".join(
            part for part in [mutation_goal, last_error, self._extract_signal(source)] if part
        )
        query = query or "fused moe fp8 triton dequantization"

        try:
            candidates: list[dict[str, Any]] = []
            candidates.extend(self._hard_guardrails(last_error))

            # Keep seed retrieval useful: drop low-signal run-history rows first.
            hint_rows = [
                row for row in retrieved_hint
                if str(row.get("category", "")).lower() != "runs"
            ]
            if hint_rows:
                candidates.extend(hint_rows)

            candidates.extend(
                self.rag_store.get_patterns(query=query, tags=["moe", "fp8", "triton"], k=8)
            )
            if last_error:
                candidates.extend(self.rag_store.get_error_fixes(last_error))

            # De-duplicate by ID and keep insertion order.
            seen: set[str] = set()
            patterns: list[RetrievedPattern] = []
            for idx, row in enumerate(candidates):
                rid = str(row.get("id") or f"anon::{idx}")
                if rid in seen:
                    continue
                seen.add(rid)
                patterns.append(
                    RetrievedPattern(
                        id=rid,
                        category=str(row.get("category", "pattern")),
                        relevance_score=max(0.0, 1.0 - idx * 0.08),
                        description=self._description(row),
                        code_hint=str(row.get("code_hint", "")).strip(),
                        constraint=str(row.get("constraint", "")).strip(),
                    )
                )
                if len(patterns) >= self.max_patterns:
                    break

            return AgentResult(ok=True, data={"patterns": [p.to_dict() for p in patterns], "query": query})
        except Exception as exc:
            return AgentResult(ok=False, error=f"RAG retrieval failed: {exc}")

    @staticmethod
    def _hard_guardrails(last_error: str) -> list[dict[str, Any]]:
        """Always inject critical runtime/correctness guardrails."""
        rows: list[dict[str, Any]] = [
            {
                "id": "guard::fp8_pre_dot_dequant",
                "category": "guardrail",
                "text": (
                    "FP8 dequantization must happen before tl.dot; avoid bf16/fp16 * fp32 "
                    "promotion leaks by explicit recast."
                ),
                "code_hint": (
                    "a_fp16 = (a.to(tl.float16) * a_scale[:, None]).to(tl.float16); "
                    "acc += tl.dot(a_fp16, w_fp16)"
                ),
                "constraint": "No fp32 operands inside tl.dot()",
            },
            {
                "id": "guard::n_tail_load_mask",
                "category": "guardrail",
                "text": (
                    "Any tl.load from pointers indexed by offs_n/offs_bn must have N-tail protection. "
                    "Use `% N` in pointer arithmetic or mask with `(offs_n < N)`."
                ),
                "code_hint": (
                    "w = tl.load(w_ptrs, mask=(offs_k[:, None] < K_rem) & (offs_n[None, :] < N), other=0.0)"
                ),
                "constraint": "Prevent OOB global-memory access on tail tiles",
            },
            {
                "id": "guard::sorted_pair_indexing",
                "category": "guardrail",
                "text": (
                    "In sorted token dispatch, distinguish token-id vs pair-index. Use token-id only for original "
                    "hidden-state/routing tensors; use sorted pair index (offs_token_id) for intermediate buffers "
                    "shaped by num_tokens (e.g., gemm1_output, C_bf16, gemm2_output)."
                ),
                "code_hint": (
                    "offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)\n"
                    "offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)\n"
                    "a_hidden_ptrs = hidden_ptr + offs_token[:, None] * stride_am\n"
                    "c_ptrs = c_ptr + offs_token_id[:, None] * stride_cm"
                ),
                "constraint": "Prevent semantic indexing mismatch and incorrect numerical results",
            },
        ]

        lowered = last_error.lower()
        if (
            "operation not supported on global/shared address space" in lowered
            or "acceleratorerror" in lowered
        ):
            rows.insert(
                0,
                {
                    "id": "guard::async_cuda_error_oob",
                    "category": "guardrail",
                    "text": (
                        "Async CUDA accelerator errors can be downstream symptoms of earlier illegal memory "
                        "access. Audit tl.load/tl.store masks for N/K tails first."
                    ),
                    "code_hint": (
                        "ptr uses offs_n -> add mask including offs_n < N for load/store tail blocks."
                    ),
                    "constraint": "No unmasked tail memory access",
                },
            )
        return rows

    @staticmethod
    def _extract_signal(source: str, max_lines: int = 30) -> str:
        signals: list[str] = []
        for line in source.splitlines():
            if "tl.dot(" in line or "fp8" in line.lower() or "dequant" in line.lower():
                signals.append(line.strip())
            if len(signals) >= max_lines:
                break
        return " ".join(signals)

    @staticmethod
    def _description(row: dict[str, Any]) -> str:
        for key in ("text", "pattern", "fix", "error", "when", "cause"):
            val = str(row.get(key, "")).strip()
            if val:
                return val[:280]
        return "No description available."

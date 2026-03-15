"""Coordinator agent orchestrating multi-stage generation and validation."""

from __future__ import annotations

import difflib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.agents.constraint_agent import ConstraintAgent
from kernelforge.agents.feedback_agent import FeedbackAgent
from kernelforge.agents.generator_agent import GeneratorAgent
from kernelforge.agents.pre_validation_agent import PreValidationAgent
from kernelforge.agents.rag_agent import RagAgent
from kernelforge.agents.validation_agent import ValidationAgent


@dataclass(frozen=True)
class MutationResult:
    name: str
    params: dict[str, Any]
    source: str
    diff: str


@dataclass(frozen=True)
class Proposal:
    mutations: list[MutationResult]
    source: str
    reasoning: str = ""


class CoordinatorAgent:
    """End-to-end mutation coordinator with retry and targeted feedback loops."""

    def __init__(
        self,
        seed: int,
        rag_store: Any,
        log_dir: Path | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        max_attempts: int = 4,
        verbose: bool = True,
    ) -> None:
        self.rng = random.Random(seed)
        self.log_dir = log_dir
        self.max_attempts = max_attempts
        self.verbose = verbose

        self.rag_agent = RagAgent(rag_store=rag_store, max_patterns=5)
        self.generator_agent = GeneratorAgent(model=model)
        self.pre_validation_agent = PreValidationAgent()
        self.validation_agent = ValidationAgent()
        self.constraint_agent = ConstraintAgent(max_shared_mem_kb=227)
        self.feedback_agent = FeedbackAgent()

        self._iteration = 0
        self._run_history: list[dict[str, Any]] = []

    def propose(
        self,
        source: str,
        retrieved: list[dict[str, Any]] | None = None,
        max_mutations: int = 2,
        last_error: str = "",
    ) -> Proposal:
        del max_mutations  # Coordinator currently emits a single focused mutation.

        retrieved = retrieved or []
        task_instruction = self._task_instruction(source, last_error)
        attempt_error = self._detailed_error(last_error)
        correctness_mode = (
            "incorrect_numerical" in attempt_error.lower()
            or "max_abs_error" in attempt_error.lower()
            or "correctness_error" in attempt_error.lower()
        )
        feedback: dict[str, Any] = {}
        previous_generated = ""
        iter_start = time.perf_counter()

        iter_trace: dict[str, Any] = {
            "iteration": self._iteration,
            "task_instruction": task_instruction,
            "input_last_error": last_error,
            "attempts": [],
        }

        for attempt in range(self.max_attempts):
            attempt_start = time.perf_counter()
            attempt_trace: dict[str, Any] = {"attempt": attempt, "timings_sec": {}}
            self._print(f"    [attempt {attempt + 1}/{self.max_attempts}] starting")

            t0 = time.perf_counter()
            self._print("      -> RAG agent")
            rag_result = self.rag_agent.run(
                {
                    "source": source,
                    "last_error": attempt_error,
                    "mutation_goal": task_instruction,
                    "retrieved_hint": retrieved,
                }
            )
            attempt_trace["timings_sec"]["rag"] = round(time.perf_counter() - t0, 4)
            attempt_trace["rag"] = rag_result.data if rag_result.ok else {"error": rag_result.error}
            if rag_result.ok:
                pattern_count = len(rag_result.data.get("patterns", []))
                self._print(f"         RAG ok ({pattern_count} patterns)")
            else:
                self._print(f"         RAG failed: {rag_result.error[:120]}")

            patterns = rag_result.data.get("patterns", []) if rag_result.ok else retrieved[:5]

            t0 = time.perf_counter()
            self._print("      -> Generator agent")
            gen_result = self.generator_agent.run(
                {
                    "source": source,
                    "patterns": patterns,
                    "attempt": attempt,
                    "task_instruction": task_instruction,
                    "error": attempt_error,
                    "feedback": feedback,
                    "previous_generated": previous_generated,
                }
            )
            attempt_trace["timings_sec"]["generator"] = round(time.perf_counter() - t0, 4)
            attempt_trace["generator"] = {
                "ok": gen_result.ok,
                "error": gen_result.error,
                "mutation_name": gen_result.data.get("mutation_name", ""),
                "reasoning": gen_result.data.get("reasoning", ""),
            }
            if gen_result.ok:
                self._print(
                    f"         Generator ok ({gen_result.data.get('mutation_name', 'unnamed')})"
                )
            else:
                self._print(f"         Generator failed: {gen_result.error[:140]}")

            if not gen_result.ok:
                self._print("      -> Feedback agent")
                feedback = self.feedback_agent.run(
                    {
                        "stage": "generator",
                        "error": gen_result.error,
                        "attempt": attempt,
                        "report": {},
                    }
                ).data
                attempt_error = gen_result.error
                attempt_trace["feedback"] = feedback
                attempt_trace["timings_sec"]["attempt_total"] = round(
                    time.perf_counter() - attempt_start, 4
                )
                self._print("         Attempt failed at generator stage")
                iter_trace["attempts"].append(attempt_trace)
                self._log_attempt(attempt, gen_result.data, attempt_trace)
                continue

            generated_source = str(gen_result.data.get("source", ""))
            patch_text = str(gen_result.data.get("patch", ""))
            if patch_text.strip():
                try:
                    generated_source = _apply_search_replace_patch(source, patch_text)
                    attempt_trace["patch_mode"] = "search_replace"
                except Exception as exc:
                    self._print(f"         Patch apply failed: {str(exc)[:140]}")
                    self._print("      -> Feedback agent")
                    feedback = self.feedback_agent.run(
                        {
                            "stage": "generator",
                            "error": f"Patch apply failed: {exc}",
                            "attempt": attempt,
                            "report": {"patch": patch_text[:8000]},
                        }
                    ).data
                    attempt_error = f"Patch apply failed: {exc}"
                    attempt_trace["feedback"] = feedback
                    attempt_trace["timings_sec"]["attempt_total"] = round(
                        time.perf_counter() - attempt_start, 4
                    )
                    self._print("         Attempt failed at patch-apply stage")
                    iter_trace["attempts"].append(attempt_trace)
                    self._log_attempt(attempt, gen_result.data, attempt_trace)
                    continue
            elif not generated_source.strip():
                self._print("      -> Feedback agent")
                feedback = self.feedback_agent.run(
                    {
                        "stage": "generator",
                        "error": "Attempt did not return a patch.",
                        "attempt": attempt,
                        "report": {},
                    }
                ).data
                attempt_error = "Attempt did not return a patch."
                attempt_trace["feedback"] = feedback
                attempt_trace["timings_sec"]["attempt_total"] = round(
                    time.perf_counter() - attempt_start, 4
                )
                self._print("         Attempt failed at patch-apply stage")
                iter_trace["attempts"].append(attempt_trace)
                self._log_attempt(attempt, gen_result.data, attempt_trace)
                continue

            if correctness_mode and _is_tuning_only_change(source, generated_source):
                self._print("      -> Feedback agent")
                feedback = self.feedback_agent.run(
                    {
                        "stage": "generator",
                        "error": (
                            "Previous run failed correctness, but patch is tuning-only "
                            "(tiles/warps/stages) without semantic/indexing fixes."
                        ),
                        "attempt": attempt,
                        "report": {},
                    }
                ).data
                attempt_error = (
                    "Correctness failed previously; apply semantic fix "
                    "(token/pair indexing, expert/block mapping, masking) before tuning."
                )
                attempt_trace["feedback"] = feedback
                attempt_trace["timings_sec"]["attempt_total"] = round(
                    time.perf_counter() - attempt_start, 4
                )
                self._print("         Attempt rejected: tuning-only change under correctness failure")
                iter_trace["attempts"].append(attempt_trace)
                self._log_attempt(attempt, gen_result.data, attempt_trace)
                continue
            previous_generated = generated_source

            t0 = time.perf_counter()
            self._print("      -> Pre-validation agent")
            pre_result = self.pre_validation_agent.run({"source": generated_source})
            attempt_trace["timings_sec"]["pre_validation"] = round(
                time.perf_counter() - t0, 4
            )
            attempt_trace["pre_validation"] = pre_result.data
            if pre_result.ok:
                self._print("         Pre-validation ok")
            else:
                self._print(f"         Pre-validation failed: {pre_result.error[:140]}")
            if not pre_result.ok:
                self._print("      -> Feedback agent")
                feedback = self.feedback_agent.run(
                    {
                        "stage": "pre_validation",
                        "error": pre_result.error,
                        "attempt": attempt,
                        "report": pre_result.data,
                    }
                ).data
                attempt_error = pre_result.error
                attempt_trace["feedback"] = feedback
                attempt_trace["timings_sec"]["attempt_total"] = round(
                    time.perf_counter() - attempt_start, 4
                )
                self._print("         Attempt failed at pre-validation stage")
                iter_trace["attempts"].append(attempt_trace)
                self._log_attempt(attempt, gen_result.data, attempt_trace)
                continue

            t0 = time.perf_counter()
            self._print("      -> Validation agent")
            validation_result = self.validation_agent.run(
                {
                    "source": generated_source,
                    "baseline_source": source,
                }
            )
            attempt_trace["timings_sec"]["validation"] = round(time.perf_counter() - t0, 4)
            attempt_trace["validation"] = validation_result.data
            if validation_result.ok:
                self._print("         Validation ok")
            else:
                self._print(f"         Validation failed: {validation_result.error[:140]}")
            if not validation_result.ok:
                self._print("      -> Feedback agent")
                feedback = self.feedback_agent.run(
                    {
                        "stage": "validation",
                        "error": validation_result.error,
                        "attempt": attempt,
                        "report": validation_result.data,
                    }
                ).data
                attempt_error = validation_result.error
                attempt_trace["feedback"] = feedback
                attempt_trace["timings_sec"]["attempt_total"] = round(
                    time.perf_counter() - attempt_start, 4
                )
                self._print("         Attempt failed at validation stage")
                iter_trace["attempts"].append(attempt_trace)
                self._log_attempt(attempt, gen_result.data, attempt_trace)
                continue

            t0 = time.perf_counter()
            self._print("      -> Constraint agent")
            constraint_result = self.constraint_agent.run({"source": generated_source})
            attempt_trace["timings_sec"]["constraint"] = round(time.perf_counter() - t0, 4)
            attempt_trace["constraint"] = constraint_result.data
            if constraint_result.ok:
                self._print("         Constraint check ok")
            else:
                self._print(f"         Constraint check failed: {constraint_result.error[:140]}")
            if not constraint_result.ok:
                self._print("      -> Feedback agent")
                feedback = self.feedback_agent.run(
                    {
                        "stage": "constraint",
                        "error": constraint_result.error,
                        "attempt": attempt,
                        "report": constraint_result.data,
                    }
                ).data
                attempt_error = constraint_result.error
                attempt_trace["feedback"] = feedback
                attempt_trace["timings_sec"]["attempt_total"] = round(
                    time.perf_counter() - attempt_start, 4
                )
                self._print("         Attempt failed at constraint stage")
                iter_trace["attempts"].append(attempt_trace)
                self._log_attempt(attempt, gen_result.data, attempt_trace)
                continue

            mutation_name = str(gen_result.data.get("mutation_name", f"multi_agent_iter_{self._iteration}"))
            reasoning = str(gen_result.data.get("reasoning", ""))
            diff = _make_diff(source, generated_source)
            mutation = MutationResult(
                name=mutation_name,
                params={
                    "iteration": self._iteration,
                    "attempt": attempt,
                    "pipeline": "multi_agent",
                    "model": self.generator_agent.model,
                },
                source=generated_source,
                diff=diff,
            )

            attempt_trace["success"] = True
            attempt_trace["timings_sec"]["attempt_total"] = round(
                time.perf_counter() - attempt_start, 4
            )
            iter_trace["attempts"].append(attempt_trace)
            iter_trace["result"] = "success"
            iter_trace["successful_attempt"] = attempt
            iter_trace["total_time_sec"] = round(time.perf_counter() - iter_start, 4)
            iter_trace["agents_called"] = [
                "rag",
                "generator",
                "pre_validation",
                "validation",
                "constraint",
                "feedback_on_failure",
            ]
            self._print(
                f"    [attempt {attempt + 1}] success in {attempt_trace['timings_sec']['attempt_total']:.2f}s"
            )
            self._log_attempt(attempt, gen_result.data, attempt_trace)
            self._log_iteration_summary(iter_trace)

            self._run_history.append(
                {
                    "iter": self._iteration,
                    "mutations": [mutation_name],
                        "reasoning": reasoning,
                        "diffs": [diff],
                        "score": None,
                        "attempts_used": attempt + 1,
                    }
            )
            self._iteration += 1
            return Proposal(mutations=[mutation], source=generated_source, reasoning=reasoning)

        failure_text = str(feedback.get("fix_instructions", attempt_error or "Failed after retries."))
        failed_mutation = MutationResult(
            name=f"multi_agent_failed_iter_{self._iteration}",
            params={
                "iteration": self._iteration,
                "pipeline": "multi_agent",
                "error": attempt_error,
            },
            source=source,
            diff="",
        )

        iter_trace["result"] = "failed"
        iter_trace["final_error"] = attempt_error
        iter_trace["feedback"] = feedback
        iter_trace["total_time_sec"] = round(time.perf_counter() - iter_start, 4)
        iter_trace["agents_called"] = [
            "rag",
            "generator",
            "pre_validation",
            "validation",
            "constraint",
            "feedback",
        ]
        self._print(
            f"    all {self.max_attempts} attempts failed; last error: {attempt_error[:140]}"
        )
        self._log_iteration_summary(iter_trace)

        self._run_history.append(
            {
                "iter": self._iteration,
                "mutations": [failed_mutation.name],
                "reasoning": failure_text,
                "diffs": [""],
                "score": None,
                "error": attempt_error,
            }
        )
        self._iteration += 1

        return Proposal(
            mutations=[failed_mutation],
            source=source,
            reasoning=(
                f"Multi-agent pipeline failed after {self.max_attempts} attempts. "
                f"Latest guidance: {failure_text}"
            ),
        )

    def update_last_run(self, score: float, summary: dict[str, Any], error: str = "") -> None:
        if self._run_history:
            self._run_history[-1]["score"] = score
            self._run_history[-1]["summary"] = summary
            self._run_history[-1]["status_summary"] = summary.get("status_summary", "")
            self._run_history[-1]["error"] = error

    def _task_instruction(self, source: str, last_error: str) -> str:
        lowered = last_error.lower()

        has_per_expert_loop = "for le in range(E_LOCAL)" in source or "for le in range(" in source
        has_pre_dequant_all = (
            "W13_fp32 = gemm1_weights.to(torch.float32)" in source
            or "torch.repeat_interleave(S13" in source
        )
        has_grouped_gemm = (
            "sorted_ids" in source
            and ("_grouped_gemm" in source or "tile_starts" in source)
        )
        has_fp8_dot = "tl.float8e4nv" in source or ".to(tl.float8e4nv)" in source

        if has_per_expert_loop and not has_grouped_gemm:
            return (
                "PRIORITY 1 (biggest win): Replace the per-expert Python for-loop with "
                "sort-then-grouped-GEMM. "
                "Steps: (a) sort topk_ids by local expert id → sorted_ids [N_pairs], "
                "expert_starts [E], counts [E], tile_starts [E+1]; "
                "(b) launch `_grouped_gemm1_swiglu_kernel[(NUM_SMS,)]` with `grid=(148,)` "
                "— persistent kernel loops over tiles across all experts; "
                "(c) launch `_grouped_gemm2_kernel[(NUM_SMS,)]` similarly; "
                "(d) gather routing weights in sorted order, write GEMM2 to flat buffer "
                "[N_pairs, H], then `accum.index_add_(0, sorted_ids, flat_out * rw[:, None])`. "
                "If the last attempt failed, fix that bug inside this grouped template rather "
                "than returning to seed-stage `_gemm1_tile_kernel` tweaks. "
                "Use the GROUPED_GEMM_FP8_TEMPLATE as your reference implementation, and "
                "use the helper names exactly as shown there. Do not emit legacy "
                "`gemm1_kernel`, `gemm2_kernel`, `expert_ids_per_block`, or Python "
                "`token_expert_pairs` list-building."
            )

        # ── Error recovery: always fix errors before optimizing ──────────────
        if (
            "operation not supported on global/shared address space" in lowered
            or "acceleratorerror" in lowered
        ):
            return (
                "Fix probable illegal memory access: ensure all offs_n/offs_bn tail loads and "
                "stores are N-bounded (`offs_n < N`) or safely wrapped with `% N`. "
                "If you are attempting grouped FP8, keep the vetted helper names exactly and "
                "do not fall back to legacy `gemm1_kernel`/`gemm2_kernel` dispatch."
            )
        if "incorrect_numerical" in lowered or "max_abs_error" in lowered:
            return (
                "Fix numerical correctness first. Common causes in grouped GEMM: "
                "(1) token-id vs pair-index mixup — use tok_id (from sorted_ids) for "
                "hidden_states/routing, pair-position for flat intermediate buffers; "
                "(2) block-scale application — apply `acc *= a_scale[:, None] * w_scale[None, :]` "
                "every 128 K-steps, not once at the end; "
                "(3) SwiGLU gate/up ordering — gate is rows [0..I), up is rows [I..2I)."
            )
        if last_error:
            return (
                "Fix the runtime/compile issue first with minimal edits. "
                "Do not introduce broad rewrites in the same attempt."
            )

        # ── Seed stage: no @triton.jit yet ───────────────────────────────────
        if "def _gemm1_subpath(" in source and "@triton.jit" not in source:
            return (
                "Use the vetted GEMM1-only Triton micro-kernel template. Replace only "
                "`_gemm1_subpath(...)`; keep routing, SwiGLU, GEMM2, and final index_add_ "
                "in PyTorch. Do not introduce more than one @triton.jit kernel in this attempt."
            )

        # Priority 2 (saves ~93% memory traffic): Eliminate pre-dequantization
        if has_pre_dequant_all and not has_fp8_dot:
            return (
                "PRIORITY 2: Eliminate pre-dequantization of all expert weights. "
                "Remove the W13_fp32/W13/W2_fp32/W2 blocks that expand all 32 experts upfront. "
                "Instead, inside the Triton kernel load FP8 weights directly and cast to "
                "`tl.float8e4nv` before `tl.dot`. Apply block scales to the FP32 accumulator "
                "every Q_BLOCK=128 K-steps: `acc = acc * a_scale[:, None] * w_scale[None, :]`. "
                "Activation scale layout: hidden_states_scale[H//128, T] → load as "
                "`tl.load(A_scale_ptr + k_blk * T + tok_ids)`. "
                "Weight scale layout: gemm1_weights_scale[E, 2I//128, H//128]."
            )

        # Priority 3: Fuse SwiGLU into GEMM1 kernel (eliminate intermediate write)
        if has_grouped_gemm and has_fp8_dot and "swiglu" not in source.lower():
            return (
                "PRIORITY 3: Fuse SwiGLU activation into the GEMM1 Triton kernel. "
                "After computing the gate [0..I) and up [I..2I) accumulators, apply "
                "`swiglu = silu(gate) * up` in registers before writing to the output buffer. "
                "Output shape changes from [N_pairs, 2I] to [N_pairs, I]."
            )

        # Priority 4: Tile size and pipeline depth improvements
        if self._iteration >= 2:
            return (
                "PRIORITY 4: Tune tile sizes and pipeline depth for B200. "
                "Try BLOCK_M=128, BLOCK_N=128 or 256, BLOCK_K=64, num_stages=3, num_warps=8. "
                "With FP8 (1 byte/elem): 3*(128*64 + 256*64)*1 = 73KB SMEM — within budget. "
                "Also consider grouped launch ordering: sort tiles by expert for L2 reuse. "
                "Add @triton.autotune over these configs."
            )

        return (
            "Apply the next focused optimization: improve FP8 block-scale application, "
            "memory access coalescing, or pipeline depth. Keep correctness intact."
        )

    def _detailed_error(self, last_error: str) -> str:
        if not last_error or not self.log_dir or self._iteration <= 0:
            return last_error

        error_file = self.log_dir / f"iter_{self._iteration - 1:03d}" / "benchmark_error.txt"
        if not error_file.exists():
            return last_error

        try:
            content = error_file.read_text(encoding="utf-8")
        except Exception:
            return last_error

        if len(content) <= 3000:
            return content

        lines = content.splitlines()
        excerpt = "\n".join(lines[:40])
        return f"{excerpt}\n... ({len(lines) - 40} lines truncated)"

    def _log_attempt(self, attempt: int, generator_data: dict[str, Any], attempt_trace: dict[str, Any]) -> None:
        if not self.log_dir:
            return

        try:
            iter_dir = self.log_dir / f"iter_{self._iteration:03d}" / f"attempt_{attempt:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            system_prompt = str(generator_data.get("system_prompt", ""))
            user_prompt = str(generator_data.get("user_prompt", ""))
            raw_response = str(generator_data.get("raw_response", ""))
            generated_source = str(generator_data.get("source", ""))
            patch_text = str(generator_data.get("patch", ""))

            if system_prompt:
                (iter_dir / "generator_system_prompt.txt").write_text(system_prompt, encoding="utf-8")
            if user_prompt:
                (iter_dir / "generator_user_prompt.txt").write_text(user_prompt, encoding="utf-8")
            if raw_response:
                (iter_dir / "generator_raw_response.txt").write_text(raw_response, encoding="utf-8")
            if patch_text:
                (iter_dir / "generator_patch.txt").write_text(patch_text, encoding="utf-8")
            if generated_source:
                (iter_dir / "generated_kernel.py").write_text(generated_source, encoding="utf-8")

            (iter_dir / "attempt_trace.json").write_text(
                json.dumps(attempt_trace, indent=2), encoding="utf-8"
            )
        except Exception:
            # Logging should never crash the optimization loop.
            pass

    def _log_iteration_summary(self, summary: dict[str, Any]) -> None:
        if not self.log_dir:
            return
        try:
            iter_dir = self.log_dir / f"iter_{self._iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "coordinator_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def _print(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)


def _make_diff(old: str, new: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile="before/kernel.py",
            tofile="after/kernel.py",
            lineterm="",
        )
    )


_PATCH_BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _apply_search_replace_patch(source: str, patch_text: str) -> str:
    patch = _strip_code_fences(patch_text).replace("\r\n", "\n")
    matches = list(_PATCH_BLOCK_RE.finditer(patch))
    if not matches:
        raise ValueError("no SEARCH/REPLACE blocks found")

    updated = source
    for match in matches:
        search = match.group(1)
        replace = match.group(2)
        if search not in updated:
            raise ValueError("SEARCH block not found in baseline source")
        updated = updated.replace(search, replace, 1)
    return updated


def _is_tuning_only_change(old: str, new: str) -> bool:
    if old == new:
        return True
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""))
    changed_lines = [
        line[1:]
        for line in diff
        if line and line[0] in {"+", "-"} and not line.startswith(("+++", "---"))
    ]
    if not changed_lines:
        return True

    semantic_markers = {
        "sorted_tokens",
        "expert_ids_per_block",
        "token_mask",
        "offs_token",
        "offs_token_id",
        "num_tokens_post_padded",
        "index_add_",
        "routing_weights",
        "topk_idx",
        "weights =",
        "expert_to_tokens",
        "packed_tokens",
    }
    for line in changed_lines:
        lower = line.lower()
        if any(marker.lower() in lower for marker in semantic_markers):
            return False

    tuning_markers = (
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "DEFAULT_NUM_WARPS",
        "DEFAULT_NUM_STAGES",
    )
    for line in changed_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if any(marker in stripped for marker in tuning_markers):
            continue
        # Any non-marker code line is likely semantic enough.
        if "=" in stripped or "(" in stripped or ")" in stripped:
            return False
    return True

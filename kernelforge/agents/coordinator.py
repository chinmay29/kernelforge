"""Coordinator agent orchestrating multi-stage generation and validation."""

from __future__ import annotations

import difflib
import json
import random
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
        max_attempts: int = 3,
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
        task_instruction = self._task_instruction(last_error)
        attempt_error = self._detailed_error(last_error)
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
            validation_result = self.validation_agent.run({"source": generated_source})
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

    def _task_instruction(self, last_error: str) -> str:
        lowered = last_error.lower()
        if (
            "operation not supported on global/shared address space" in lowered
            or "acceleratorerror" in lowered
        ):
            return (
                "Fix probable illegal memory access first: ensure all offs_n/offs_bn tail loads and stores are "
                "N-bounded (`offs_n < N`) or safely wrapped with `% N`."
            )
        if "incorrect_numerical" in lowered or "max_abs_error" in lowered:
            return (
                "Fix numerical correctness first. In sorted dispatch, ensure token-id vs pair-index semantics: "
                "use token-id for hidden/routing tensors and offs_token_id for intermediates sized by num_tokens."
            )
        if last_error:
            return (
                "Fix the runtime/compile issue first with minimal edits. "
                "Do not introduce broad rewrites in the same attempt."
            )
        if self._iteration == 0:
            return (
                "Convert one high-impact GEMM path to Triton with correct FP8 dequantization, "
                "and keep the rest stable."
            )
        return (
            "Apply one focused optimization that improves speed while preserving numerical correctness "
            "and Triton constraints."
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

            if system_prompt:
                (iter_dir / "generator_system_prompt.txt").write_text(system_prompt, encoding="utf-8")
            if user_prompt:
                (iter_dir / "generator_user_prompt.txt").write_text(user_prompt, encoding="utf-8")
            if raw_response:
                (iter_dir / "generator_raw_response.txt").write_text(raw_response, encoding="utf-8")
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

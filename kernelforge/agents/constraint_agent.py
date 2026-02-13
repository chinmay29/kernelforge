"""Constraint verification agent for hardware and API guardrails."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from kernelforge.agents.base import Agent, AgentResult


@dataclass(frozen=True)
class ConstraintViolation:
    constraint: str
    severity: str
    message: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint,
            "severity": self.severity,
            "message": self.message,
            "line": self.line,
        }


@dataclass(frozen=True)
class ConstraintReport:
    ok: bool
    violations: list[ConstraintViolation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "violations": [v.to_dict() for v in self.violations],
        }


class ConstraintAgent(Agent):
    """Verify B200 and Triton constraints with deterministic checks."""

    name = "constraint"

    _DOT_ARGS_RE = re.compile(r"tl\.dot\(([^,]+),\s*([^)]+)\)")
    _CAST_ASSIGN_RE = re.compile(r"(\w+)\s*=\s*.*\.to\(tl\.(?:float16|bfloat16)\)")

    def __init__(self, max_shared_mem_kb: int = 227) -> None:
        self.max_shared_mem_kb = max_shared_mem_kb

    def run(self, inputs: dict[str, Any]) -> AgentResult:
        source = str(inputs.get("source", ""))
        report = self.verify(source)

        error = ""
        for violation in report.violations:
            if violation.severity == "error":
                line_info = f" line {violation.line}" if violation.line else ""
                error = f"{violation.constraint}{line_info}: {violation.message}"
                break

        return AgentResult(ok=report.ok, error=error, data=report.to_dict())

    def verify(self, source: str) -> ConstraintReport:
        lines = source.splitlines()
        violations: list[ConstraintViolation] = []

        violations.extend(self._check_shared_memory(lines))
        violations.extend(self._check_dot_operands(lines))
        violations.extend(self._check_atomic_add(lines))
        violations.extend(self._check_n_load_bounds(lines))
        violations.extend(self._check_access_pattern(lines))

        has_errors = any(v.severity == "error" for v in violations)
        return ConstraintReport(ok=not has_errors, violations=violations)

    def _check_shared_memory(self, lines: list[str]) -> list[ConstraintViolation]:
        block_m = 128
        block_n = 64
        block_k = 64
        num_stages = 3

        patterns = [
            (re.compile(r"BLOCK_SIZE_M\s*=\s*(\d+)"), "M"),
            (re.compile(r"BLOCK_SIZE_N\s*=\s*(\d+)"), "N"),
            (re.compile(r"BLOCK_SIZE_K\s*=\s*(\d+)"), "K"),
            (re.compile(r"BLOCK_M\s*=\s*(\d+)"), "M"),
            (re.compile(r"BLOCK_N\s*=\s*(\d+)"), "N"),
            (re.compile(r"BLOCK_K\s*=\s*(\d+)"), "K"),
            (re.compile(r"NUM_STAGES\s*=\s*(\d+)"), "S"),
            (re.compile(r"num_stages\s*=\s*(\d+)"), "S"),
        ]

        for line in lines:
            stripped = line.strip()
            for pat, key in patterns:
                m = pat.search(stripped)
                if not m:
                    continue
                value = int(m.group(1))
                if key == "M":
                    block_m = value
                elif key == "N":
                    block_n = value
                elif key == "K":
                    block_k = value
                elif key == "S":
                    num_stages = value

        elem_bytes = 2
        smem_kb = ((block_m * block_k + block_k * block_n) * elem_bytes * num_stages) / 1024
        if smem_kb > self.max_shared_mem_kb:
            return [
                ConstraintViolation(
                    constraint="smem_limit",
                    severity="error",
                    message=(
                        f"Estimated shared memory {smem_kb:.0f}KB exceeds {self.max_shared_mem_kb}KB "
                        f"(BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}, NUM_STAGES={num_stages})."
                    ),
                )
            ]
        return []

    def _check_dot_operands(self, lines: list[str]) -> list[ConstraintViolation]:
        casted_vars: set[str] = set()
        violations: list[ConstraintViolation] = []

        for line in lines:
            m = self._CAST_ASSIGN_RE.match(line.strip())
            if m:
                casted_vars.add(m.group(1))

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if "tl.dot(" not in stripped:
                continue
            match = self._DOT_ARGS_RE.search(stripped)
            if not match:
                continue
            arg1 = match.group(1).strip()
            arg2 = match.group(2).strip()

            for arg in (arg1, arg2):
                if "fp8" in arg.lower():
                    violations.append(
                        ConstraintViolation(
                            constraint="fp8_dot_operand",
                            severity="error",
                            message="tl.dot does not support FP8 operands directly. Dequantize before tl.dot.",
                            line=idx,
                        )
                    )

            # Warning if neither arg appears to be explicitly cast/dequantized.
            if arg1 not in casted_vars and arg2 not in casted_vars:
                violations.append(
                    ConstraintViolation(
                        constraint="dequantization_signal",
                        severity="warning",
                        message=(
                            "No explicit cast/dequantization signal found for tl.dot operands on this line. "
                            "Verify FP8 block-scale dequantization happens before dot."
                        ),
                        line=idx,
                    )
                )

            if ".to(tl.float32)" in stripped:
                violations.append(
                    ConstraintViolation(
                        constraint="tl_dot_fp32",
                        severity="error",
                        message="tl.dot operands should not be tl.float32.",
                        line=idx,
                    )
                )

        return violations

    @staticmethod
    def _check_atomic_add(lines: list[str]) -> list[ConstraintViolation]:
        for idx, line in enumerate(lines, start=1):
            if "tl.atomic_add" in line:
                return [
                    ConstraintViolation(
                        constraint="atomic_add_fp32",
                        severity="error",
                        message="Avoid tl.atomic_add fp32 accumulation; use post-kernel index_add_.",
                        line=idx,
                    )
                ]
        return []

    @staticmethod
    def _check_access_pattern(lines: list[str]) -> list[ConstraintViolation]:
        has_load = any("tl.load(" in line for line in lines)
        has_arange = any("tl.arange(" in line for line in lines)
        if has_load and not has_arange:
            return [
                ConstraintViolation(
                    constraint="coalescing_signal",
                    severity="warning",
                    message="Kernel uses tl.load but no tl.arange pattern was found; review memory coalescing.",
                )
            ]
        return []

    @staticmethod
    def _check_n_load_bounds(lines: list[str]) -> list[ConstraintViolation]:
        """Constraint-level guard for offs_n OOB loads on N-tail blocks."""
        ptr_uses_offs_n: dict[str, bool] = {}
        n_tail_guard_vars: set[str] = set()
        n_safe_index_vars: set[str] = set()
        assign_re = re.compile(r"^\s*(\w+)\s*=\s*(.+)$")
        load_var_re = re.compile(r"tl\.load\(\s*(\w+)")

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            assign_match = assign_re.match(stripped)
            if assign_match:
                lhs = assign_match.group(1)
                rhs = assign_match.group(2)
                rhs_no_space = rhs.replace(" ", "")
                if ("offs_n<N" in rhs_no_space) or ("offs_bn<N" in rhs_no_space):
                    n_tail_guard_vars.add(lhs)
                if ("%N" in rhs_no_space) and ("offs_n" in rhs or "offs_bn" in rhs):
                    n_safe_index_vars.add(lhs)
                if "offs_n" in rhs:
                    ptr_uses_offs_n[lhs] = (
                        ("% N" in rhs)
                        or ("%N" in rhs_no_space)
                        or ("mod" in rhs.lower())
                        or any(var in rhs for var in n_safe_index_vars)
                    )

            if "tl.load(" not in stripped:
                continue

            var_match = load_var_re.search(stripped)
            if var_match:
                ptr = var_match.group(1)
                if ptr in ptr_uses_offs_n:
                    has_mod_n = ptr_uses_offs_n[ptr]
                    has_mask = "mask=" in stripped
                    has_n_bound = (
                        ("offs_n" in stripped and "< N" in stripped)
                        or ("offs_bn" in stripped and "< N" in stripped)
                        or any(f"{name}" in stripped for name in n_tail_guard_vars)
                    )
                    if not has_mod_n and (not has_mask or not has_n_bound):
                        return [
                            ConstraintViolation(
                                constraint="n_tail_oob_load",
                                severity="error",
                                message=(
                                    f"Potential OOB N-tail load for `{ptr}`. Add `offs_n < N` mask "
                                    "or wrap pointer indices with `% N`."
                                ),
                                line=idx,
                            )
                        ]

            if "offs_n" in stripped and "tl.load(" in stripped:
                has_mod_n = "% N" in stripped
                has_mask = "mask=" in stripped
                has_n_bound = "< N" in stripped
                if not has_mod_n and (not has_mask or not has_n_bound):
                    return [
                        ConstraintViolation(
                            constraint="n_tail_oob_load",
                            severity="error",
                            message="Inline offs_n load is missing N-bound protection.",
                            line=idx,
                        )
                    ]

        return []

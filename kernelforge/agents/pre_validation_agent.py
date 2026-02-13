"""Fast critical checks before full validation/benchmarking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from kernelforge.agents.base import Agent, AgentResult


@dataclass(frozen=True)
class PreValidationReport:
    ok: bool
    error: str = ""
    fix: str = ""
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "fix": self.fix,
            "line": self.line,
        }


class PreValidationAgent(Agent):
    """Catch repeated high-cost mistakes (type promotion, FP8 misuse) quickly."""

    name = "pre_validation"

    _ASSIGN_RE = re.compile(r"^(?P<lhs>\w+)\s*=\s*(?P<rhs>.+)$")

    def run(self, inputs: dict[str, Any]) -> AgentResult:
        source = str(inputs.get("source", ""))
        report = self.check_fp8_patterns(source)
        return AgentResult(ok=report.ok, error=report.error, data=report.to_dict())

    def check_fp8_patterns(self, source: str) -> PreValidationReport:
        lines = source.splitlines()
        for idx, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ".to(tl.float16)" in line and "*" in line and "=" in line:
                bug = self._type_promotion_bug(line, dtype="float16")
                if bug:
                    return PreValidationReport(
                        ok=False,
                        error=(
                            f"line {idx}: potential FP16 type-promotion bug. "
                            f"`{line[:150]}` can promote to fp32 after multiplication."
                        ),
                        fix=(
                            "Cast result back to tl.float16: "
                            "`x_fp16 = (x.to(tl.float16) * scale).to(tl.float16)` "
                            "or cast scale first."
                        ),
                        line=idx,
                    )
            if ".to(tl.bfloat16)" in line and "*" in line and "=" in line:
                bug = self._type_promotion_bug(line, dtype="bfloat16")
                if bug:
                    return PreValidationReport(
                        ok=False,
                        error=(
                            f"line {idx}: potential BF16 type-promotion bug. "
                            f"`{line[:150]}` can promote to fp32 after multiplication."
                        ),
                        fix=(
                            "Cast result back to tl.bfloat16: "
                            "`x_bf16 = (x.to(tl.bfloat16) * scale).to(tl.bfloat16)` "
                            "or cast scale first."
                        ),
                        line=idx,
                    )

            if "tl.dot(" in line and ".to(tl.float32)" in line:
                return PreValidationReport(
                    ok=False,
                    error=f"line {idx}: tl.dot operand is cast to fp32, which is unsupported for operand dtypes.",
                    fix="Dequantize to fp16/bf16 before tl.dot and keep dot operands non-fp32.",
                    line=idx,
                )

        return PreValidationReport(ok=True)

    def _type_promotion_bug(self, line: str, dtype: str) -> bool:
        match = self._ASSIGN_RE.match(line)
        if not match:
            return False
        rhs = match.group("rhs")

        # Already explicitly cast back after multiply.
        if f").to(tl.{dtype})" in rhs:
            return False

        # Scale casted before multiply usually avoids promotion.
        if f"scale_{'fp16' if dtype == 'float16' else 'bf16'}" in rhs:
            return False
        if f".to(tl.{dtype})" in rhs and "scale" in rhs and rhs.count(f".to(tl.{dtype})") >= 2:
            return False

        cast_pat = f".to(tl.{dtype})"
        return cast_pat in rhs and "*" in rhs

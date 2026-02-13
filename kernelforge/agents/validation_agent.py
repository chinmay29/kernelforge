"""Structured source validation for generated kernels."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from kernelforge.agents.base import Agent, AgentResult


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    ok: bool
    message: str = ""
    line: int | None = None
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "message": self.message,
            "line": self.line,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    checks: list[ValidationCheck]

    @property
    def first_failure(self) -> ValidationCheck | None:
        for check in self.checks:
            if not check.ok:
                return check
        return None

    def to_dict(self) -> dict[str, Any]:
        first = self.first_failure
        return {
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
            "first_failure": first.to_dict() if first else None,
        }


class ValidationAgent(Agent):
    """Comprehensive Python + Triton safety validator."""

    name = "validation"

    _FP32_CAST_RE = re.compile(r"(\w+)\s*=\s*.*\.to\(tl\.float32\)")
    _DOT_ARGS_RE = re.compile(r"tl\.dot\(([^,]+),\s*([^)]+)\)")

    def run(self, inputs: dict[str, Any]) -> AgentResult:
        source = str(inputs.get("source", ""))
        report = self.validate(source)
        error = ""
        if not report.ok and report.first_failure:
            first = report.first_failure
            line_info = f" line {first.line}" if first.line else ""
            error = f"{first.name}{line_info}: {first.message}"
        return AgentResult(ok=report.ok, error=error, data=report.to_dict())

    def validate(self, source: str) -> ValidationReport:
        checks: list[ValidationCheck] = []
        lines = source.splitlines()

        syntax_check, tree = self._check_syntax(source)
        checks.append(syntax_check)

        checks.append(self._check_required_import(lines, "import torch", "required_import_torch"))
        checks.append(self._check_required_import(lines, "import triton", "required_import_triton"))
        checks.append(
            self._check_required_import(lines, "import triton.language as tl", "required_import_tl")
        )

        checks.append(self._check_required_function(lines))
        checks.append(self._check_return_statement(lines))

        if tree is not None:
            checks.append(self._check_global_access(tree))
        else:
            checks.append(
                ValidationCheck(
                    name="triton_global_access",
                    ok=False,
                    message="Skipped because Python syntax is invalid.",
                )
            )

        checks.append(self._check_tl_dot_fp32(lines))
        checks.append(self._check_atomic_add(lines))
        checks.append(self._check_n_dimension_load_bounds(lines))
        checks.append(self._check_sorted_token_indexing_consistency(lines))
        checks.append(self._check_shared_memory(lines))
        checks.append(self._check_type_promotion(lines))

        return ValidationReport(ok=all(c.ok for c in checks), checks=checks)

    def _check_syntax(self, source: str) -> tuple[ValidationCheck, ast.AST | None]:
        try:
            tree = ast.parse(source)
            return ValidationCheck(name="python_syntax", ok=True), tree
        except SyntaxError as exc:
            return (
                ValidationCheck(
                    name="python_syntax",
                    ok=False,
                    message=exc.msg,
                    line=exc.lineno,
                ),
                None,
            )

    @staticmethod
    def _check_required_import(
        lines: list[str], required: str, name: str
    ) -> ValidationCheck:
        for idx, line in enumerate(lines, start=1):
            if required in line:
                return ValidationCheck(name=name, ok=True, line=idx)
        return ValidationCheck(name=name, ok=False, message=f"Missing `{required}`")

    @staticmethod
    def _check_required_function(lines: list[str]) -> ValidationCheck:
        for idx, line in enumerate(lines, start=1):
            if line.strip().startswith("def kernel("):
                return ValidationCheck(name="required_kernel_function", ok=True, line=idx)
        return ValidationCheck(name="required_kernel_function", ok=False, message="Missing `def kernel(`")

    @staticmethod
    def _check_return_statement(lines: list[str]) -> ValidationCheck:
        in_kernel = False
        kernel_indent: int | None = None
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("def kernel("):
                in_kernel = True
                kernel_indent = len(line) - len(line.lstrip(" \t"))
                continue
            if in_kernel and stripped.startswith("def "):
                indent = len(line) - len(line.lstrip(" \t"))
                if indent <= (kernel_indent or 0):
                    in_kernel = False
            if in_kernel and stripped.startswith("return "):
                return ValidationCheck(name="kernel_return", ok=True, line=idx)

        return ValidationCheck(name="kernel_return", ok=False, message="`def kernel` has no return statement")

    @staticmethod
    def _check_global_access(tree: ast.AST) -> ValidationCheck:
        module_globals: set[str] = set()
        jit_funcs: dict[str, ast.FunctionDef] = {}

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_globals.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                module_globals.add(node.target.id)
            elif isinstance(node, ast.FunctionDef):
                if ValidationAgent._is_triton_jit(node):
                    jit_funcs[node.name] = node

        safe_names = {
            "tl",
            "triton",
            "torch",
            "math",
            "True",
            "False",
            "None",
            "range",
            "min",
            "max",
            "len",
            "float",
            "int",
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "NUM_STAGES",
            "NUM_WARPS",
        }

        for name, fn in jit_funcs.items():
            params = {arg.arg for arg in fn.args.args}
            for child in ast.walk(fn):
                if isinstance(child, ast.Name) and child.id in module_globals:
                    if child.id not in params and child.id not in safe_names:
                        return ValidationCheck(
                            name="triton_global_access",
                            ok=False,
                            message=(
                                f"@triton.jit function `{name}` references Python global "
                                f"`{child.id}`. Pass as tl.constexpr argument or inline it."
                            ),
                            line=getattr(child, "lineno", None),
                        )

        return ValidationCheck(name="triton_global_access", ok=True)

    @staticmethod
    def _is_triton_jit(fn: ast.FunctionDef) -> bool:
        for deco in fn.decorator_list:
            if isinstance(deco, ast.Attribute):
                if deco.attr.lower() == "jit":
                    return True
            elif isinstance(deco, ast.Name) and deco.id.lower() == "jit":
                return True
            elif isinstance(deco, ast.Call):
                call_name = ast.unparse(deco.func) if hasattr(ast, "unparse") else ""
                if "jit" in call_name.lower():
                    return True
        return False

    def _check_tl_dot_fp32(self, lines: list[str]) -> ValidationCheck:
        fp32_vars: set[str] = set()
        for line in lines:
            stripped = line.strip()
            match = self._FP32_CAST_RE.match(stripped)
            if match:
                fp32_vars.add(match.group(1))

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if "tl.dot(" not in stripped:
                continue
            if ".to(tl.float32)" in stripped:
                return ValidationCheck(
                    name="tl_dot_operand_dtype",
                    ok=False,
                    message="tl.dot operands must be fp8/fp16/bf16, not fp32 casts.",
                    line=idx,
                )
            dot_match = self._DOT_ARGS_RE.search(stripped)
            if dot_match:
                arg1 = dot_match.group(1).strip()
                arg2 = dot_match.group(2).strip()
                for arg in (arg1, arg2):
                    if arg in fp32_vars:
                        return ValidationCheck(
                            name="tl_dot_operand_dtype",
                            ok=False,
                            message=f"tl.dot operand `{arg}` was created with tl.float32 cast.",
                            line=idx,
                        )

        return ValidationCheck(name="tl_dot_operand_dtype", ok=True)

    @staticmethod
    def _check_atomic_add(lines: list[str]) -> ValidationCheck:
        for idx, line in enumerate(lines, start=1):
            if "tl.atomic_add" in line:
                return ValidationCheck(
                    name="atomic_add_usage",
                    ok=False,
                    message="Avoid tl.atomic_add for fp32 accumulation in this workload.",
                    line=idx,
                )
        return ValidationCheck(name="atomic_add_usage", ok=True)

    @staticmethod
    def _check_n_dimension_load_bounds(lines: list[str]) -> ValidationCheck:
        """Detect loads from pointers indexed by offs_n without N-bound masks/modulo."""
        ptr_uses_offs_n: dict[str, tuple[int, bool]] = {}
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
                    has_mod_n = (
                        ("% N" in rhs)
                        or ("%N" in rhs_no_space)
                        or ("mod" in rhs.lower())
                        or any(var in rhs for var in n_safe_index_vars)
                    )
                    ptr_uses_offs_n[lhs] = (idx, has_mod_n)

            if "tl.load(" not in stripped:
                continue

            # Case 1: loading from a named ptr var derived from offs_n.
            var_match = load_var_re.search(stripped)
            if var_match:
                ptr_name = var_match.group(1)
                info = ptr_uses_offs_n.get(ptr_name)
                if info:
                    _, has_mod_n = info
                    if has_mod_n:
                        continue
                    has_mask = "mask=" in stripped
                    has_n_bound = (
                        ("offs_n" in stripped and "< N" in stripped)
                        or ("offs_bn" in stripped and "< N" in stripped)
                        or any(f"{name}" in stripped for name in n_tail_guard_vars)
                    )
                    if not has_mask or not has_n_bound:
                        return ValidationCheck(
                            name="n_dimension_load_bounds",
                            ok=False,
                            message=(
                                f"`tl.load({ptr_name}, ...)` uses pointer indexed by offs_n "
                                "without explicit `offs_n < N` mask (or modulo-N pointer). "
                                "This can cause OOB global memory access."
                            ),
                            line=idx,
                        )

            # Case 2: inline pointer expression with offs_n.
            if "tl.load(" in stripped and "offs_n" in stripped:
                has_mask = "mask=" in stripped
                has_n_bound = "< N" in stripped
                has_mod_n = "% N" in stripped
                if not has_mod_n and (not has_mask or not has_n_bound):
                    return ValidationCheck(
                        name="n_dimension_load_bounds",
                        ok=False,
                        message=(
                            "Found tl.load with offs_n-indexed pointer missing `offs_n < N` bound."
                        ),
                        line=idx,
                    )

        return ValidationCheck(name="n_dimension_load_bounds", ok=True)

    @staticmethod
    def _check_sorted_token_indexing_consistency(lines: list[str]) -> ValidationCheck:
        """Catch token-id vs pair-index mixups that often cause large numerical errors."""
        source = "\n".join(lines)
        if "sorted_token_ids_ptr" not in source:
            return ValidationCheck(name="sorted_token_indexing_consistency", ok=True)

        def _extract_fn_block(fn_name: str) -> str:
            start = None
            for i, line in enumerate(lines):
                if line.strip().startswith(f"def {fn_name}("):
                    start = i
                    break
            if start is None:
                return ""
            end = len(lines)
            for j in range(start + 1, len(lines)):
                stripped = lines[j].strip()
                if stripped.startswith("def ") or stripped.startswith("@triton.jit"):
                    end = j
                    break
            return "\n".join(lines[start:end])

        gemm1 = _extract_fn_block("gemm1_kernel")
        gemm2 = _extract_fn_block("gemm2_kernel")

        # If output is shaped by num_tokens, using offs_token (original token id) as row index
        # for intermediate/output tensors is usually incorrect; row index should be offs_token_id.
        if (
            "gemm1_output = torch.zeros((num_tokens" in source
            and "offs_token_id =" in gemm1
            and "offs_token = tl.load(sorted_token_ids_ptr" in gemm1
            and re.search(r"\boffs_m\s*=\s*offs_token\b", gemm1)
            and re.search(r"\boffs_om\s*=\s*offs_m", gemm1)
            and "out_ptrs = out_ptr +" in gemm1
        ):
            line_no = None
            for i, line in enumerate(lines, start=1):
                if "def gemm1_kernel(" in line:
                    in_g1 = True
                if "offs_m = offs_token" in line:
                    line_no = i
                    break
            return ValidationCheck(
                name="sorted_token_indexing_consistency",
                ok=False,
                message=(
                    "gemm1_kernel appears to index output rows with `offs_token` (token id) "
                    "instead of sorted pair index (`offs_token_id`). For buffers shaped "
                    "as `(num_tokens, ...)`, row indexing should use pair positions."
                ),
                line=line_no,
            )

        # For GEMM2, if input a_ptr points to C_bf16/gemm1_output space (num_tokens rows),
        # indexing a_ptr by offs_token is inconsistent.
        if (
            "C_bf16 = " in source
            and "offs_token_id =" in gemm2
            and "offs_token = tl.load(sorted_token_ids_ptr" in gemm2
            and re.search(r"\boffs_m\s*=\s*offs_token\b", gemm2)
            and re.search(r"\ba_ptrs\s*=\s*a_ptr\s*\+\s*\(offs_m\[:,\s*None\]", gemm2)
        ):
            line_no = None
            for i, line in enumerate(lines, start=1):
                if "def gemm2_kernel(" in line:
                    in_g2 = True
                if "a_ptrs = a_ptr + (offs_m[:, None]" in line:
                    line_no = i
                    break
            return ValidationCheck(
                name="sorted_token_indexing_consistency",
                ok=False,
                message=(
                    "gemm2_kernel input rows appear indexed by `offs_token` (original token ids). "
                    "When a_ptr is derived from C_bf16/gemm1_output shaped `(num_tokens, I)`, "
                    "index with sorted pair index (`offs_token_id`) instead."
                ),
                line=line_no,
            )

        return ValidationCheck(name="sorted_token_indexing_consistency", ok=True)

    @staticmethod
    def _check_shared_memory(lines: list[str]) -> ValidationCheck:
        block_m = 128
        block_n = 64
        block_k = 64
        num_stages = 3

        capture_patterns = [
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
            for pat, key in capture_patterns:
                m = pat.search(stripped)
                if not m:
                    continue
                val = int(m.group(1))
                if key == "M":
                    block_m = val
                elif key == "N":
                    block_n = val
                elif key == "K":
                    block_k = val
                elif key == "S":
                    num_stages = val

        elem_bytes = 2
        smem_bytes = (block_m * block_k + block_k * block_n) * elem_bytes * num_stages
        smem_kb = smem_bytes / 1024
        max_smem_kb = 227
        if smem_kb > max_smem_kb:
            return ValidationCheck(
                name="shared_memory_limit",
                ok=False,
                message=(
                    f"Estimated shared memory {smem_kb:.0f}KB exceeds B200 limit {max_smem_kb}KB "
                    f"(BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}, NUM_STAGES={num_stages})."
                ),
            )
        return ValidationCheck(name="shared_memory_limit", ok=True)

    @staticmethod
    def _check_type_promotion(lines: list[str]) -> ValidationCheck:
        pattern_map = {
            "float16": re.compile(r"(\w+)\s*=\s*.*\.to\(tl\.float16\)\s*\*"),
            "bfloat16": re.compile(r"(\w+)\s*=\s*.*\.to\(tl\.bfloat16\)\s*\*"),
        }

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if "=" not in stripped or "*" not in stripped:
                continue

            for dtype, pat in pattern_map.items():
                m = pat.match(stripped)
                if not m:
                    continue
                # Explicit post-cast to dtype avoids promotion leak.
                if f").to(tl.{dtype})" in stripped:
                    continue
                # Scale cast to target dtype before multiply also avoids it.
                if stripped.count(f".to(tl.{dtype})") >= 2:
                    continue
                return ValidationCheck(
                    name="type_promotion",
                    ok=False,
                    message=(
                        f"Potential type promotion: `{m.group(1)} = x.to(tl.{dtype}) * y` can produce fp32. "
                        f"Cast result back to tl.{dtype} or cast multiplier first."
                    ),
                    line=idx,
                )

        return ValidationCheck(name="type_promotion", ok=True)

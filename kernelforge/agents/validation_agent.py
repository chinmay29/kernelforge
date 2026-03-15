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
        baseline_source = inputs.get("baseline_source")
        baseline = str(baseline_source) if baseline_source else None
        report = self.validate(source, baseline_source=baseline)
        error = ""
        if not report.ok and report.first_failure:
            first = report.first_failure
            line_info = f" line {first.line}" if first.line else ""
            error = f"{first.name}{line_info}: {first.message}"
        return AgentResult(ok=report.ok, error=error, data=report.to_dict())

    def validate(self, source: str, baseline_source: str | None = None) -> ValidationReport:
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
            checks.append(self._check_kernel_signature_stability(tree, baseline_source))
            checks.append(self._check_seed_stage_scope(source, tree, baseline_source))
            checks.append(self._check_seed_stage_numerics(source, tree, baseline_source))
            checks.append(self._check_grouped_gemm_template_alignment(source, baseline_source))
            checks.append(self._check_global_access(tree))
            checks.append(self._check_kernel_reachability(tree))
        else:
            checks.append(
                ValidationCheck(
                    name="kernel_signature_stability",
                    ok=False,
                    message="Skipped because Python syntax is invalid.",
                )
            )
            checks.append(
                ValidationCheck(
                    name="seed_stage_scope",
                    ok=False,
                    message="Skipped because Python syntax is invalid.",
                )
            )
            checks.append(
                ValidationCheck(
                    name="seed_stage_numerics",
                    ok=False,
                    message="Skipped because Python syntax is invalid.",
                )
            )
            checks.append(
                ValidationCheck(
                    name="grouped_gemm_template_alignment",
                    ok=False,
                    message="Skipped because Python syntax is invalid.",
                )
            )
            checks.append(
                ValidationCheck(
                    name="triton_global_access",
                    ok=False,
                    message="Skipped because Python syntax is invalid.",
                )
            )
            checks.append(
                ValidationCheck(
                    name="kernel_reachability",
                    ok=False,
                    message="Skipped because Python syntax is invalid.",
                )
            )

        checks.append(self._check_forbidden_baseline_calls(lines))
        checks.append(self._check_non_trivial_compute(lines))
        checks.append(self._check_tl_dot_fp32(lines))
        checks.append(self._check_atomic_add(lines))
        checks.append(self._check_n_dimension_load_bounds(lines))
        checks.append(self._check_sorted_token_indexing_consistency(lines))
        checks.append(self._check_shared_memory(lines))
        checks.append(self._check_type_promotion(lines))

        return ValidationReport(ok=all(c.ok for c in checks), checks=checks)

    @staticmethod
    def _find_function(tree: ast.AST, fn_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
                return node
        return None

    @staticmethod
    def _format_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        parts: list[str] = []
        posonly = [arg.arg for arg in fn.args.posonlyargs]
        regular = [arg.arg for arg in fn.args.args]
        kwonly = [arg.arg for arg in fn.args.kwonlyargs]

        if posonly:
            parts.extend(posonly)
            parts.append("/")
        parts.extend(regular)
        if fn.args.vararg is not None:
            parts.append(f"*{fn.args.vararg.arg}")
        elif kwonly:
            parts.append("*")
        parts.extend(kwonly)
        if fn.args.kwarg is not None:
            parts.append(f"**{fn.args.kwarg.arg}")
        return f"{fn.name}({', '.join(parts)})"

    @staticmethod
    def _signature_shape(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[Any, ...]:
        return (
            tuple(arg.arg for arg in fn.args.posonlyargs),
            tuple(arg.arg for arg in fn.args.args),
            fn.args.vararg.arg if fn.args.vararg is not None else None,
            tuple(arg.arg for arg in fn.args.kwonlyargs),
            fn.args.kwarg.arg if fn.args.kwarg is not None else None,
            len(fn.args.defaults),
            sum(1 for default in fn.args.kw_defaults if default is not None),
        )

    def _check_kernel_signature_stability(
        self,
        tree: ast.AST,
        baseline_source: str | None,
    ) -> ValidationCheck:
        if not baseline_source:
            return ValidationCheck(name="kernel_signature_stability", ok=True)

        try:
            baseline_tree = ast.parse(baseline_source)
        except SyntaxError:
            return ValidationCheck(name="kernel_signature_stability", ok=True)

        current_fn = self._find_function(tree, "kernel")
        baseline_fn = self._find_function(baseline_tree, "kernel")
        if current_fn is None or baseline_fn is None:
            return ValidationCheck(name="kernel_signature_stability", ok=True)

        if self._signature_shape(current_fn) == self._signature_shape(baseline_fn):
            return ValidationCheck(name="kernel_signature_stability", ok=True)

        return ValidationCheck(
            name="kernel_signature_stability",
            ok=False,
            message=(
                "Do not change the entrypoint signature while optimizing the kernel. "
                f"Expected `{self._format_signature(baseline_fn)}`, "
                f"got `{self._format_signature(current_fn)}`."
            ),
            line=getattr(current_fn, "lineno", None),
        )

    def _check_seed_stage_scope(
        self,
        source: str,
        tree: ast.AST,
        baseline_source: str | None,
    ) -> ValidationCheck:
        if not baseline_source or "def _gemm1_subpath(" not in baseline_source:
            return ValidationCheck(name="seed_stage_scope", ok=True)

        try:
            baseline_tree = ast.parse(baseline_source)
        except SyntaxError:
            return ValidationCheck(name="seed_stage_scope", ok=True)

        baseline_jit = [
            node.name
            for node in ast.iter_child_nodes(baseline_tree)
            if isinstance(node, ast.FunctionDef) and self._is_triton_jit(node)
        ]
        current_jit = [
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.FunctionDef) and self._is_triton_jit(node)
        ]

        if baseline_jit or not current_jit:
            return ValidationCheck(name="seed_stage_scope", ok=True)

        if current_jit != ["_gemm1_tile_kernel"]:
            return ValidationCheck(
                name="seed_stage_scope",
                ok=False,
                message=(
                    "Seed-stage Tritonization may introduce only one Triton micro-kernel "
                    "named `_gemm1_tile_kernel`."
                ),
            )

        required_fragments = (
            "def _gemm1_subpath(",
            "G1 = _gemm1_subpath(A_e, W13_e)",
            "C = torch.nn.functional.silu(X2) * X1",
            "O = C.matmul(W2_e.t())",
            "accum.index_add_(",
            "_gemm1_tile_kernel[",
        )
        missing = [frag for frag in required_fragments if frag not in source]
        if missing:
            return ValidationCheck(
                name="seed_stage_scope",
                ok=False,
                message=(
                    "Seed-stage Tritonization must keep routing/SwiGLU/GEMM2/index_add_ "
                    "in PyTorch and only replace `_gemm1_subpath(...)`. "
                    f"Missing required fragment: `{missing[0]}`."
                ),
            )

        return ValidationCheck(name="seed_stage_scope", ok=True)

    def _check_seed_stage_numerics(
        self,
        source: str,
        tree: ast.AST,
        baseline_source: str | None,
    ) -> ValidationCheck:
        if not baseline_source or "def _gemm1_subpath(" not in baseline_source:
            return ValidationCheck(name="seed_stage_numerics", ok=True)

        try:
            baseline_tree = ast.parse(baseline_source)
        except SyntaxError:
            return ValidationCheck(name="seed_stage_numerics", ok=True)

        baseline_jit = [
            node.name
            for node in ast.iter_child_nodes(baseline_tree)
            if isinstance(node, ast.FunctionDef) and self._is_triton_jit(node)
        ]
        current_jit = [
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.FunctionDef) and self._is_triton_jit(node)
        ]
        if baseline_jit or "_gemm1_tile_kernel" not in current_jit:
            return ValidationCheck(name="seed_stage_numerics", ok=True)

        kernel_fn = self._find_function(tree, "_gemm1_tile_kernel")
        if kernel_fn is None:
            return ValidationCheck(name="seed_stage_numerics", ok=True)

        kernel_source = ast.get_source_segment(source, kernel_fn) or ""
        if ".to(tl.bfloat16)" in kernel_source or ".to(tl.float16)" in kernel_source:
            return ValidationCheck(
                name="seed_stage_numerics",
                ok=False,
                message=(
                    "Seed-stage GEMM1 already receives dequantized fp32 tensors. "
                    "Do not downcast `_gemm1_tile_kernel` operands to bf16/fp16; "
                    "keep loads in fp32 and use `tl.dot(..., input_precision=\"ieee\")` "
                    "or equivalent fp32 accumulation."
                ),
                line=getattr(kernel_fn, "lineno", None),
            )

        if (
            "tl.dot(" in kernel_source
            and 'input_precision="ieee"' not in kernel_source
            and "input_precision='ieee'" not in kernel_source
        ):
            return ValidationCheck(
                name="seed_stage_numerics",
                ok=False,
                message=(
                    "Seed-stage GEMM1 uses already-dequantized fp32 tensors. "
                    "Request IEEE fp32 dot precision with "
                    "`tl.dot(..., input_precision=\"ieee\")`, or use an equivalent "
                    "explicit fp32 accumulation path."
                ),
                line=getattr(kernel_fn, "lineno", None),
            )

        return ValidationCheck(name="seed_stage_numerics", ok=True)

    def _check_grouped_gemm_template_alignment(
        self,
        source: str,
        baseline_source: str | None,
    ) -> ValidationCheck:
        if not baseline_source:
            return ValidationCheck(name="grouped_gemm_template_alignment", ok=True)

        baseline_is_seed = (
            "def _gemm1_subpath(" in baseline_source
            and "@triton.jit" in baseline_source
            and "for le in range(E_LOCAL)" in baseline_source
        )
        if not baseline_is_seed:
            return ValidationCheck(name="grouped_gemm_template_alignment", ok=True)

        grouped_markers = (
            "def _sort_tokens(",
            "_grouped_gemm1_swiglu_kernel",
            "_grouped_gemm2_kernel",
            "tile_starts",
            "sorted_ids",
        )
        legacy_grouped_markers = (
            "def gemm1_kernel(",
            "def gemm2_kernel(",
            "expert_ids_per_block",
            "token_expert_pairs = []",
        )
        attempts_grouped = (
            any(marker in source for marker in grouped_markers)
            or any(marker in source for marker in legacy_grouped_markers)
            or "for le in range(E_LOCAL)" not in source
        )
        if not attempts_grouped:
            return ValidationCheck(name="grouped_gemm_template_alignment", ok=True)

        required_fragments = (
            "NUM_SMS = 148",
            "def _sort_tokens(",
            "sorted_ids",
            "expert_starts",
            "counts",
            "tile_starts",
            "_grouped_gemm1_swiglu_kernel",
            "_grouped_gemm2_kernel",
            "tl.float8e4nv",
        )
        for fragment in required_fragments:
            if fragment not in source:
                return ValidationCheck(
                    name="grouped_gemm_template_alignment",
                    ok=False,
                    message=(
                        "Grouped FP8 attempts must stay close to the vetted template. "
                        f"Missing required fragment: `{fragment}`."
                    ),
                    line=self._line_for_fragment(source, fragment),
                )

        persistent_launch_markers = (
            "_grouped_gemm1_swiglu_kernel[(NUM_SMS,)]",
            "_grouped_gemm2_kernel[(NUM_SMS,)]",
            "grid = (NUM_SMS,)",
            "grid=(NUM_SMS,)",
        )
        if not any(marker in source for marker in persistent_launch_markers):
            return ValidationCheck(
                name="grouped_gemm_template_alignment",
                ok=False,
                message=(
                    "Grouped FP8 attempts must use the persistent launch shape from the "
                    "template: `grid=(NUM_SMS,)` with NUM_SMS=148."
                ),
            )

        if not re.search(r"index_add_\(\s*0\s*,\s*sorted_ids\b", source):
            return ValidationCheck(
                name="grouped_gemm_template_alignment",
                ok=False,
                message=(
                    "Grouped FP8 attempts must scatter grouped outputs back with "
                    "`accum.index_add_(0, sorted_ids, ...)`."
                ),
                line=self._line_for_fragment(source, "index_add_("),
            )

        forbidden_fragments = {
            "def gemm1_kernel(": (
                "Do not fall back to the legacy sorted-token kernels `gemm1_kernel`/`gemm2_kernel`; "
                "use `_grouped_gemm1_swiglu_kernel` and `_grouped_gemm2_kernel` exactly."
            ),
            "def gemm2_kernel(": (
                "Do not fall back to the legacy sorted-token kernels `gemm1_kernel`/`gemm2_kernel`; "
                "use `_grouped_gemm1_swiglu_kernel` and `_grouped_gemm2_kernel` exactly."
            ),
            "token_expert_pairs = []": (
                "Avoid rebuilding token-expert lists in Python. Use `_sort_tokens(...)` from the "
                "grouped template."
            ),
            "expert_ids_per_block": (
                "Avoid legacy `expert_ids_per_block` dispatch. Use `expert_starts`, `counts`, and "
                "`tile_starts` for persistent grouped GEMM scheduling."
            ),
            "W13_fp32 = gemm1_weights.to(torch.float32)": (
                "Grouped FP8 attempts must not pre-dequantize all GEMM1 weights upfront."
            ),
            "W2_fp32 = gemm2_weights.to(torch.float32)": (
                "Grouped FP8 attempts must not pre-dequantize all GEMM2 weights upfront."
            ),
            "for le in range(E_LOCAL)": (
                "Grouped FP8 attempts must replace the per-expert Python loop with grouped dispatch."
            ),
        }
        for fragment, message in forbidden_fragments.items():
            if fragment in source:
                return ValidationCheck(
                    name="grouped_gemm_template_alignment",
                    ok=False,
                    message=message,
                    line=self._line_for_fragment(source, fragment),
                )

        return ValidationCheck(name="grouped_gemm_template_alignment", ok=True)

    @staticmethod
    def _line_for_fragment(source: str, fragment: str) -> int | None:
        for idx, line in enumerate(source.splitlines(), start=1):
            if fragment in line:
                return idx
        return None

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
    def _check_kernel_reachability(tree: ast.AST) -> ValidationCheck:
        jit_funcs: set[str] = set()
        top_level_fns: dict[str, ast.FunctionDef] = {}
        kernel_fn: ast.FunctionDef | None = None

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            top_level_fns[node.name] = node
            if node.name == "kernel":
                kernel_fn = node
            if ValidationAgent._is_triton_jit(node):
                jit_funcs.add(node.name)

        if not jit_funcs:
            if kernel_fn is not None and ValidationAgent._has_nontrivial_torch_compute(kernel_fn):
                return ValidationCheck(
                    name="kernel_reachability",
                    ok=True,
                    message="No Triton launch found, but a non-trivial PyTorch fallback path is present.",
                    line=getattr(kernel_fn, "lineno", None),
                    severity="warning",
                )
            return ValidationCheck(
                name="kernel_reachability",
                ok=False,
                message="No @triton.jit kernel function detected.",
            )
        if kernel_fn is None:
            return ValidationCheck(
                name="kernel_reachability",
                ok=False,
                message="Missing entrypoint `def kernel(...)`.",
            )

        def _fn_has_launch(fn: ast.FunctionDef, visited: set[str]) -> bool:
            if fn.name in visited:
                return False
            visited.add(fn.name)

            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                call_fn = node.func
                if isinstance(call_fn, ast.Subscript) and isinstance(call_fn.value, ast.Name):
                    if call_fn.value.id in jit_funcs:
                        return True
                if isinstance(call_fn, ast.Name):
                    callee = top_level_fns.get(call_fn.id)
                    if callee is not None and _fn_has_launch(callee, visited):
                        return True
            return False

        if _fn_has_launch(kernel_fn, set()):
            return ValidationCheck(
                name="kernel_reachability",
                ok=True,
            )

        return ValidationCheck(
            name="kernel_reachability",
            ok=False,
            message=(
                "No reachable Triton launch found from `kernel(...)`. "
                "Expected launch pattern like `my_kernel[grid](...)`."
            ),
            line=getattr(kernel_fn, "lineno", None),
        )

    @staticmethod
    def _has_nontrivial_torch_compute(kernel_fn: ast.FunctionDef) -> bool:
        heavy_ops = {"matmul", "mm", "bmm", "index_add_", "index_copy_", "scatter_add_"}
        useful_ops = {
            "topk",
            "sigmoid",
            "repeat_interleave",
            "masked_fill",
            "scatter_",
            "permute",
            "reshape",
            "view",
            "expand",
            "silu",
        }

        seen_useful_ops: set[str] = set()
        has_loop = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(kernel_fn))
        for node in ast.walk(kernel_fn):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute):
                if fn.attr in heavy_ops:
                    return True
                if fn.attr in useful_ops:
                    seen_useful_ops.add(fn.attr)
            elif isinstance(fn, ast.Name) and fn.id in useful_ops:
                seen_useful_ops.add(fn.id)

        return has_loop and len(seen_useful_ops) >= 2

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

    @staticmethod
    def _check_forbidden_baseline_calls(lines: list[str]) -> ValidationCheck:
        forbidden_patterns = [
            ("flashinfer_bench", "Calling FlashInfer-Bench internals from candidate kernel is disallowed."),
            ("build_baseline(", "Direct baseline execution path is disallowed."),
            ("baseline(", "Baseline helper invocation detected."),
            ("torch.ops.flashinfer", "Direct FlashInfer operator invocation bypasses candidate kernel path."),
            ("flashinfer.", "Direct FlashInfer runtime call detected; avoid baseline bypass."),
        ]
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for pattern, message in forbidden_patterns:
                if pattern in stripped:
                    return ValidationCheck(
                        name="forbidden_baseline_calls",
                        ok=False,
                        message=message,
                        line=idx,
                    )
        return ValidationCheck(name="forbidden_baseline_calls", ok=True)

    @staticmethod
    def _check_non_trivial_compute(lines: list[str]) -> ValidationCheck:
        suspicious_returns = [
            "return torch.zeros",
            "return torch.zeros_like",
            "return hidden_states",
            "return x",
            "return input",
        ]
        has_triton_launch = any(
            ("[" in line and "](" in line and "_kernel" in line)
            for line in lines
        )
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip().replace(" ", "")
            for marker in suspicious_returns:
                marker_cmp = marker.replace(" ", "")
                if marker_cmp in stripped and not has_triton_launch:
                    return ValidationCheck(
                        name="non_trivial_compute",
                        ok=False,
                        message=(
                            "Detected trivial output path without a reachable kernel launch. "
                            "Candidate must execute non-trivial compute."
                        ),
                        line=idx,
                    )
        return ValidationCheck(name="non_trivial_compute", ok=True)

    def _check_tl_dot_fp32(self, lines: list[str]) -> ValidationCheck:
        fp32_vars: set[str] = set()
        fp8_vars: set[str] = set()  # float8e4nv — native FP8 on B200, valid in tl.dot
        _FP8_CAST_RE = re.compile(r"(\w+)\s*=\s*.*\.to\(tl\.float8e4nv\)")
        for line in lines:
            stripped = line.strip()
            match = self._FP32_CAST_RE.match(stripped)
            if match:
                fp32_vars.add(match.group(1))
            fp8_match = _FP8_CAST_RE.match(stripped)
            if fp8_match:
                fp8_vars.add(fp8_match.group(1))

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
                    # fp32 vars are wrong; fp8 vars (float8e4nv) are valid on B200
                    if arg in fp32_vars and arg not in fp8_vars:
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
                    message=(
                        "Avoid tl.atomic_add for fp32 accumulation. "
                        "Instead write GEMM2 results to a flat buffer [N_pairs, H] indexed by "
                        "sorted-pair position (no conflicts), then accumulate with PyTorch "
                        "`accum.index_add_(0, sorted_tok_ids, flat_out * rw[:, None])`."
                    ),
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

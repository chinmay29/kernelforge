"""LLM-based mutation proposer using Claude API.

Uses Anthropic's Claude API to generate Triton kernel mutations
guided by RAG-retrieved knowledge (hardware constraints, patterns,
error fixes, research insights, and a reference Triton kernel).
"""

from __future__ import annotations

import difflib
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


# --------------------------------------------------------------------------
# vLLM reference Triton kernel (stripped to FP8 block-scale essentials)
# --------------------------------------------------------------------------

_REFERENCE_TRITON_KERNEL = '''\
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N, K, EM, num_valid_tokens,
    # Strides
    stride_am, stride_ak, stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bse, stride_bsk, stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr, group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr, compute_type: tl.constexpr,
):
    """vLLM-style fused MoE GEMM kernel for FP8 block-scale quantization."""
    # Map program ids to block of C (grouped ordering for L2 reuse)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Load sorted token ids for this block
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    # Get expert for this block
    off_experts = tl.load(expert_ids_ptr + pid_m)

    # Pointer setup
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am
                      + offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr + off_experts * stride_be
              + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Block-scale setup
    a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
    offs_bsn = offs_bn // group_n
    b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse
                    + offs_bsn * stride_bsn)

    # Main compute loop — accumulate in FP32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                     mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                     other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Block-scale dequant: load scales per K-block
        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                          mask=token_mask, other=0.0)
        b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

        # CRITICAL: Dequantize (convert AND scale) BEFORE tl.dot
        # Triton tl.dot does NOT support FP8 dtypes
        a_fp16 = a.to(tl.float16) * a_scale[:, None]
        b_fp16 = b.to(tl.float16) * b_scale[None, :]
        accumulator += tl.dot(a_fp16, b_fp16)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Router weight multiplication
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # Store output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
'''

# --------------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in Triton and CUDA kernels \
for NVIDIA B200 (Blackwell, SM100, compute capability 10.0) GPUs.

You are optimizing a Fused MoE (Mixture of Experts) kernel for the MLSys 2026 \
KernelForge competition hosted by FlashInfer.

## Problem Definition
The kernel implements DeepSeek-V3/R1 FP8 block-scale MoE:
- **Input**: T tokens × 256 experts (32 local), hidden_size=7168, intermediate=2048
- **Routing**: sigmoid + grouped top-k (8 groups, pick top-4 groups, top-8 experts total)
- **Computation**: routing → dequant → GEMM1(H→2I) → SwiGLU → GEMM2(I→H) → weighted accumulation
- **Quantization**: FP8 (e4m3fn) with per-block (128) scale factors
- **Output**: [T, 7168] bfloat16 tensor

## Scoring Criteria
- Correctness: max_abs_error must be below threshold (vs reference PyTorch implementation)
- Performance: speedup = reference_latency / your_latency (must be > 1.0 to score)
- Score = -1e9 if ANY workload fails correctness; otherwise proportional to median speedup
- 19 workloads with varying batch sizes tested on B200 GPU

## Rules
1. The function MUST be named `kernel` with the EXACT same signature as shown
2. Import triton, triton.language as tl, and torch at the top level
3. You can use torch ops, @triton.jit kernels, or a mix — whatever is fastest
4. Output MUST be [T, H] bfloat16 tensor, numerically close to reference
5. Focus on ONE clear optimization per iteration — don't rewrite everything
6. Keep the code self-contained — no external deps beyond torch and triton
7. The kernel runs on B200 GPU with CUDA — do NOT worry about macOS compatibility
8. ALWAYS start by ensuring the code is correct before making it fast
9. CRITICAL: @triton.jit functions CANNOT access Python globals (like BLOCK=128). \
Pass all values as tl.constexpr parameters or inline the number directly.
10. SHARED MEMORY LIMIT: B200 has 227KB max per threadblock. Your tile sizes MUST satisfy: \
2 * BLOCK_M * BLOCK_K * element_bytes * NUM_STAGES <= 227KB. \
Safe defaults: BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, NUM_STAGES=3, num_warps=4. \
NEVER use BLOCK_M=128, BLOCK_N=128, BLOCK_K=128 together — that exceeds shared memory.
11. CRITICAL: Triton tl.dot() does NOT support FP8 dtypes (fp8e4m3, fp8e4nv). For FP8 block-scale quantization, \
you MUST dequantize (convert type AND multiply by scale) BEFORE tl.dot(). \
WARNING: Type promotion! `bf16 * fp32 → fp32`, `fp16 * fp32 → fp32`. If scales are fp32, you MUST: \
CORRECT: `w_bf16 = (w.to(tl.bfloat16) * w_scale[None, :]).to(tl.bfloat16); accumulator += tl.dot(a, w_bf16)` \
OR: `w_scale_bf16 = w_scale.to(tl.bfloat16); w_bf16 = w.to(tl.bfloat16) * w_scale_bf16[None, :]; accumulator += tl.dot(a, w_bf16)` \
WRONG: `w_bf16 = w.to(tl.bfloat16) * w_scale[None, :]` (produces fp32, not bf16!). \
WRONG: `tl.dot(a, w) * scale` (FP8 not supported in tl.dot). WRONG: `w.to(tl.float16)` without scale (incorrect dequant).
12. tl.atomic_add with float32 is NOT reliably supported. For MoE weighted accumulation, \
use PyTorch `index_add_` after the Triton kernel instead of atomic_add inside it.

## Return Format
Return ONLY a JSON object (no markdown fences) with:
- "reasoning": what you changed and why (1-3 sentences)
- "mutation_name": short identifier (e.g., "fuse_routing_gemm", "triton_gemm1")
- "source": the COMPLETE new kernel.py source code (every line, NOT a diff)
"""

_USER_TEMPLATE = """\
## Current Kernel Source ({num_lines} lines)
```python
{current_source}
```

## Reference Triton MoE Kernel (from vLLM)
This is a production-quality Triton fused MoE GEMM kernel used in vLLM. \
Study its patterns: sorted token dispatch, L2-grouped block ordering, \
FP8 block-scale dequant, and router weight fusion.
```python
{reference_kernel}
```

## NVIDIA B200 Hardware Constraints
{hardware_context}

## Relevant Optimization Patterns & Knowledge
{patterns_context}

## Tuning Knob Definitions
{knobs_context}

## Research Insights (Blackwell Architecture)
{research_context}

## Previous Iteration Results
{history_context}

## Previous Diffs (what was tried)
{diff_context}

## Instructions
{instruction}

Return ONLY a JSON object with "reasoning", "mutation_name", and "source" fields.
"""

# --------------------------------------------------------------------------
# Context formatters
# --------------------------------------------------------------------------

def _format_hardware(constraints: list[dict[str, Any]]) -> str:
    if not constraints:
        return "No hardware constraints available."
    lines = []
    for c in constraints:
        val = c.get("value", "?")
        constraint = c.get("constraint", "")
        lines.append(
            f"- **{c.get('parameter', '?')}**: {val} {c.get('unit', '')} "
            f"— {c.get('text', '')[:120]}"
            + (f"\n  Constraint: `{constraint}`" if constraint else "")
        )
    return "\n".join(lines)


def _format_patterns(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "No patterns available."
    lines = []
    for p in entries:
        cat = p.get("category", "?")
        lines.append(f"### [{cat}] {p.get('id', '?')}")
        for field_name in ("when", "pattern", "text", "error", "cause", "fix", "relevance"):
            val = p.get(field_name)
            if val:
                lines.append(f"**{field_name.title()}:** {val[:250]}")
        hint = p.get("code_hint")
        if hint:
            lines.append(f"**Code:** `{hint[:200]}`")
        affected = p.get("knobs_affected")
        if affected:
            lines.append(f"**Knobs affected:** {', '.join(affected)}")
        lines.append("")
    return "\n".join(lines)


def _format_knobs(knobs: list[dict[str, Any]]) -> str:
    if not knobs:
        return "No knob definitions available."
    lines = []
    for k in knobs:
        impact = k.get("impact", "?")
        constraints = k.get("constraints", "")
        lines.append(
            f"- **{k.get('knob', '?')}** [{impact} impact]: "
            f"values={k.get('values', '?')} — {k.get('text', '')[:100]}"
            + (f"\n  `{constraints[:120]}`" if constraints else "")
        )
    return "\n".join(lines)


def _format_research(insights: list[dict[str, Any]]) -> str:
    if not insights:
        return "No research insights available."
    lines = []
    for r in insights:
        lines.append(f"- **{r.get('id', '?')}** ({r.get('source', '?')}): {r.get('text', '')[:200]}")
        rel = r.get("relevance")
        if rel:
            lines.append(f"  → *Relevance:* {rel[:150]}")
    return "\n".join(lines)


def _format_history(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "No previous runs. This is the first iteration."
    lines = ["| Iter | Score | Speedup | Status | Mutation | Error |",
             "|------|-------|---------|--------|----------|-------|"]
    for r in runs[-8:]:
        summary = r.get("summary", {})
        speedup = summary.get("median_speedup", "?")
        lines.append(
            f"| {r.get('iter', '?')} | {r.get('score', '?')} | {speedup} | "
            f"{r.get('status_summary', '?')} | {', '.join(r.get('mutations', []))} | "
            f"{r.get('error', '')[:150]} |"
        )
    return "\n".join(lines)


def _format_diffs(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "No previous diffs."
    lines = []
    for r in runs[-3:]:
        diffs = r.get("diffs", [])
        mutations = r.get("mutations", [])
        reasoning = r.get("reasoning", "")
        score = r.get("score", "?")
        if diffs:
            for name, diff in zip(mutations, diffs):
                lines.append(f"### {name} (score={score})")
                if reasoning:
                    lines.append(f"Reasoning: {reasoning[:150]}")
                if diff:
                    lines.append(f"```diff\n{diff[:500]}\n```")
                lines.append("")
    return "\n".join(lines) if lines else "No diffs from previous iterations."


def _get_instruction(last_error: str, iteration: int, provenance_dir: Path | None = None) -> str:
    """Generate instruction for the LLM based on error status and iteration.

    Args:
        last_error: Summary error message (may be truncated)
        iteration: Current iteration number
        provenance_dir: Path to provenance directory to read detailed error logs
    """
    if last_error:
        # Try to read detailed error from benchmark_error.txt
        detailed_error = last_error
        if provenance_dir and iteration > 0:
            error_file = provenance_dir / f"iter_{iteration-1:03d}" / "benchmark_error.txt"
            if error_file.exists():
                try:
                    detailed_error = error_file.read_text()
                    # Truncate if extremely long (keep error type + message + location)
                    if len(detailed_error) > 2000:
                        lines = detailed_error.split("\n")
                        # Keep first 30 lines (usually has the key error info)
                        detailed_error = "\n".join(lines[:30])
                        if len(lines) > 30:
                            detailed_error += f"\n... ({len(lines)-30} more lines truncated)"
                except Exception:
                    # Fall back to summary error if file read fails
                    pass

        return (
            f"⚠️ The previous kernel FAILED with this error:\n"
            f"```\n{detailed_error}\n```\n\n"
            "Fix this error while keeping the optimization direction. "
            "Make the minimum change needed to fix the error. "
            "Study the error fix suggestions in the patterns section above."
        )
    if iteration == 0:
        return (
            "🎯 **FIRST ITERATION**: The current kernel uses pure PyTorch ops (torch.matmul, loops).\n\n"
            "Your first step should be to convert the expert GEMM loop into a Triton @triton.jit kernel.\n"
            "Study the vLLM reference kernel above for the pattern:\n"
            "1. Pre-sort tokens by expert (sorted_token_ids)\n"
            "2. Launch grid = (num_m_blocks * num_n_blocks,)\n"
            "3. Each program handles one (M_block, N_block) tile for one expert\n"
            "4. Use L2-grouped ordering (GROUP_SIZE_M)\n\n"
            "Keep routing in PyTorch for now. Focus on getting GEMM1 into Triton first.\n"
            "Make sure the output is CORRECT — a wrong kernel scores -1e9."
        )
    if iteration <= 3:
        return (
            "Continue optimizing. Focus on the SINGLE highest-impact change:\n"
            "1. If GEMM1 is in Triton but GEMM2 is not → convert GEMM2 to Triton\n"
            "2. If both GEMMs are in Triton → fuse SwiGLU activation into GEMM1 epilogue\n"
            "3. If all ops are in Triton → tune tile sizes (BLOCK_M, BLOCK_N, BLOCK_K)\n"
            "4. If tile sizes are tuned → add multi-stage pipelining (NUM_STAGES)\n\n"
            "Study the previous diffs above to avoid repeating failed approaches."
        )
    return (
        "Advanced optimization. Consider:\n"
        "1. Fuse GEMM1→SwiGLU→GEMM2 into fewer kernel launches\n"
        "2. Use tl.dot with acc=accumulator for FP8 fast accumulation\n"
        "3. Optimize memory access patterns (coalescing, TMA-friendly layouts)\n"
        "4. Tune num_warps and num_stages via @triton.autotune\n"
        "5. Implement split-K for small batch sizes\n"
        "6. Batch expert GEMMs into grouped GEMM pattern\n\n"
        "Pick ONE change. Study previous diffs to avoid repeating failures."
    )


# --------------------------------------------------------------------------
# Source validation
# --------------------------------------------------------------------------

def _validate_source(source: str) -> tuple[bool, str]:
    """Validate LLM-generated kernel source before benchmarking.
    
    Returns (is_valid, error_message).
    """
    # 1. Python syntax check
    try:
        compile(source, "kernel.py", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    # 2. Required function exists
    if "def kernel(" not in source:
        return False, "Missing required 'def kernel(' function"

    # 3. Required imports
    if "import torch" not in source:
        return False, "Missing 'import torch'"

    # 4. Return type check (must return something)
    lines = source.splitlines()
    in_kernel = False
    has_return = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def kernel("):
            in_kernel = True
        elif in_kernel and stripped.startswith("def ") and not stripped.startswith("def kernel"):
            # Nested function definitions are OK, but new top-level defs end kernel scope
            if not line.startswith(" ") and not line.startswith("\t"):
                in_kernel = False
        elif in_kernel and stripped.startswith("return "):
            has_return = True

    if not has_return:
        return False, "'def kernel' has no return statement"

    # 5. Triton global variable access check
    # Detect Python globals used inside @triton.jit functions
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True, ""  # Already caught above

    # Collect module-level variable names (non-function, non-class, non-import)
    module_globals: set[str] = set()
    jit_functions: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    module_globals.add(target.id)
        elif isinstance(node, ast.FunctionDef):
            # Check for @triton.jit decorator
            for deco in node.decorator_list:
                deco_name = ""
                if isinstance(deco, ast.Attribute):
                    deco_name = ast.dump(deco)
                elif isinstance(deco, ast.Name):
                    deco_name = deco.id
                if "jit" in str(deco_name).lower():
                    jit_functions.append(node.name)

    # Check if jit functions reference module globals
    # Exclude known-safe names and constexpr-decorated params
    safe_names = {"tl", "triton", "torch", "math", "BLOCK_M", "BLOCK_N", "BLOCK_K",
                  "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
                  "NUM_STAGES", "NUM_WARPS", "True", "False", "None"}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name in jit_functions:
            # Get parameter names (these are allowed)
            param_names = {arg.arg for arg in node.args.args}
            # Walk the function body for Name references
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id in module_globals:
                    if child.id not in param_names and child.id not in safe_names:
                        return False, (
                            f"Triton @jit function '{node.name}' uses Python global "
                            f"'{child.id}'. Pass it as a tl.constexpr parameter or "
                            f"inline the value directly."
                        )

    # 6. Check for tl.dot with fp32 operands (common LLM mistake)
    # Track variables that were cast to fp32 (e.g., a_fp32 = x.to(tl.float32) * scale)
    import re as _re2
    fp32_vars: set[str] = set()
    for line in lines:
        stripped = line.strip()
        # Track: var = something.to(tl.float32) [* ...]  or  var = something.to(tl.float32)
        m = _re2.match(r'(\w+)\s*=\s*.*\.to\(tl\.float32\)', stripped)
        if m:
            fp32_vars.add(m.group(1))
        # Check: tl.dot(fp32_var, ...) or tl.dot(..., fp32_var)
        if "tl.dot(" in stripped:
            # Same-line cast
            if ".to(tl.float32)" in stripped:
                return False, (
                    "tl.dot() operands must be fp8/fp16/bf16, NOT fp32. "
                    "For FP8: dequantize BEFORE dot: `a_fp16 = a.to(tl.float16) * scale; accumulator += tl.dot(a_fp16, w_fp16)`. "
                    f"Line: {stripped[:100]}"
                )
            # Check if dot args are known fp32 variables
            dot_match = _re2.search(r'tl\.dot\(([^,]+),\s*([^)]+)\)', stripped)
            if dot_match:
                arg1 = dot_match.group(1).strip()
                arg2 = dot_match.group(2).strip()
                for arg in (arg1, arg2):
                    if arg in fp32_vars:
                        return False, (
                            f"tl.dot() operand '{arg}' was cast to fp32. "
                            "tl.dot operands must be fp8/fp16/bf16. "
                            "For FP8: dequantize BEFORE dot: `a_fp16 = a.to(tl.float16) * scale; accumulator += tl.dot(a_fp16, w_fp16)`."
                        )

    # 7. Check for tl.atomic_add with fp32 (not reliably supported)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "tl.atomic_add" in stripped:
            return False, (
                "tl.atomic_add with float32 is not reliably supported. "
                "Use tl.store per-expert then PyTorch index_add_ for accumulation. "
                f"Line {i + 1}: {stripped[:100]}"
            )

    # 8. Shared memory estimation for Triton kernels
    # Parse BLOCK sizes and NUM_STAGES to estimate shared memory usage
    import re as _re
    block_m = block_n = block_k = 128  # defaults
    num_stages = 3
    for line in lines:
        stripped = line.strip()
        # Look for BLOCK_SIZE_M = 128 or BLOCK_M = 128 patterns in kernel() body
        for pattern, var in [
            (r'BLOCK_SIZE_M\s*=\s*(\d+)', 'M'),
            (r'BLOCK_SIZE_N\s*=\s*(\d+)', 'N'),
            (r'BLOCK_SIZE_K\s*=\s*(\d+)', 'K'),
            (r'BLOCK_M\s*=\s*(\d+)', 'M'),
            (r'BLOCK_N\s*=\s*(\d+)', 'N'),
            (r'BLOCK_K\s*=\s*(\d+)', 'K'),
            (r'num_stages\s*=\s*(\d+)', 'S'),
            (r'NUM_STAGES\s*=\s*(\d+)', 'S'),
        ]:
            m = _re.search(pattern, stripped)
            if m:
                val = int(m.group(1))
                if var == 'M': block_m = val
                elif var == 'N': block_n = val
                elif var == 'K': block_k = val
                elif var == 'S': num_stages = val

    # Estimate: A tile + B tile, doubled for double-buffering, times stages
    # A tile: BLOCK_M * BLOCK_K, B tile: BLOCK_K * BLOCK_N
    # Conservative estimate with FP8 (1 byte) + FP32 accum overhead
    elem_bytes = 2  # conservative: bf16/fp16
    smem_bytes = (block_m * block_k + block_k * block_n) * elem_bytes * num_stages
    smem_kb = smem_bytes / 1024
    max_smem_kb = 227  # B200 limit

    if smem_kb > max_smem_kb:
        return False, (
            f"Estimated shared memory {smem_kb:.0f}KB exceeds B200 limit of {max_smem_kb}KB. "
            f"Current: BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}, "
            f"NUM_STAGES={num_stages}. Reduce tile sizes or NUM_STAGES."
        )

    # 9. Check for type promotion bugs in FP8 dequantization
    # Pattern: x_bf16 = x.to(tl.bfloat16) * scale  <- scale is fp32, result is fp32!
    # Pattern: x_fp16 = x.to(tl.float16) * scale   <- scale is fp32, result is fp32!
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for: var = something.to(tl.bfloat16) * something
        # or:       var = something.to(tl.float16) * something
        if ".to(tl.bfloat16)" in stripped and "*" in stripped and "=" in stripped:
            # Check if it's a simple assignment without explicit re-cast
            if ".to(tl.bfloat16)).to(tl.bfloat16)" not in stripped:
                # Likely bug: bf16 * fp32 scale = fp32 result
                # Extract variable name
                var_match = _re.match(r'(\w+)\s*=\s*.*\.to\(tl\.bfloat16\)\s*\*', stripped)
                if var_match and "bf16" in var_match.group(1).lower():
                    return False, (
                        f"TYPE PROMOTION BUG at line {i+1}: `{var_match.group(1)} = x.to(tl.bfloat16) * scale` "
                        f"produces fp32 because bf16*fp32→fp32. MUST cast result back: "
                        f"`{var_match.group(1)} = (x.to(tl.bfloat16) * scale[None, :]).to(tl.bfloat16)` "
                        f"OR convert scale first: `scale_bf16 = scale.to(tl.bfloat16); {var_match.group(1)} = x.to(tl.bfloat16) * scale_bf16[None, :]`. "
                        f"Line: {stripped[:100]}"
                    )
        if ".to(tl.float16)" in stripped and "*" in stripped and "=" in stripped:
            # Check if it's a simple assignment without explicit re-cast
            if ".to(tl.float16)).to(tl.float16)" not in stripped:
                var_match = _re.match(r'(\w+)\s*=\s*.*\.to\(tl\.float16\)\s*\*', stripped)
                if var_match and "fp16" in var_match.group(1).lower():
                    return False, (
                        f"TYPE PROMOTION BUG at line {i+1}: `{var_match.group(1)} = x.to(tl.float16) * scale` "
                        f"produces fp32 because fp16*fp32→fp32. MUST cast result back: "
                        f"`{var_match.group(1)} = (x.to(tl.float16) * scale[None, :]).to(tl.float16)` "
                        f"OR convert scale first: `scale_fp16 = scale.to(tl.float16); {var_match.group(1)} = x.to(tl.float16) * scale_fp16[None, :]`. "
                        f"Line: {stripped[:100]}"
                    )

    return True, ""


# --------------------------------------------------------------------------
# LLM caller
# --------------------------------------------------------------------------

def _call_claude(
    system: str,
    user: str,
    model: str = "claude-opus-4-20250514",
    max_tokens: int = 16384,
    temperature: float = 0.3,
) -> str:
    """Call the Anthropic Claude API via HTTP."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Run:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "Or add it to ~/.zshrc and run: source ~/.zshrc"
        )

    import requests

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=180,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Claude API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    text_parts = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block["text"])
    return "\n".join(text_parts)


def _parse_response(raw: str) -> dict[str, Any]:
    """Extract JSON from Claude's response, handling markdown fences."""
    # Try markdown JSON block first
    json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if json_match:
        raw_json = json_match.group(1)
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            pass

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find outermost braces
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from Claude response:\n{raw[:500]}")


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


# --------------------------------------------------------------------------
# Proposer
# --------------------------------------------------------------------------

class Proposer:
    """LLM-based kernel mutation proposer using Claude API."""

    def __init__(
        self,
        seed: int,
        model: str = "claude-sonnet-4-5-20250929",
        rag_store: Any | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.model = model
        self.rag_store = rag_store
        self._iteration = 0
        self._run_history: list[dict[str, Any]] = []
        self.log_dir = log_dir

    def _log_interaction(
        self,
        system: str,
        user: str,
        raw_response: str,
        parsed: dict[str, Any] | None,
        validation_ok: bool,
        validation_error: str,
    ) -> None:
        """Save full prompt/response to disk for debugging."""
        if not self.log_dir:
            return
        try:
            iter_dir = self.log_dir / f"iter_{self._iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            (iter_dir / "system_prompt.txt").write_text(system, encoding="utf-8")
            (iter_dir / "user_prompt.txt").write_text(user, encoding="utf-8")
            (iter_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")

            meta = {
                "iteration": self._iteration,
                "model": self.model,
                "validation_ok": validation_ok,
                "validation_error": validation_error,
                "mutation_name": parsed.get("mutation_name", "") if parsed else "",
                "reasoning": parsed.get("reasoning", "") if parsed else "",
                "source_lines": len(parsed.get("source", "").splitlines()) if parsed else 0,
            }
            (iter_dir / "meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

            if parsed and parsed.get("source"):
                (iter_dir / "generated_kernel.py").write_text(
                    parsed["source"], encoding="utf-8"
                )

            print(f"         📝 Logs saved to {iter_dir}")
        except Exception as e:
            print(f"         ⚠️  Failed to save logs: {e}")

    def propose(
        self,
        source: str,
        retrieved: list[dict[str, Any]] | None = None,
        max_mutations: int = 2,
        last_error: str = "",
    ) -> Proposal:
        """Generate a kernel mutation using Claude API with full RAG context."""
        retrieved = retrieved or []

        # ---- Build rich context from RAG store ----
        if self.rag_store is not None:
            hardware = _format_hardware(self.rag_store.get_constraints())
            knobs = _format_knobs(self.rag_store.get_knobs())
            # Always include research insights
            research = _format_research(
                self.rag_store.get_research(
                    "TMEM tensor core Blackwell MoE FP8 memory bandwidth", k=5
                )
            )
        else:
            hardware = _format_hardware(
                [r for r in retrieved if r.get("category") == "hardware"]
            )
            knobs = _format_knobs(
                [r for r in retrieved if r.get("category") == "knob"]
            )
            research = "No research insights available."

        patterns = _format_patterns(retrieved)
        history = _format_history(self._run_history)
        diffs = _format_diffs(self._run_history)
        instruction = _get_instruction(last_error, self._iteration, self.log_dir)

        user_msg = _USER_TEMPLATE.format(
            current_source=source,
            num_lines=len(source.splitlines()),
            reference_kernel=_REFERENCE_TRITON_KERNEL,
            hardware_context=hardware,
            patterns_context=patterns,
            knobs_context=knobs,
            research_context=research,
            history_context=history,
            diff_context=diffs,
            instruction=instruction,
        )

        # ---- Call Claude ----
        raw = _call_claude(
            system=_SYSTEM_PROMPT,
            user=user_msg,
            model=self.model,
            temperature=0.3 if not last_error else 0.1,
        )

        # ---- Parse response ----
        try:
            result = _parse_response(raw)
        except ValueError as e:
            self._log_interaction(_SYSTEM_PROMPT, user_msg, raw, None, False, str(e))
            self._iteration += 1
            return Proposal(
                mutations=[MutationResult(
                    name="llm_parse_error",
                    params={"error": str(e), "raw_response": raw[:500]},
                    source=source,
                    diff="",
                )],
                source=source,
                reasoning=f"Failed to parse LLM response: {e}",
            )

        new_source = result.get("source", source)
        reasoning = result.get("reasoning", "No reasoning provided")
        mutation_name = result.get("mutation_name", f"llm_iter_{self._iteration}")

        # ---- Validate generated source ----
        is_valid, validation_error = _validate_source(new_source)
        if not is_valid:
            print(f"  ⚠️  Validation failed: {validation_error}")
            print(f"  ↩️  Falling back to original source")
            self._log_interaction(_SYSTEM_PROMPT, user_msg, raw, result, False, validation_error)
            self._iteration += 1
            return Proposal(
                mutations=[MutationResult(
                    name=f"{mutation_name}_INVALID",
                    params={"validation_error": validation_error},
                    source=source,
                    diff="",
                )],
                source=source,
                reasoning=f"VALIDATION FAILED: {validation_error}. Original: {reasoning}",
            )

        diff = _make_diff(source, new_source)

        # Log the full interaction (success path)
        self._log_interaction(_SYSTEM_PROMPT, user_msg, raw, result, True, "")

        mutation = MutationResult(
            name=mutation_name,
            params={"model": self.model, "iteration": self._iteration},
            source=new_source,
            diff=diff,
        )

        # Track run history for next iteration's context
        self._run_history.append({
            "iter": self._iteration,
            "mutations": [mutation_name],
            "reasoning": reasoning,
            "diffs": [diff],
            "score": None,  # filled by engine after benchmark
        })

        self._iteration += 1
        return Proposal(
            mutations=[mutation],
            source=new_source,
            reasoning=reasoning,
        )

    def update_last_run(self, score: float, summary: dict[str, Any], error: str = "") -> None:
        """Update the last run entry with benchmark results (called by engine)."""
        if self._run_history:
            self._run_history[-1]["score"] = score
            self._run_history[-1]["summary"] = summary
            self._run_history[-1]["status_summary"] = summary.get("status_summary", "")
            self._run_history[-1]["error"] = error

"""Constrained mutation registry for Triton kernels."""

from __future__ import annotations

import difflib
import random
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class MutationResult:
    name: str
    params: dict[str, int | str]
    source: str
    diff: str


MutationFn = Callable[[str, random.Random], tuple[str, dict[str, int | str]]]


def _with_diff(name: str, old: str, new: str, params: dict[str, int | str]) -> MutationResult:
    diff = "\n".join(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile="before/kernel.py",
            tofile="after/kernel.py",
            lineterm="",
        )
    )
    return MutationResult(name=name, params=params, source=new, diff=diff)


def _ensure_import(source: str, line: str) -> str:
    if line in source:
        return source
    return source.replace("import triton\n", f"import triton\n{line}\n", 1)


def _mut_insert_meta_defaults(source: str, rng: random.Random) -> tuple[str, dict[str, int | str]]:
    num_warps = rng.choice([2, 4, 8])
    num_stages = rng.choice([2, 3, 4])
    insert = f"DEFAULT_NUM_WARPS = {num_warps}\nDEFAULT_NUM_STAGES = {num_stages}\n"
    if "DEFAULT_NUM_WARPS" in source:
        source = re.sub(r"DEFAULT_NUM_WARPS\s*=\s*\d+", f"DEFAULT_NUM_WARPS = {num_warps}", source)
        source = re.sub(r"DEFAULT_NUM_STAGES\s*=\s*\d+", f"DEFAULT_NUM_STAGES = {num_stages}", source)
    else:
        source = source.replace("import triton.language as tl\n", f"import triton.language as tl\n\n{insert}", 1)
    return source, {"num_warps": num_warps, "num_stages": num_stages}


def _mut_insert_block_m(source: str, rng: random.Random) -> tuple[str, dict[str, int | str]]:
    block_m = rng.choice([32, 64, 128, 256])
    if "BLOCK_M" in source:
        source = re.sub(r"BLOCK_M\s*=\s*\d+", f"BLOCK_M = {block_m}", source)
    else:
        source = source.replace("import triton.language as tl\n", f"import triton.language as tl\n\nBLOCK_M = {block_m}\n", 1)
    return source, {"block_m": block_m}


def _mut_add_grid_helper(source: str, rng: random.Random) -> tuple[str, dict[str, int | str]]:
    if "def launch_grid" in source:
        return source, {"noop": "launch_grid_exists"}
    helper = (
        "\n\ndef launch_grid(meta):\n"
        "    # Single-dimension grid helper; concrete kernels can override this.\n"
        "    return (triton.cdiv(meta.get('M', 1), meta.get('BLOCK_M', BLOCK_M if 'BLOCK_M' in globals() else 128)),)\n"
    )
    source = source.rstrip() + helper + "\n"
    return source, {"helper": "launch_grid"}


def _mut_replace_pass_with_skeleton(source: str, rng: random.Random) -> tuple[str, dict[str, int | str]]:
    if "pass" not in source:
        return source, {"noop": "no_pass_found"}
    body = (
        "    pid = tl.program_id(axis=0)\n"
        "    _ = pid  # Keep structure valid until signature-specific logic is added.\n"
        "    return\n"
    )
    source = source.replace("    pass\n", body, 1)
    return source, {"action": "replace_pass"}


def _mut_add_heuristics_decorator(source: str, rng: random.Random) -> tuple[str, dict[str, int | str]]:
    source = _ensure_import(source, "from triton import heuristics")
    if "@heuristics(" in source:
        return source, {"noop": "decorator_exists"}
    marker = "@triton.jit\n"
    if marker not in source:
        return source, {"noop": "jit_marker_missing"}
    deco = (
        "@heuristics(\n"
        "    {\n"
        "        'BLOCK_M': lambda args: 128,\n"
        "    }\n"
        ")\n"
    )
    source = source.replace(marker, deco + marker, 1)
    return source, {"action": "add_heuristics"}


MUTATION_REGISTRY: dict[str, MutationFn] = {
    "insert_meta_defaults": _mut_insert_meta_defaults,
    "insert_block_m": _mut_insert_block_m,
    "add_grid_helper": _mut_add_grid_helper,
    "replace_pass_with_skeleton": _mut_replace_pass_with_skeleton,
    "add_heuristics_decorator": _mut_add_heuristics_decorator,
}


def apply_mutation(name: str, source: str, rng: random.Random) -> MutationResult:
    if name not in MUTATION_REGISTRY:
        raise KeyError(f"Unknown mutation: {name}")
    new_source, params = MUTATION_REGISTRY[name](source, rng)
    return _with_diff(name, source, new_source, params)


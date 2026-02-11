"""Lightweight local retrieval for mutation guidance."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from kernelforge.io_utils import append_jsonl, read_jsonl


TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _tf(tokens: list[str]) -> Counter[str]:
    return Counter(tokens)


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class RagStore:
    def __init__(self, patterns_path: Path, runs_path: Path) -> None:
        self.patterns_path = patterns_path
        self.runs_path = runs_path

    def bootstrap_patterns(self) -> None:
        if self.patterns_path.exists():
            return
        seeds = [
            {
                "id": "triton_meta_tuning",
                "text": "Tune num_warps and num_stages conservatively for fused_moe kernels.",
                "mutations": ["insert_meta_defaults"],
            },
            {
                "id": "block_sweep",
                "text": "Sweep BLOCK_M values 64 128 256 and pick stable fast options.",
                "mutations": ["insert_block_m"],
            },
            {
                "id": "grid_clarity",
                "text": "Use explicit launch_grid helper to isolate launch geometry changes.",
                "mutations": ["add_grid_helper"],
            },
            {
                "id": "jit_body_skeleton",
                "text": "Replace pass statements with minimal valid Triton body to avoid empty kernels.",
                "mutations": ["replace_pass_with_skeleton"],
            },
        ]
        for row in seeds:
            append_jsonl(self.patterns_path, row)

    def retrieve(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        patterns = read_jsonl(self.patterns_path)
        runs = read_jsonl(self.runs_path)
        docs = patterns + [
            {
                "id": f"run::{i}",
                "text": f"{r.get('definition', '')} {r.get('status_summary', '')} "
                f"{' '.join(r.get('mutations', []))} {r.get('error', '')}",
                "mutations": r.get("mutations", []),
            }
            for i, r in enumerate(runs[-200:])
        ]
        qtf = _tf(_tokens(query))
        scored = []
        for row in docs:
            rtf = _tf(_tokens(row.get("text", "")))
            score = _cosine(qtf, rtf)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:k]]


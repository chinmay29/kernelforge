"""Lightweight local retrieval for mutation guidance.

Loads all JSONL files from the rag_store/ directory and supports:
- TF-IDF cosine similarity search
- Tag-based filtering (intersection boost)
- Category filtering (hardware, knob, pattern, error_fix, research)
- Dedicated accessors for constraints and knob definitions
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from kernelforge.io_utils import append_jsonl, read_jsonl


TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]+")

# Files that should always be loaded (if they exist)
_STORE_FILES = [
    "hardware_constraints.jsonl",
    "knob_definitions.jsonl",
    "optimization_patterns.jsonl",
    "error_fixes.jsonl",
    "research_insights.jsonl",
]


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


def _tag_boost(query_tags: set[str], row_tags: list[str]) -> float:
    """Boost score by tag intersection ratio (0.0 to 0.5)."""
    if not query_tags or not row_tags:
        return 0.0
    row_set = {t.lower() for t in row_tags}
    overlap = len(query_tags & row_set)
    return 0.5 * overlap / max(len(query_tags), 1)


class RagStore:
    """Multi-file RAG store with tag-based filtering."""

    def __init__(self, store_dir: Path, runs_path: Path | None = None) -> None:
        self.store_dir = store_dir
        self.runs_path = runs_path or store_dir / "runs.jsonl"
        self._cache: dict[str, list[dict[str, Any]]] | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> dict[str, list[dict[str, Any]]]:
        """Load all JSONL files from the store directory."""
        if self._cache is not None:
            return self._cache
        result: dict[str, list[dict[str, Any]]] = {}
        for fname in _STORE_FILES:
            path = self.store_dir / fname
            if path.exists():
                key = fname.replace(".jsonl", "")
                result[key] = read_jsonl(path)
        # Also load runs
        if self.runs_path.exists():
            result["runs"] = read_jsonl(self.runs_path)
        self._cache = result
        return result

    def invalidate_cache(self) -> None:
        """Force re-read from disk on next access."""
        self._cache = None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 6,
        categories: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k entries matching the query.

        Args:
            query: Natural language query
            k: Number of results to return
            categories: Filter to specific categories (e.g. ["pattern", "knob"])
            tags: Boost entries with matching tags (e.g. ["moe", "fp8"])
        """
        all_data = self._load_all()
        query_tags = {t.lower() for t in (tags or [])}
        qtf = _tf(_tokens(query))

        docs: list[dict[str, Any]] = []
        for cat_name, entries in all_data.items():
            # Skip non-matching categories
            if categories and cat_name not in categories:
                continue
            # Transform run history into searchable docs
            if cat_name == "runs":
                for i, r in enumerate(entries[-200:]):
                    docs.append({
                        "id": f"run::{i}",
                        "category": "runs",
                        "tags": [],
                        "text": (
                            f"{r.get('definition', '')} "
                            f"{r.get('status_summary', '')} "
                            f"{' '.join(r.get('mutations', []))} "
                            f"{r.get('error', '')}"
                        ),
                    })
            else:
                docs.extend(entries)

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in docs:
            # Build searchable text from ALL relevant fields across schemas
            parts = [
                row.get("text", ""),
                row.get("pattern", ""),
                row.get("code_hint", ""),
                row.get("fix", ""),
                row.get("error", ""),       # error_fixes
                row.get("cause", ""),       # error_fixes
                row.get("when", ""),        # optimization_patterns
                row.get("constraint", ""),  # hardware_constraints
                row.get("parameter", ""),   # hardware_constraints
                row.get("relevance", ""),   # research_insights
                " ".join(row.get("knobs_affected", [])),
            ]
            text = " ".join(p for p in parts if p)

            rtf = _tf(_tokens(text))
            sim = _cosine(qtf, rtf)
            boost = _tag_boost(query_tags, row.get("tags", []))
            total = sim + boost
            if total > 0:
                scored.append((total, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:k]]

    # ------------------------------------------------------------------
    # Dedicated accessors
    # ------------------------------------------------------------------

    def get_constraints(self) -> list[dict[str, Any]]:
        """Return all hardware constraints (immutable facts)."""
        data = self._load_all()
        return data.get("hardware_constraints", [])

    def get_knobs(self) -> list[dict[str, Any]]:
        """Return all tuning knob definitions."""
        data = self._load_all()
        return data.get("knob_definitions", [])

    def get_error_fixes(self, error_text: str) -> list[dict[str, Any]]:
        """Find error fixes matching the given error text."""
        return self.retrieve(
            query=error_text,
            k=3,
            categories=["error_fixes"],
        )

    def get_patterns(
        self, query: str, tags: list[str] | None = None, k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve optimization patterns relevant to the query."""
        return self.retrieve(
            query=query,
            k=k,
            categories=["optimization_patterns"],
            tags=tags,
        )

    def get_research(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Retrieve research insights relevant to the query."""
        return self.retrieve(
            query=query,
            k=k,
            categories=["research_insights"],
        )

    # ------------------------------------------------------------------
    # Run history logging
    # ------------------------------------------------------------------

    def log_run(self, row: dict[str, Any]) -> None:
        """Append a run result to the runs log."""
        append_jsonl(self.runs_path, row)
        self.invalidate_cache()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, int]:
        """Return counts per category."""
        data = self._load_all()
        return {cat: len(entries) for cat, entries in data.items()}

"""Benchmark wrappers with structured results."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class BenchSettings:
    mode: str
    warmup_runs: int
    iterations: int
    num_trials: int
    workload_focus: str | None


def kernel_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def make_cache_key(source_hash: str, settings: BenchSettings) -> str:
    focus = settings.workload_focus or "all"
    payload = (
        f"{source_hash}|{settings.mode}|{settings.warmup_runs}|"
        f"{settings.iterations}|{settings.num_trials}|{focus}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _filter_workloads(results: dict[str, Any], workload_focus: str | None) -> dict[str, Any]:
    if not workload_focus:
        return results
    out: dict[str, Any] = {}
    for def_name, workloads in results.items():
        if workload_focus in workloads:
            out[def_name] = {workload_focus: workloads[workload_focus]}
        else:
            out[def_name] = {}
    return out


def _summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    statuses: list[str] = []
    speedups: list[float] = []
    latencies: list[float] = []
    errors: list[float] = []

    for workloads in results.values():
        for row in workloads.values():
            status = str(row.get("status", "unknown"))
            statuses.append(status)
            if row.get("speedup_factor") is not None:
                speedups.append(float(row["speedup_factor"]))
            if row.get("latency_ms") is not None:
                latencies.append(float(row["latency_ms"]))
            if row.get("max_abs_error") is not None:
                errors.append(float(row["max_abs_error"]))

    success_count = sum(1 for s in statuses if s.lower() == "success")
    total = len(statuses)
    score = -1e9
    if total > 0 and success_count == total:
        score = median(speedups) if speedups else -1e6

    return {
        "total_workloads": total,
        "success_count": success_count,
        "status_summary": f"{success_count}/{total} success",
        "median_speedup": median(speedups) if speedups else None,
        "median_latency_ms": median(latencies) if latencies else None,
        "max_abs_error": max(errors) if errors else None,
        "score": score,
    }


def run_benchmark(settings: BenchSettings) -> dict[str, Any]:
    from scripts.pack_solution import pack_solution

    solution_path = pack_solution()

    if settings.mode == "local":
        # Local mode: needs flashinfer_bench + triton + CUDA (Linux GPU only)
        from flashinfer_bench import BenchmarkConfig, Solution
        from scripts.run_local import run_benchmark as run_local_benchmark

        solution = Solution.model_validate_json(
            solution_path.read_text(encoding="utf-8")
        )
        config = BenchmarkConfig(
            warmup_runs=settings.warmup_runs,
            iterations=settings.iterations,
            num_trials=settings.num_trials,
        )
        raw_results = run_local_benchmark(solution, config)

    elif settings.mode == "modal":
        # Modal mode: sends raw JSON to the cloud — no flashinfer_bench needed locally
        from scripts.run_modal import app as modal_app
        from scripts.run_modal import run_benchmark as run_modal_benchmark

        solution_json = solution_path.read_text(encoding="utf-8")
        with modal_app.run():
            raw_results = run_modal_benchmark.remote(
                solution_json,
                warmup_runs=settings.warmup_runs,
                iterations=settings.iterations,
                num_trials=settings.num_trials,
            )
    else:
        raise ValueError(f"Unsupported mode: {settings.mode}")

    filtered = _filter_workloads(raw_results, settings.workload_focus)
    summary = _summarize_results(filtered)
    return {
        "results": filtered,
        "summary": summary,
    }

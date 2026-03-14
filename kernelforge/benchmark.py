"""Benchmark wrappers with structured results."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from statistics import median
from typing import Any

_SUCCESS_STATUSES = {"success", "passed", "clean", "ok"}


@dataclass(frozen=True)
class BenchSettings:
    mode: str
    warmup_runs: int
    iterations: int
    num_trials: int
    workload_focus: str | None


@dataclass(frozen=True)
class PreflightSettings:
    enabled: bool = True
    quick_workloads: int = 3
    quick_warmup_runs: int = 1
    quick_iterations: int = 1
    quick_num_trials: int = 1
    run_sanitizer: bool = True
    sanitizer_workloads: int = 1
    sanitizer_timeout: int = 300
    sanitizer_types: tuple[str, ...] = field(default_factory=lambda: ("memcheck",))
    fail_on_sanitizer_error: bool = True

    def cache_tag(self) -> str:
        sanitizer_types = ",".join(sorted(self.sanitizer_types))
        return (
            f"preflight:{int(self.enabled)}"
            f"|quick={self.quick_workloads}:{self.quick_warmup_runs}:{self.quick_iterations}:{self.quick_num_trials}"
            f"|san={int(self.run_sanitizer)}:{self.sanitizer_workloads}:{self.sanitizer_timeout}:{sanitizer_types}"
            f"|fail_on_san={int(self.fail_on_sanitizer_error)}"
        )


def kernel_hash(source: str) -> str:
    normalized = canonicalize_kernel_source(source)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if body and isinstance(body[0], ast.Expr):
            expr = body[0].value
            if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = self._strip(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._strip(node.body)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = self._strip(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.body = self._strip(node.body)
        self.generic_visit(node)
        return node


def canonicalize_kernel_source(source: str) -> str:
    try:
        tree = ast.parse(source)
        tree = _StripDocstrings().visit(tree)  # type: ignore[assignment]
        ast.fix_missing_locations(tree)
        return ast.unparse(tree).strip()
    except Exception:
        normalized_lines: list[str] = []
        for raw in source.splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if line:
                normalized_lines.append(line)
        return "\n".join(normalized_lines).strip()


def make_cache_key(source_hash: str, settings: BenchSettings, extra: str | None = None) -> str:
    focus = settings.workload_focus or "all"
    payload = (
        f"{source_hash}|{settings.mode}|{settings.warmup_runs}|"
        f"{settings.iterations}|{settings.num_trials}|{focus}"
    )
    if extra:
        payload = f"{payload}|{extra}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _filter_workloads(results: dict[str, Any], workload_focus: str | None) -> dict[str, Any]:
    if not workload_focus:
        return results
    out: dict[str, Any] = {}
    for def_name, workloads in results.items():
        if str(def_name).startswith("_") or not isinstance(workloads, dict):
            out[def_name] = workloads
            continue
        if workload_focus in workloads:
            out[def_name] = {workload_focus: workloads[workload_focus]}
            continue
        out[def_name] = {}
    return out


def _iter_workload_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for def_name, workloads in results.items():
        if str(def_name).startswith("_"):
            continue
        if not isinstance(workloads, dict):
            continue
        for row in workloads.values():
            if isinstance(row, dict) and "status" in row:
                rows.append(row)
    return rows


def _summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    statuses: list[str] = []
    speedups: list[float] = []
    latencies: list[float] = []
    errors: list[float] = []

    for row in _iter_workload_rows(results):
        status = str(row.get("status", "unknown"))
        statuses.append(status)
        if row.get("speedup_factor") is not None:
            speedups.append(float(row["speedup_factor"]))
        if row.get("latency_ms") is not None:
            latencies.append(float(row["latency_ms"]))
        if row.get("max_abs_error") is not None:
            errors.append(float(row["max_abs_error"]))

    success_count = sum(1 for s in statuses if s.lower() in _SUCCESS_STATUSES)
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


def _summarize_compile_results(results: dict[str, Any]) -> dict[str, Any]:
    build_error = results.get("_build_error")
    if isinstance(build_error, dict):
        return {
            "total_workloads": 0,
            "success_count": 0,
            "status_summary": "compile_failed",
            "median_speedup": None,
            "median_latency_ms": None,
            "max_abs_error": None,
            "score": -1e12,
        }
    return {
        "total_workloads": 0,
        "success_count": 0,
        "status_summary": "compile_success",
        "median_speedup": None,
        "median_latency_ms": None,
        "max_abs_error": None,
        "score": 0.0,
    }


def _summarize_sanitizer_results(results: dict[str, Any]) -> dict[str, Any]:
    statuses: list[str] = []
    for row in _iter_workload_rows(results):
        statuses.append(str(row.get("status", "unknown")).lower())

    skipped_reason = results.get("_sanitizer_skipped")
    if not statuses and skipped_reason:
        return {
            "total_workloads": 0,
            "success_count": 0,
            "status_summary": "sanitizer_skipped",
            "median_speedup": None,
            "median_latency_ms": None,
            "max_abs_error": None,
            "score": 0.0,
        }

    success_statuses = {"success", "clean", "passed", "ok"}
    success_count = sum(1 for status in statuses if status in success_statuses)
    total = len(statuses)
    all_ok = total > 0 and success_count == total
    status_summary = f"{success_count}/{total} sanitizer clean"
    if total == 0:
        status_summary = "0/0 sanitizer clean"
    return {
        "total_workloads": total,
        "success_count": success_count,
        "status_summary": status_summary,
        "median_speedup": None,
        "median_latency_ms": None,
        "max_abs_error": None,
        "score": 0.0 if all_ok or total == 0 else -1e12,
    }


def _first_non_success_error(results: dict[str, Any]) -> str:
    for def_name, workloads in results.items():
        if str(def_name).startswith("_") or not isinstance(workloads, dict):
            continue
        for wl_id, row in workloads.items():
            if not isinstance(row, dict):
                continue
            status = str(row.get("status", "")).lower()
            if status in _SUCCESS_STATUSES:
                continue
            if status == "incorrect_numerical":
                max_abs = row.get("max_abs_error")
                max_rel = row.get("max_rel_error")
                return (
                    f"incorrect_numerical: workload={wl_id} "
                    f"max_abs_error={max_abs} max_rel_error={max_rel}"
                )
            if status == "correctness_error":
                max_abs = row.get("max_abs_error")
                return f"correctness_error: workload={wl_id} max_abs_error={max_abs}"
            return str(row.get("error") or f"{status}: workload={wl_id}")
    build_error = results.get("_build_error")
    if isinstance(build_error, dict):
        return str(build_error.get("error_message") or "build failed")
    return ""


def _failure_details(results: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for def_name, workloads in results.items():
        if str(def_name).startswith("_") or not isinstance(workloads, dict):
            continue
        for wl_id, row in workloads.items():
            if not isinstance(row, dict):
                continue
            status = str(row.get("status", "")).lower()
            if status in _SUCCESS_STATUSES:
                continue
            details.append(
                {
                    "definition": def_name,
                    "workload_id": wl_id,
                    "status": row.get("status"),
                    "error": row.get("error"),
                    "max_abs_error": row.get("max_abs_error"),
                    "max_rel_error": row.get("max_rel_error"),
                }
            )
            if len(details) >= limit:
                return details
    return details


def run_benchmark(
    settings: BenchSettings,
    *,
    phase: str = "full",
    max_workloads: int | None = None,
    stop_on_error: bool = False,
    sanitizer_types: list[str] | None = None,
    sanitizer_timeout: int = 300,
) -> dict[str, Any]:
    from scripts.pack_solution import pack_solution

    solution_path = pack_solution()

    if settings.mode == "local":
        # Local mode: needs flashinfer_bench + triton + CUDA (Linux GPU only)
        from flashinfer_bench import BenchmarkConfig, Solution
        from scripts.run_local import run_benchmark as run_local_benchmark
        from scripts.run_local import run_sanitizer as run_local_sanitizer

        solution = Solution.model_validate_json(
            solution_path.read_text(encoding="utf-8")
        )
        config = BenchmarkConfig(
            warmup_runs=settings.warmup_runs,
            iterations=settings.iterations,
            num_trials=settings.num_trials,
        )
        if phase == "sanitizer":
            raw_results = run_local_sanitizer(
                solution,
                sanitizer_types=sanitizer_types,
                max_workloads=max_workloads,
                timeout=sanitizer_timeout,
            )
        else:
            raw_results = run_local_benchmark(
                solution,
                config,
                max_workloads=max_workloads,
                stop_on_error=stop_on_error,
                compile_only=(phase == "compile_only"),
            )

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
                phase=phase,
                max_workloads=max_workloads,
                stop_on_error=stop_on_error,
                sanitizer_types=sanitizer_types,
                sanitizer_timeout=sanitizer_timeout,
            )
    else:
        raise ValueError(f"Unsupported mode: {settings.mode}")

    filtered = _filter_workloads(raw_results, settings.workload_focus)
    if phase == "compile_only":
        summary = _summarize_compile_results(filtered)
    elif phase == "sanitizer":
        summary = _summarize_sanitizer_results(filtered)
    else:
        summary = _summarize_results(filtered)
    return {
        "results": filtered,
        "summary": summary,
        "phase": phase,
    }


def run_preflight(settings: BenchSettings, preflight: PreflightSettings) -> dict[str, Any]:
    if not preflight.enabled:
        return {
            "ok": True,
            "status_summary": "preflight_disabled",
            "stages": [],
        }

    stages: list[dict[str, Any]] = []

    compile_out = run_benchmark(settings, phase="compile_only")
    compile_ok = compile_out["summary"].get("status_summary") == "compile_success"
    compile_error = _first_non_success_error(compile_out.get("results", {}))
    stages.append(
        {
            "phase": "compile_only",
            "ok": compile_ok,
            "summary": compile_out["summary"],
            "error": compile_error,
        }
    )
    if not compile_ok:
        return {
            "ok": False,
            "failed_stage": "compile_only",
            "error": compile_error or "compile check failed",
            "stages": stages,
            "results": compile_out.get("results", {}),
            "summary": compile_out.get("summary", {}),
            "failure_details": _failure_details(compile_out.get("results", {})),
        }

    quick_settings = BenchSettings(
        mode=settings.mode,
        warmup_runs=preflight.quick_warmup_runs,
        iterations=preflight.quick_iterations,
        num_trials=preflight.quick_num_trials,
        workload_focus=settings.workload_focus,
    )
    quick_out = run_benchmark(
        quick_settings,
        phase="quick",
        max_workloads=max(1, int(preflight.quick_workloads)),
        stop_on_error=True,
    )
    quick_summary = quick_out["summary"]
    quick_total = int(quick_summary.get("total_workloads") or 0)
    quick_success = int(quick_summary.get("success_count") or 0)
    quick_ok = quick_total > 0 and quick_total == quick_success
    quick_error = _first_non_success_error(quick_out.get("results", {}))
    stages.append(
        {
            "phase": "quick_correctness",
            "ok": quick_ok,
            "summary": quick_summary,
            "error": quick_error,
        }
    )
    if not quick_ok:
        return {
            "ok": False,
            "failed_stage": "quick_correctness",
            "error": quick_error or "quick correctness check failed",
            "stages": stages,
            "results": quick_out.get("results", {}),
            "summary": quick_summary,
            "failure_details": _failure_details(quick_out.get("results", {})),
        }

    if preflight.run_sanitizer:
        sanitizer_out = run_benchmark(
            settings,
            phase="sanitizer",
            max_workloads=max(1, int(preflight.sanitizer_workloads)),
            stop_on_error=True,
            sanitizer_types=list(preflight.sanitizer_types),
            sanitizer_timeout=int(preflight.sanitizer_timeout),
        )
        sanitizer_summary = sanitizer_out["summary"]
        sanitizer_status = str(sanitizer_summary.get("status_summary", ""))
        sanitizer_total = int(sanitizer_summary.get("total_workloads") or 0)
        sanitizer_success = int(sanitizer_summary.get("success_count") or 0)
        sanitizer_ok = (
            sanitizer_status == "sanitizer_skipped"
            or (sanitizer_total > 0 and sanitizer_total == sanitizer_success)
        )
        sanitizer_error = _first_non_success_error(sanitizer_out.get("results", {}))
        stages.append(
            {
                "phase": "sanitizer",
                "ok": sanitizer_ok,
                "summary": sanitizer_summary,
                "error": sanitizer_error,
            }
        )
        if not sanitizer_ok and preflight.fail_on_sanitizer_error:
            return {
                "ok": False,
                "failed_stage": "sanitizer",
                "error": sanitizer_error or "sanitizer failed",
                "stages": stages,
                "results": sanitizer_out.get("results", {}),
                "summary": sanitizer_summary,
                "failure_details": _failure_details(sanitizer_out.get("results", {})),
            }

    return {
        "ok": True,
        "status_summary": "preflight_passed",
        "stages": stages,
    }

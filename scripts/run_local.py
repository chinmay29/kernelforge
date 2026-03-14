"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
"""

import os
import sys
import json
import traceback
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution

_SUCCESS_STATUSES = {"success", "passed", "clean", "ok"}


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def _sanitize_output_to_status(output: object) -> tuple[str, str]:
    text = ""
    if isinstance(output, str):
        text = output
    else:
        try:
            text = json.dumps(output)
        except Exception:
            text = str(output)
    lower = text.lower()
    if "error summary: 0 errors" in lower or "0 errors from 0 contexts" in lower:
        return "success", ""
    if "error summary:" in lower:
        return "sanitizer_error", "compute-sanitizer reported memory/sync errors"
    if '"status": "error"' in lower or '"success": false' in lower:
        return "sanitizer_error", "sanitizer tool reported an error status"
    return "success", ""


def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig = None,
    *,
    max_workloads: int | None = None,
    stop_on_error: bool = False,
    compile_only: bool = False,
) -> dict:
    """Run benchmark locally and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    workloads = sorted(workloads, key=lambda w: w.workload.uuid)
    if max_workloads is not None and max_workloads > 0:
        workloads = workloads[:max_workloads]

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    if compile_only:
        from flashinfer_bench.compile import get_builder_registry

        registry = get_builder_registry()
        try:
            runnable = registry.build(definition, solution)
            try:
                runnable.close()
            except Exception:
                pass
            return {
                definition.name: {},
                "_compile_check": {"status": "success"},
            }
        except Exception as e:
            tb = traceback.format_exc()
            return {
                definition.name: {},
                "_build_error": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": tb,
                },
            }

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry
            if stop_on_error and str(trace.evaluation.status.value).lower() not in _SUCCESS_STATUSES:
                break

    return results


def run_sanitizer(
    solution: Solution,
    *,
    sanitizer_types: list[str] | None = None,
    max_workloads: int | None = 1,
    timeout: int = 300,
) -> dict:
    """Run compute-sanitizer checks on a small set of workloads."""
    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)
    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    workloads = sorted(workloads, key=lambda w: w.workload.uuid)
    if max_workloads is not None and max_workloads > 0:
        workloads = workloads[:max_workloads]

    if not workloads:
        return {
            definition.name: {},
            "_sanitizer_skipped": "no_workloads",
        }

    try:
        from flashinfer_bench.agents import flashinfer_bench_run_sanitizer
    except Exception as exc:
        return {
            definition.name: {},
            "_sanitizer_skipped": f"sanitizer_unavailable: {exc}",
        }

    san_types = sanitizer_types or ["memcheck"]
    results = {definition.name: {}}
    for wl_trace in workloads:
        wl = wl_trace.workload
        try:
            output = flashinfer_bench_run_sanitizer(
                solution=solution,
                workload=wl,
                sanitizer_types=san_types,
                timeout=timeout,
            )
            status, err = _sanitize_output_to_status(output)
            row = {
                "status": status,
                "solution": solution.name,
            }
            if err:
                row["error"] = err
            results[definition.name][wl.uuid] = row
        except Exception as e:
            results[definition.name][wl.uuid] = {
                "status": "runtime_error",
                "solution": solution.name,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


def main():
    """Pack solution and run benchmark."""
    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark...")
    results = run_benchmark(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)


if __name__ == "__main__":
    main()

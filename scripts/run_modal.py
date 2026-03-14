"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/mlsys26-contest /mlsys26-contest
"""

import sys
import json
import os
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT = "/data"
DEFAULT_TRACE_SET_PATH = os.environ.get("FLASHINFER_TRACESET_PATH", "/data/mlsys26-contest")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)

_SUCCESS_STATUSES = {"success", "passed", "clean", "ok"}


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


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={VOLUME_MOUNT: trace_volume})
def run_benchmark(
    solution_json: str,
    warmup_runs: int = 3,
    iterations: int = 100,
    num_trials: int = 5,
    phase: str = "full",
    trace_set_path: str = DEFAULT_TRACE_SET_PATH,
    max_workloads: int | None = None,
    stop_on_error: bool = False,
    sanitizer_types: list[str] | None = None,
    sanitizer_timeout: int = 300,
) -> dict:
    """Run benchmark on Modal B200 and return results.

    Runs everything IN-PROCESS to avoid IPC "header too large" errors
    that occur when PersistentRunner/IsolatedRunner try to transfer
    massive FP8 tensors between processes.
    """
    import traceback

    import torch
    from flashinfer_bench import BenchmarkConfig, Solution, TraceSet
    from flashinfer_bench.bench.evaluators import resolve_evaluator
    from flashinfer_bench.compile import get_builder_registry

    solution = Solution.model_validate_json(solution_json)
    config = BenchmarkConfig(
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=num_trials,
    )

    trace_root = Path(trace_set_path)
    if not trace_root.exists():
        raise FileNotFoundError(
            f"Trace set not found inside Modal volume at {trace_set_path}. "
            "Upload the dataset to the volume and/or pass `--trace-set-path`."
        )

    trace_set = TraceSet.from_path(trace_root)

    if solution.definition not in trace_set.definitions:
        raise ValueError(
            f"Definition '{solution.definition}' not found in trace set. "
            f"Available: {list(trace_set.definitions.keys())}"
        )

    definition = trace_set.definitions[solution.definition]
    workload_traces = trace_set.workloads.get(solution.definition, [])
    workload_traces = sorted(workload_traces, key=lambda w: w.workload.uuid)
    if max_workloads is not None and max_workloads > 0:
        workload_traces = workload_traces[:max_workloads]

    if not workload_traces:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    device = "cuda:0"
    torch.cuda.set_device(0)

    # Build the solution once (compile / import the kernel)
    registry = get_builder_registry()
    build_error_detail = None
    try:
        runnable = registry.build(definition, solution)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[run_modal] Build failed: {e}\n{tb}")
        build_error_detail = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": tb,
        }
        return {
            definition.name: {},
            "_build_error": build_error_detail  # Include detailed error
        }

    if phase == "compile_only":
        try:
            runnable.close()
        except Exception:
            pass
        return {
            definition.name: {},
            "_compile_check": {"status": "success"},
        }

    if phase == "sanitizer":
        results = {definition.name: {}}
        san_types = sanitizer_types or ["memcheck"]
        try:
            from flashinfer_bench.agents import flashinfer_bench_run_sanitizer
        except Exception as exc:
            return {
                definition.name: {},
                "_sanitizer_skipped": f"sanitizer_unavailable: {exc}",
            }
        for wl_trace in workload_traces:
            wl = wl_trace.workload
            uuid_short = wl.uuid[:8]
            try:
                san_output = flashinfer_bench_run_sanitizer(
                    solution=solution,
                    workload=wl,
                    sanitizer_types=san_types,
                    timeout=sanitizer_timeout,
                )
                status, err = _sanitize_output_to_status(san_output)
                row = {
                    "status": status,
                    "solution": solution.name,
                }
                if err:
                    row["error"] = err
                results[definition.name][wl.uuid] = row
                print(f"[run_modal] sanitizer wl={uuid_short}: {status}")
                if status != "success" and stop_on_error:
                    break
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[run_modal] sanitizer wl={uuid_short}: ERROR: {e}")
                results[definition.name][wl.uuid] = {
                    "status": "runtime_error",
                    "solution": solution.name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": tb,
                }
                if stop_on_error:
                    break
        try:
            runnable.close()
        except Exception:
            pass
        return results

    evaluator_cls = resolve_evaluator(definition)
    results = {definition.name: {}}

    for wl_trace in workload_traces:
        wl = wl_trace.workload
        uuid_short = wl.uuid[:8]

        try:
            # Build baseline (reference impl) — IN-PROCESS, no IPC
            baseline = evaluator_cls.build_baseline(
                defn=definition,
                workload=wl,
                cfg=config,
                device=device,
                traceset_root=trace_root,
            )

            # Evaluate the solution against the baseline — IN-PROCESS
            evaluation = evaluator_cls.evaluate(
                defn=definition,
                sol_runnable=runnable,
                inputs=baseline.inputs,
                ref_outputs=baseline.outputs,
                ref_mean_latency_ms=baseline.mean_latency_ms,
                cfg=config,
                log_path=f"/tmp/eval_{wl.uuid}.log",
                device=device,
            )

            entry = {
                "status": evaluation.status.value,
                "solution": solution.name,
            }
            if evaluation.performance:
                entry["latency_ms"] = evaluation.performance.latency_ms
                entry["reference_latency_ms"] = evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = evaluation.performance.speedup_factor
            if evaluation.correctness:
                entry["max_abs_error"] = evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = evaluation.correctness.max_relative_error

            results[definition.name][wl.uuid] = entry
            status_str = evaluation.status.value
            print(f"[run_modal] wl={uuid_short}: {status_str}", end="")
            if evaluation.performance:
                print(f" | {evaluation.performance.speedup_factor:.2f}x", end="")
            print()
            if stop_on_error and status_str.lower() not in _SUCCESS_STATUSES:
                break

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[run_modal] wl={uuid_short}: ERROR: {e}")
            print(f"[run_modal] Full traceback:\n{tb}")
            results[definition.name][wl.uuid] = {
                "status": "runtime_error",
                "solution": solution.name,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": tb,  # Include full traceback
            }
            if stop_on_error:
                break

    try:
        runnable.close()
    except Exception:
        pass

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


@app.local_entrypoint()
def main(
    phase: str = "full",
    warmup_runs: int = 3,
    iterations: int = 100,
    num_trials: int = 5,
    trace_set_path: str = DEFAULT_TRACE_SET_PATH,
    max_workloads: int = 0,
    stop_on_error: bool = False,
    sanitizer_types: str = "memcheck",
    sanitizer_timeout: int = 300,
    print_json: bool = False,
):
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution JSON...")
    solution_json = solution_path.read_text(encoding="utf-8")

    sanitized_types = [item.strip() for item in sanitizer_types.split(",") if item.strip()]
    max_workload_arg = max_workloads if max_workloads > 0 else None

    print("\nRunning benchmark on Modal B200...")
    print(
        json.dumps(
            {
                "phase": phase,
                "warmup_runs": warmup_runs,
                "iterations": iterations,
                "num_trials": num_trials,
                "trace_set_path": trace_set_path,
                "max_workloads": max_workload_arg,
                "stop_on_error": stop_on_error,
                "sanitizer_types": sanitized_types,
                "sanitizer_timeout": sanitizer_timeout,
            },
            indent=2,
        )
    )
    results = run_benchmark.remote(
        solution_json,
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=num_trials,
        phase=phase,
        trace_set_path=trace_set_path,
        max_workloads=max_workload_arg,
        stop_on_error=stop_on_error,
        sanitizer_types=sanitized_types or None,
        sanitizer_timeout=sanitizer_timeout,
    )

    if not results:
        print("No results returned!")
        return

    if print_json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)

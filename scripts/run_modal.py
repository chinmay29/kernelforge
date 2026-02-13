"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={VOLUME_MOUNT: trace_volume})
def run_benchmark(solution_json: str, warmup_runs: int = 3, iterations: int = 100, num_trials: int = 5) -> dict:
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
    from flashinfer_bench.data import EvaluationStatus

    solution = Solution.model_validate_json(solution_json)
    config = BenchmarkConfig(
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=num_trials,
    )

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(
            f"Definition '{solution.definition}' not found in trace set. "
            f"Available: {list(trace_set.definitions.keys())}"
        )

    definition = trace_set.definitions[solution.definition]
    workload_traces = trace_set.workloads.get(solution.definition, [])

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
                traceset_root=Path(TRACE_SET_PATH),
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
def main():
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution JSON...")
    solution_json = solution_path.read_text(encoding="utf-8")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution_json)

    if not results:
        print("No results returned!")
        return

    print_results(results)

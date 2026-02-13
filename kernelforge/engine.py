"""Main optimization loop for KernelForge."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from kernelforge.benchmark import BenchSettings, kernel_hash, make_cache_key, run_benchmark
from kernelforge.config import load_project_config
from kernelforge.io_utils import append_jsonl, dump_json, load_json
from kernelforge.agents import CoordinatorAgent
from kernelforge.rag import RagStore


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_loop(
    project_root: Path,
    mode: str,
    iters: int,
    seed: int,
    warmup_runs: int,
    iterations: int,
    num_trials: int,
    workload_focus: str | None,
) -> dict[str, Any]:
    cfg = load_project_config(project_root)
    kernel_path = cfg.kernel_path
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel source not found: {kernel_path}")

    base_dir = project_root / "kernelforge"
    rag_dir = base_dir / "rag_store"
    cache_dir = base_dir / "cache"
    prov_dir = base_dir / "provenance"
    rag_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)

    runs_path = rag_dir / "runs.jsonl"
    cache_path = cache_dir / "results.json"
    summary_path = prov_dir / "latest_summary.json"

    rag = RagStore(store_dir=rag_dir, runs_path=runs_path)
    llm_log_dir = prov_dir / "llm_logs"
    llm_log_dir.mkdir(parents=True, exist_ok=True)
    coordinator = CoordinatorAgent(seed=seed, rag_store=rag, log_dir=llm_log_dir)
    cache = load_json(cache_path, default={})

    best_source = kernel_path.read_text(encoding="utf-8")
    best_hash = kernel_hash(best_source)
    best_score = -1e12
    best_result: dict[str, Any] | None = None
    last_error = ""
    last_perf = ""

    for i in range(iters):
        print(f"\n{'='*60}")
        print(f"  🔄 ITERATION {i+1}/{iters}")
        print(f"{'='*60}")

        # Use error-specific retrieval when previous iter had errors
        print(f"  📚 [1/4] Seed RAG retrieval...", end=" ", flush=True)
        if last_error:
            retrieved = rag.get_error_fixes(last_error)
            print(f"error-mode ({len(retrieved)} fixes)")
        else:
            query = f"{cfg.definition} {last_perf}"
            tags = ["moe", "fp8", "blackwell"]
            retrieved = rag.retrieve(query=query, k=6, tags=tags)
            print(f"ok ({len(retrieved)} entries)")

        print(f"  🤖 [2/4] Multi-agent generation/validation (LLM + checks)...", flush=True)
        proposal = coordinator.propose(
            best_source,
            retrieved=retrieved,
            max_mutations=2,
            last_error=last_error,
        )
        mutation_names = [m.name for m in proposal.mutations]
        print(f"         Mutation: {mutation_names}")
        print(f"         Reasoning: {proposal.reasoning[:120]}")

        candidate_source = proposal.source
        candidate_hash = kernel_hash(candidate_source)
        kernel_path.write_text(candidate_source, encoding="utf-8")
        print(f"         Lines: {len(candidate_source.splitlines())} | Hash: {candidate_hash[:12]}")

        proposal_failed = all(
            m.name.startswith("multi_agent_failed_iter_") for m in proposal.mutations
        )
        if proposal_failed:
            print("         ⚠️  Proposal failed validation/constraints; skipping benchmark for this iteration.")
            bench_out = {
                "results": {},
                "summary": {
                    "total_workloads": 0,
                    "success_count": 0,
                    "status_summary": "proposal_failed_pre_benchmark",
                    "median_speedup": None,
                    "median_latency_ms": None,
                    "max_abs_error": None,
                    "score": -1e12,
                },
            }
            error = "proposal_failed_pre_benchmark"
            from_cache = False
        else:
            settings = BenchSettings(
                mode=mode,
                warmup_runs=warmup_runs,
                iterations=iterations,
                num_trials=num_trials,
                workload_focus=workload_focus,
            )
            key = make_cache_key(candidate_hash, settings)
            error = ""
            from_cache = key in cache
            if from_cache:
                print(f"  📦 [3/4] Benchmark (cached)")
                bench_out = cache[key]
            else:
                print(f"  🚀 [3/4] Benchmark on {mode} (19 workloads, ~3 min)...", flush=True)
                try:
                    bench_out = run_benchmark(settings)
                except Exception as exc:
                    error = str(exc)
                    print(f"         ❌ Benchmark error: {error[:100]}")
                    bench_out = {
                        "results": {},
                        "summary": {
                            "total_workloads": 0,
                            "success_count": 0,
                            "status_summary": "0/0 success",
                            "median_speedup": None,
                            "median_latency_ms": None,
                            "max_abs_error": None,
                            "score": -1e12,
                        },
                    }
                cache[key] = bench_out
                dump_json(cache_path, cache)

        score = float(bench_out["summary"]["score"])
        status_summary = bench_out["summary"].get("status_summary", "unknown")
        med_speedup = bench_out["summary"].get("median_speedup")
        max_err = bench_out["summary"].get("max_abs_error")

        # Extract per-workload errors from results (these come from Modal)
        workload_errors: list[str] = []
        detailed_error_info = None  # Store full error details for logging
        error_statuses = {
            "runtime_error",
            "build_error",
            "compilation_error",
            "incorrect_numerical",
            "correctness_error",
        }
        for workloads in bench_out.get("results", {}).values():
            for wl_id, row in workloads.items():
                wl_status = str(row.get("status", "")).lower()
                wl_error = row.get("error", "")
                if wl_status in error_statuses:
                    if not wl_error:
                        if wl_status == "incorrect_numerical":
                            wl_error = (
                                f"INCORRECT_NUMERICAL: max_abs_error={row.get('max_abs_error')} "
                                f"workload={wl_id}"
                            )
                        else:
                            wl_error = f"{wl_status}: workload={wl_id}"
                    if wl_error not in workload_errors:
                        workload_errors.append(wl_error)
                    # Capture detailed error info from first error
                    if detailed_error_info is None:
                        detailed_error_info = {
                            "error": wl_error,
                            "error_type": row.get("error_type", "Unknown"),
                            "traceback": row.get("traceback", ""),
                            "workload_id": wl_id,
                            "status": wl_status,
                        }

        # Check for build errors
        if "_build_error" in bench_out.get("results", {}):
            build_err = bench_out["results"]["_build_error"]
            detailed_error_info = {
                "error": build_err.get("error_message", ""),
                "error_type": build_err.get("error_type", "BuildError"),
                "traceback": build_err.get("traceback", ""),
                "workload_id": "build",
                "status": "build_error",
            }

        # Save detailed error to file for next iteration
        if detailed_error_info:
            error_file = llm_log_dir / f"iter_{i:03d}" / "benchmark_error.txt"
            error_file.parent.mkdir(parents=True, exist_ok=True)
            with open(error_file, "w") as f:
                f.write(f"Error Type: {detailed_error_info['error_type']}\n")
                f.write(f"Error Message: {detailed_error_info['error']}\n")
                f.write(f"Status: {detailed_error_info['status']}\n")
                f.write(f"Workload: {detailed_error_info['workload_id']}\n\n")
                f.write("Full Traceback:\n")
                f.write("="*80 + "\n")
                f.write(detailed_error_info['traceback'])

        # Combine exception error with workload errors
        if workload_errors and not error:
            error = workload_errors[0][:800]  # first unique error, truncated
            print(f"         ❌ Workload error: {error[:150]}")
        if len(workload_errors) > 1:
            print(f"         ({len(workload_errors)} unique workload errors total)")

        print(f"  📊 [4/4] Results:")
        print(f"         Score: {score:.4f}")
        print(f"         Speedup: {med_speedup}")
        print(f"         Status: {status_summary}")
        print(f"         Max error: {max_err}")

        if score > best_score:
            print(f"         🏆 NEW BEST! ({score:.4f} > {best_score:.4f})")
            best_score = score
            best_hash = candidate_hash
            best_source = candidate_source
            best_result = bench_out
        else:
            print(f"         ↩️  Keeping previous best ({best_score:.4f})")
            kernel_path.write_text(best_source, encoding="utf-8")

        last_perf = f"median_speedup={med_speedup}" if med_speedup is not None else ""
        last_error = error

        run_row = {
            "timestamp_utc": _utc_now(),
            "iter": i,
            "definition": cfg.definition,
            "mode": mode,
            "kernel_hash": candidate_hash,
            "status_summary": status_summary,
            "score": score,
            "error": error,
            "from_cache": from_cache,
            "mutations": [m.name for m in proposal.mutations],
            "mutation_params": [m.params for m in proposal.mutations],
            "diffs": [m.diff for m in proposal.mutations],
            "reasoning": proposal.reasoning,
            "retrieved": [r.get("id") for r in retrieved],
            "summary": bench_out["summary"],
        }
        rag.log_run(run_row)
        append_jsonl(prov_dir / "runs.jsonl", run_row)
        coordinator.update_last_run(score=score, summary=bench_out["summary"], error=error)

    print(f"\n{'='*60}")
    print(f"  ✅ DONE — {iters} iterations complete")
    print(f"  Best score: {best_score:.4f}")
    print(f"{'='*60}")

    kernel_path.write_text(best_source, encoding="utf-8")
    final = {
        "timestamp_utc": _utc_now(),
        "mode": mode,
        "iters": iters,
        "seed": seed,
        "definition": cfg.definition,
        "best_kernel_hash": best_hash,
        "best_score": best_score,
        "best_summary": (best_result or {}).get("summary"),
        "kernel_path": str(kernel_path),
    }
    dump_json(summary_path, final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="KernelForge optimization loop")
    parser.add_argument("--mode", choices=["local", "modal"], default="local")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--workload-id", type=str, default=None)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    result = run_loop(
        project_root=args.project_root.resolve(),
        mode=args.mode,
        iters=args.iters,
        seed=args.seed,
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        workload_focus=args.workload_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

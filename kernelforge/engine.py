"""Main optimization loop for KernelForge."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from kernelforge.benchmark import (
    BenchSettings,
    PreflightSettings,
    kernel_hash,
    make_cache_key,
    run_benchmark,
    run_preflight,
)
from kernelforge.config import load_project_config
from kernelforge.io_utils import append_jsonl, dump_json, load_json
from kernelforge.agents import CoordinatorAgent
from kernelforge.rag import RagStore


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _preflight_failure_result(preflight_out: dict[str, Any]) -> dict[str, Any]:
    failed_stage = str(preflight_out.get("failed_stage", "unknown"))
    status = f"preflight_failed:{failed_stage}"
    return {
        "results": preflight_out.get("results", {}),
        "summary": {
            "total_workloads": 0,
            "success_count": 0,
            "status_summary": status,
            "median_speedup": None,
            "median_latency_ms": None,
            "max_abs_error": None,
            "score": -1e12,
        },
        "preflight": preflight_out,
    }


def run_loop(
    project_root: Path,
    mode: str,
    iters: int,
    seed: int,
    warmup_runs: int,
    iterations: int,
    num_trials: int,
    workload_focus: str | None,
    preflight_enabled: bool = True,
    preflight_quick_workloads: int = 3,
    preflight_sanitizer_workloads: int = 1,
    preflight_run_sanitizer: bool = True,
    preflight_sanitizer_types: list[str] | None = None,
    autofix_correctness_only: bool = False,
    llm_model: str = "claude-sonnet-4-5-20250929",
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
    coordinator = CoordinatorAgent(
        seed=seed,
        rag_store=rag,
        log_dir=llm_log_dir,
        model=llm_model,
    )
    cache = load_json(cache_path, default={})

    preflight_settings = PreflightSettings(
        enabled=preflight_enabled,
        quick_workloads=max(1, int(preflight_quick_workloads)),
        quick_warmup_runs=1,
        quick_iterations=1,
        quick_num_trials=1,
        run_sanitizer=preflight_run_sanitizer,
        sanitizer_workloads=max(1, int(preflight_sanitizer_workloads)),
        sanitizer_timeout=300,
        sanitizer_types=tuple(preflight_sanitizer_types or ["memcheck"]),
        fail_on_sanitizer_error=True,
    )

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
        print(f"  📚 [1/5] Seed RAG retrieval...", end=" ", flush=True)
        if last_error:
            retrieved = rag.get_error_fixes(last_error)
            print(f"error-mode ({len(retrieved)} fixes)")
        else:
            query = f"{cfg.definition} {last_perf}"
            tags = ["moe", "fp8", "blackwell"]
            retrieved = rag.retrieve(query=query, k=6, tags=tags)
            print(f"ok ({len(retrieved)} entries)")

        print(f"  🤖 [2/5] Multi-agent generation/validation (LLM + checks)...", flush=True)
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
        preflight_passed_this_iter = False
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
            key = make_cache_key(
                candidate_hash,
                settings,
                extra=(
                    f"{preflight_settings.cache_tag()}|autofix={int(autofix_correctness_only)}"
                ),
            )
            error = ""
            from_cache = key in cache
            if from_cache:
                print(f"  📦 [3/5] Benchmark (cached)")
                bench_out = cache[key]
            else:
                preflight_out: dict[str, Any] = {
                    "ok": True,
                    "status_summary": "preflight_disabled",
                    "stages": [],
                }
                try:
                    if preflight_settings.enabled:
                        print("  🧪 [3/5] Preflight: compile -> quick correctness -> sanitizer...", flush=True)
                        preflight_out = run_preflight(settings, preflight_settings)
                        if not preflight_out.get("ok", False):
                            failed_stage = str(preflight_out.get("failed_stage", "unknown"))
                            error = str(preflight_out.get("error", "preflight failed"))
                            print(f"         ❌ Preflight failed at {failed_stage}: {error[:120]}")
                            details = preflight_out.get("failure_details") or []
                            if details:
                                first = details[0]
                                print(
                                    "         ↪ detail:"
                                    f" workload={first.get('workload_id')}"
                                    f" status={first.get('status')}"
                                    f" max_abs={first.get('max_abs_error')}"
                                    f" max_rel={first.get('max_rel_error')}"
                                )
                            bench_out = _preflight_failure_result(preflight_out)
                        else:
                            print("         ✅ Preflight passed")
                            preflight_passed_this_iter = True

                    if not preflight_settings.enabled or preflight_out.get("ok", False):
                        if autofix_correctness_only:
                            print("  ⏭️  [4/5] Autofix mode: skipping full benchmark after preflight pass.")
                            bench_out = {
                                "results": {},
                                "summary": {
                                    "total_workloads": 0,
                                    "success_count": 0,
                                    "status_summary": "preflight_passed",
                                    "median_speedup": None,
                                    "median_latency_ms": None,
                                    "max_abs_error": None,
                                    "score": 0.0,
                                },
                                "preflight": preflight_out,
                            }
                        else:
                            print(f"  🚀 [4/5] Full benchmark on {mode} (19 workloads, ~3 min)...", flush=True)
                            bench_out = run_benchmark(settings, phase="full")
                            bench_out["preflight"] = preflight_out
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
                        "preflight": {
                            "ok": False,
                            "status_summary": "preflight_exception",
                            "error": error,
                            "stages": [],
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
            "sanitizer_error",
        }
        for def_name, workloads in bench_out.get("results", {}).items():
            if str(def_name).startswith("_") or not isinstance(workloads, dict):
                continue
            for wl_id, row in workloads.items():
                if not isinstance(row, dict):
                    continue
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
            if isinstance(build_err, dict):
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

        print(f"  📊 [5/5] Results:")
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
            "preflight": bench_out.get("preflight"),
        }
        rag.log_run(run_row)
        append_jsonl(prov_dir / "runs.jsonl", run_row)
        coordinator.update_last_run(score=score, summary=bench_out["summary"], error=error)

        if autofix_correctness_only and preflight_passed_this_iter:
            print("         ✅ Autofix target met: preflight passed. Stopping early.")
            break

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
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Anthropic model id used by the generator agent.",
    )
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--workload-id", type=str, default=None)
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--preflight-quick-workloads", type=int, default=3)
    parser.add_argument("--preflight-sanitizer-workloads", type=int, default=1)
    parser.add_argument("--skip-preflight-sanitizer", action="store_true")
    parser.add_argument(
        "--autofix-correctness",
        action="store_true",
        help="Run iterative fix loop with preflight gating and stop once preflight passes (skips full benchmark).",
    )
    parser.add_argument(
        "--preflight-sanitizer-types",
        type=str,
        default="memcheck",
        help="Comma-separated sanitizer checks (e.g. memcheck,racecheck,synccheck,initcheck).",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    sanitizer_types = [
        t.strip() for t in args.preflight_sanitizer_types.split(",") if t.strip()
    ] or ["memcheck"]

    result = run_loop(
        project_root=args.project_root.resolve(),
        mode=args.mode,
        iters=args.iters,
        seed=args.seed,
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        workload_focus=args.workload_id,
        preflight_enabled=not args.no_preflight,
        preflight_quick_workloads=args.preflight_quick_workloads,
        preflight_sanitizer_workloads=args.preflight_sanitizer_workloads,
        preflight_run_sanitizer=not args.skip_preflight_sanitizer,
        preflight_sanitizer_types=sanitizer_types,
        autofix_correctness_only=args.autofix_correctness,
        llm_model=args.model,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

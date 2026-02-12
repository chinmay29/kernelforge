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
from kernelforge.propose import Proposer
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
    proposer = Proposer(seed=seed)
    cache = load_json(cache_path, default={})

    best_source = kernel_path.read_text(encoding="utf-8")
    best_hash = kernel_hash(best_source)
    best_score = -1e12
    best_result: dict[str, Any] | None = None
    last_error = ""
    last_perf = ""

    for i in range(iters):
        # Use error-specific retrieval when previous iter had errors
        if last_error:
            retrieved = rag.get_error_fixes(last_error)
        else:
            query = f"{cfg.definition} {last_perf}"
            tags = ["moe", "fp8", "blackwell"]
            retrieved = rag.retrieve(query=query, k=6, tags=tags)
        proposal = proposer.propose(best_source, retrieved=retrieved, max_mutations=2)

        candidate_source = proposal.source
        candidate_hash = kernel_hash(candidate_source)
        kernel_path.write_text(candidate_source, encoding="utf-8")

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
            bench_out = cache[key]
        else:
            try:
                bench_out = run_benchmark(settings)
            except Exception as exc:
                error = str(exc)
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
        if score > best_score:
            best_score = score
            best_hash = candidate_hash
            best_source = candidate_source
            best_result = bench_out
        else:
            kernel_path.write_text(best_source, encoding="utf-8")

        status_summary = bench_out["summary"].get("status_summary", "unknown")
        med_speedup = bench_out["summary"].get("median_speedup")
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
            "retrieved": [r.get("id") for r in retrieved],
            "summary": bench_out["summary"],
        }
        rag.log_run(run_row)
        append_jsonl(prov_dir / "runs.jsonl", run_row)

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


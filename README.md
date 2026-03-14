# [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/)

Create high-performance GPU kernels for state-of-the-art LLM architectures on NVIDIA Blackwell GPUs with humans and/or AI agents.

---

<p align="center">
  <a href="https://www.nvidia.com"><img src="images/nvidia-logo.svg" alt="NVIDIA" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://modal.com"><img src="images/modal-logo.png" alt="Modal" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://mlsys.org"><img src="images/mlsys-logo.svg" alt="MLSys" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer"><img src="images/flashinfer-logo.png" alt="FlashInfer" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer-bench"><img src="images/fib_logo.png" alt="FlashInfer-Bench" height="50"/></a>
</p>

---

[FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench) is our official framework to evaluate your AI-generated kernels.

## Updates

* 2026.02.05: Full dataset for definitions and workloads are released at [HuggingFace](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)

## Competition Tracks

The competition features three tracks, each targeting a critical LLM operation:

| Track | Description |
|-------|-------------|
| **fused_moe** | Fused Mixture-of-Experts kernel for efficient expert routing and computation |
| **sparse_attention** | Sparse attention mechanisms for long-context inference |
| **gated_delta_net** | Gated delta network operations for efficient state updates |

**Fork this template once per track** you want to compete in (separate repos for each track).

## Getting Started

### 1. Fork This Template

Click "Use this template" or fork this repository to create your solution repo.

### 2. Install Dependencies

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal
```

### 3. Download the TraceSet

We provide kernel definitions and workloads in [FlashInfer-Trace format](https://bench.flashinfer.ai/docs/flashinfer-trace). Clone the competition dataset from HuggingFace:

```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
```

Set the environment variable:

```bash
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

### 4. Configure Your Solution

Edit `config.toml` to set your track and team info:

```toml
[solution]
name = "my-team-solution-v1"      # Solution name
definition = "fused_moe"          # Track: fused_moe | sparse_attention | gated_delta_net
author = "team-name"              # Team/author name

[build]
language = "triton"               # triton | cuda
entry_point = "kernel"            # Kernel function name
destination_passing_style = false # set false for value-returning kernels
```

### 5. Implement Your Kernel

**For Triton:**
Edit `solution/triton/kernel.py` with your implementation.

**For CUDA:**
Edit `solution/cuda/kernel.cu` and `solution/cuda/binding.py` with your implementation.

## Development Workflow

### KernelForge Agent Loop (Triton-first)

This repo now includes a minimal search loop under `kernelforge/`:

```bash
python engine.py --mode local --iters 10 --iterations 20 --num-trials 1
```

Core behavior:

- Retrieves top-k local patterns from the JSONL stores under `kernelforge/rag_store/`
  (`optimization_patterns.jsonl`, `error_fixes.jsonl`, `hardware_constraints.jsonl`,
  `knob_definitions.jsonl`, `research_insights.jsonl`) plus prior runs
- Uses a seed-first Tritonization path: the first Triton step is constrained to a vetted GEMM1-only micro-kernel template while routing/SwiGLU/GEMM2 stay in PyTorch
- Applies 1-2 constrained mutations (diff-based edits only)
- Packs and benchmarks through existing starter-kit scripts
- Caches benchmark outputs by kernel hash + benchmark flags
- Preserves the baseline `kernel(...)` signature during validation so LLM edits do not silently break the entrypoint
- Logs full provenance to:
  - `kernelforge/rag_store/runs.jsonl`
  - `kernelforge/provenance/runs.jsonl`
  - `kernelforge/provenance/latest_summary.json`

Run on Modal B200:

```bash
python engine.py --mode modal --iters 10 --iterations 100 --num-trials 3
```

Recommended Modal flow: preflight first, then full benchmark:

1. Run preflight only (compile -> quick correctness -> sanitizer):

```bash
python - <<'PY'
import json
from kernelforge.benchmark import BenchSettings, PreflightSettings, run_preflight

settings = BenchSettings(
    mode="modal",
    warmup_runs=1,
    iterations=1,
    num_trials=1,
    workload_focus=None,
)
preflight = PreflightSettings(
    enabled=True,
    quick_workloads=3,
    quick_warmup_runs=1,
    quick_iterations=1,
    quick_num_trials=1,
    run_sanitizer=True,
    sanitizer_workloads=1,
    sanitizer_types=("memcheck",),
)
out = run_preflight(settings, preflight)
print(json.dumps(out, indent=2))
raise SystemExit(0 if out.get("ok") else 1)
PY
```

2. If preflight passes, run full benchmark:

```bash
python - <<'PY'
import json
from kernelforge.benchmark import BenchSettings, run_benchmark

settings = BenchSettings(
    mode="modal",
    warmup_runs=3,
    iterations=100,
    num_trials=3,
    workload_focus=None,
)
out = run_benchmark(settings, phase="full")
print(json.dumps(out["summary"], indent=2))
PY
```

3. For optimization runs, keep preflight enabled so each iteration gates full benchmark:

```bash
python engine.py \
  --mode modal \
  --iters 10 \
  --warmup-runs 3 \
  --iterations 100 \
  --num-trials 3 \
  --preflight-quick-workloads 3 \
  --preflight-sanitizer-workloads 1 \
  --preflight-sanitizer-types memcheck
```

4. For correctness autofix loop (auto-repair until preflight passes, no full benchmark):

```bash
python engine.py \
  --mode modal \
  --iters 20 \
  --model claude-sonnet-4-5-20250929 \
  --warmup-runs 1 \
  --iterations 1 \
  --num-trials 1 \
  --preflight-quick-workloads 3 \
  --preflight-sanitizer-workloads 1 \
  --preflight-sanitizer-types memcheck \
  --autofix-correctness
```

Debugging failed autofix iterations:

- LLM prompts/responses and patches are saved under `kernelforge/provenance/llm_logs/iter_XXX/attempt_YY/`:
  - `generator_user_prompt.txt`
  - `generator_raw_response.txt`
  - `generator_patch.txt` (if patch mode)
  - `generated_kernel.py`
- Preflight failures now report workload-level details (workload id + max abs/rel error) in loop output.

Optional: focus scoring on one workload during fast iteration:

```bash
python engine.py --mode local --iters 20 --workload-id <workload_uuid>
```

### Pack Your Solution

Generate `solution.json` from your source files:

```bash
python scripts/pack_solution.py
```

### Run Local Benchmarks

Test your solution on your local GPU:

```bash
python scripts/run_local.py
```

Requires: Local CUDA-capable GPU and `FIB_DATASET_PATH` environment variable.

### Run Cloud Benchmarks (Modal)

Test your solution on NVIDIA B200 GPUs via Modal:

**One-time setup:**

```bash
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/mlsys26-contest /mlsys26-contest
```

**Quickest single-workload run (recommended first):**

```bash
modal run scripts/run_modal.py \
  --phase quick \
  --warmup-runs 1 \
  --iterations 1 \
  --num-trials 1 \
  --max-workloads 1 \
  --stop-on-error
```

**Compile-only smoke test:**

```bash
modal run scripts/run_modal.py --phase compile_only
```

If you uploaded the trace set somewhere else inside the Modal volume, pass
`--trace-set-path /data/<your-path>`.

## Submission

To submit your solution for evaluation:

1. Ensure your implementation is complete and tested
2. Run `python scripts/pack_solution.py` to generate `solution.json`
3. Commit and push your changes
4. Tag your commit for evaluation (e.g., `git tag submission-v1`)

## Project Structure

```
flashinfer-bench-starter-kit/
├── README.md                    # This file
├── config.toml                  # Track configuration (edit this)
├── solution/                    # Solution source files
│   ├── triton/                  # Triton implementation
│   │   └── kernel.py           # Your Triton kernel
│   └── cuda/                    # CUDA implementation
│       ├── kernel.cu           # Your CUDA kernel
│       └── binding.py          # TVM FFI bindings
├── scripts/                     # Utility scripts
│   ├── run_local.py            # Local benchmark runner
│   ├── run_modal.py            # Modal cloud benchmark runner
│   └── pack_solution.py        # Pack source files into solution.json
└── images/                      # Sponsor logos
```

## Additional Resources

### FlashInfer Trace Viewer

FlashInfer Trace consists of multiple JSON objects (definitions, workloads, solutions, and traces), which can contain large code blocks. To easily visualize and inspect these objects, you can use the [FlashInfer Trace Viewer](https://bench.flashinfer.ai/viewer). Simply paste any FlashInfer Trace JSON into the viewer to get a friendly, structured view of its contents.

### Solution Handling API

```python
from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files, extract_solution_to_files

# Pack source files into a Solution object
spec = BuildSpec(
    language="triton",  # or "cuda"
    target_hardware=["cuda"],
    entry_point="my_kernel",
)
solution = pack_solution_from_files(
    path="./my_solution_dir",
    spec=spec,
    name="my_solution_v1",
    definition="fused_moe",
    author="your_name",
)

# Extract a Solution to files in a working directory
extract_solution_to_files(solution, "./output_dir")
```

### Running Sanitizers

```python
from flashinfer_bench.agents import flashinfer_bench_run_sanitizer

output = flashinfer_bench_run_sanitizer(
    solution=solution,
    workload=workload,
    sanitizer_types=["memcheck", "racecheck", "synccheck", "initcheck"],
    timeout=300,
)
print(output)
```

### NCU Profiling

```python
from flashinfer_bench.agents import flashinfer_bench_run_ncu

output = flashinfer_bench_run_ncu(
    solution=solution,
    workload=workload,
    set="detailed",
    page="details",
    timeout=120,
)
print(output)
```

### List Available Tools

```python
from flashinfer_bench.agents import get_all_tool_schemas

schemas = get_all_tool_schemas()
# Returns list of OpenAI-compatible function schemas
```

## Notes

### Destination Passing Style (DPS)

FlashInfer-Bench uses destination passing style (DPS) by default, where both inputs and outputs are passed as function parameters. DPS avoids measuring tensor allocation overhead, resulting in more accurate performance numbers. We recommend using DPS when possible, as it yields better benchmark results.

**Important:** Avoid using variadic input arguments in your kernel signatures, as they will fail the builder validation check.

If your kernel uses value-returning style (i.e., returns output tensors instead of writing to pre-allocated ones), set `destination_passing_style` to `false` in your solution's `spec`:

```json
{
  "name": "my_solution",
  "definition": "gdn_decode_qk4_v8_d128_k_last",
  "author": "my_name",
  "spec": {
    "language": "triton",
    "target_hardware": ["cuda"],
    "entry_point": "kernel.py::my_kernel",
    "dependencies": [],
    "destination_passing_style": false
  },
  "sources": [...]
}
```

**Common error when DPS is mismatched:**

```
Destination-passing style callable: expected xx parameters, but got xx
```

This can happen for two reasons: (1) your kernel function signature has the wrong number of parameters, or (2) your kernel uses value-returning style but the solution still has `destination_passing_style` set to `true` by default. For the latter case, fix by setting `destination_passing_style` to `false`.

In this repo, `scripts/pack_solution.py` now infers `destination_passing_style`
from the `kernel(...)` return behavior when the config does not specify it, but
explicitly setting it in `config.toml` is still the safest option.

### CUDA Kernel Bindings

For CUDA kernel implementations, we recommend using [TVM FFI](https://tvm.apache.org/ffi/) for Python bindings. The `flashinfer_bench.agents` module provides TVM FFI agent instruction prompts to assist with development.

You can set the `binding` field in your solution's `spec` to specify the C++ binding type. Defaults to `"tvm-ffi"` if not specified. Supported values: `"tvm-ffi"`, `"torch"`.

```json
{
  "name": "my_cuda_solution",
  "definition": "gdn_decode_qk4_v8_d128_k_last",
  "author": "my_name",
  "spec": {
    "language": "cuda",
    "target_hardware": ["cuda"],
    "entry_point": "kernel.cu::my_kernel",
    "dependencies": [],
    "binding": "torch"
  },
  "sources": [...]
}
```

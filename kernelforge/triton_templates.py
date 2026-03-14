"""Vetted Triton micro-kernel templates for seed-first extraction."""

GEMM1_TEMPLATE_KERNEL_NAME = "_gemm1_tile_kernel"

GEMM1_ONLY_NOTES = """\
- Introduce exactly one `@triton.jit` kernel named `_gemm1_tile_kernel`.
- Only replace `_gemm1_subpath(A_e, W13_e)` with a Triton-backed implementation.
- `_gemm1_subpath` already receives dequantized `torch.float32` tensors, so the
  first Triton attempt should keep GEMM1 math in fp32 for correctness.
- Keep routing, SwiGLU, GEMM2, and final `accum.index_add_` in PyTorch.
- Do not introduce additional Triton kernels in the same attempt.
- Keep `def kernel(...)` unchanged.
"""

GEMM1_ONLY_TEMPLATE = '''\
@triton.jit
def _gemm1_tile_kernel(
    a_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k
        mask_k = k_offsets < K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        w_ptrs = w_ptr + k_offsets[:, None] * stride_wk + offs_n[None, :] * stride_wn

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )

        # Seed-stage correctness-first numerics: A_e and W13_e are already
        # dequantized fp32 tensors before entering this helper.
        acc += tl.dot(a, w, input_precision="ieee")

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _gemm1_subpath(A_e, W13_e):
    M, K = A_e.shape
    N = W13_e.shape[0]
    out = torch.empty((M, N), dtype=torch.float32, device=A_e.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    _gemm1_tile_kernel[grid](
        A_e, W13_e, out,
        M, N, K,
        A_e.stride(0), A_e.stride(1),
        W13_e.stride(0), W13_e.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return out
'''

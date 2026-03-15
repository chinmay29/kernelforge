"""Vetted Triton micro-kernel templates for seed-first and grouped-GEMM extraction."""

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

# ---------------------------------------------------------------------------
# Grouped FP8 GEMM template  (post-seed, full optimization target)
# ---------------------------------------------------------------------------

GROUPED_GEMM_FP8_NOTES = """\
Key design decisions:
- Sort token-expert pairs by expert ID → contiguous blocks per expert.
- Single persistent kernel replaces the per-expert Python loop.
- Native FP8 tl.dot (float8e4nv operands) — no pre-dequantization of all experts.
- Block scales applied to FP32 accumulator every 128 K-steps.
- GEMM2 output written to flat buffer [N_pairs, H]; PyTorch index_add_ scatters.
- grid=(NUM_SMS,) = (148,) on B200.
- Use these helper names exactly: `NUM_SMS`, `_sort_tokens`, `_grouped_gemm1_swiglu_kernel`,
  `_grouped_gemm2_kernel`, `sorted_ids`, `expert_starts`, `counts`, `tile_starts`.
- Do not fall back to legacy `gemm1_kernel` / `gemm2_kernel` or `expert_ids_per_block` dispatch.

Integration steps in `def kernel(...)`:
  1. Sort: flatten topk_ids → sort by local expert id → produce sorted_ids [N_pairs], expert_starts [E], counts [E], tile_starts [E+1].
  2. Launch _grouped_gemm1_swiglu_kernel[(NUM_SMS,)]: FP8 GEMM1 + SwiGLU fused, writes to flat_out [N_pairs, I].
  3. Launch _grouped_gemm2_kernel[(NUM_SMS,)]: FP8 GEMM2, writes to out_buf [N_pairs, H].
  4. Routing weight scatter: index_add_(0, sorted_tok_ids, out_buf * rw_sorted[:, None]).
  5. Return accum.to(torch.bfloat16).
"""

GROUPED_GEMM_FP8_TEMPLATE = '''\
# --- PyTorch sort helper (runs before kernel launch) ---
def _sort_tokens(topk_ids, local_start, E_LOCAL, device):
    """Returns sorted_ids, expert_starts, counts, tile_starts for grouped GEMM."""
    T, K = topk_ids.shape
    local_mask = (topk_ids >= local_start) & (topk_ids < local_start + E_LOCAL)
    local_le   = (topk_ids - local_start).clamp(min=0)  # 0-indexed local expert
    flat_tok  = torch.arange(T, device=device).unsqueeze(1).expand(T, K).reshape(-1)
    flat_le   = local_le.reshape(-1)
    flat_valid = local_mask.reshape(-1)
    flat_tok_v = flat_tok[flat_valid].to(torch.int32)
    flat_le_v  = flat_le[flat_valid].to(torch.int32)
    order      = flat_le_v.argsort(stable=True)
    sorted_ids = flat_tok_v[order]          # [N_pairs] token indices in expert order
    sorted_le  = flat_le_v[order]
    counts     = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    counts.scatter_add_(0, sorted_le, torch.ones_like(sorted_le))
    expert_ends   = counts.cumsum(0)
    expert_starts = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    expert_starts[1:] = expert_ends[:-1]
    BLOCK_M = 128
    num_tiles   = (counts + BLOCK_M - 1) // BLOCK_M
    tile_starts = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=device)
    tile_starts[1:] = num_tiles.cumsum(0)
    return sorted_ids, expert_starts, counts, tile_starts


# --- Persistent grouped GEMM1 + SwiGLU (FP8 native) ---
@triton.jit
def _grouped_gemm1_swiglu_kernel(
    A_ptr, A_scale_ptr,          # [T, H] fp8, [H//128, T] fp32
    sorted_ids_ptr,              # [N_pairs] int32
    W13_ptr, W13_scale_ptr,      # [E, 2I, H] fp8, [E, 2I//128, H//128] fp32
    expert_starts_ptr,           # [E] int32
    counts_ptr,                  # [E] int32
    tile_starts_ptr,             # [E+1] int32
    total_tiles,
    Out_ptr,                     # [N_pairs, I] fp32  (SwiGLU output)
    T, E: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    Q_BLOCK: tl.constexpr,       # quantisation block size = 128
):
    sm_id    = tl.program_id(0)
    num_sms  = tl.num_programs(0)
    tile_id  = sm_id
    while tile_id < total_tiles:
        # binary search: expert owning this tile
        lo, hi = 0, E - 1
        for _ in range(6):
            mid = (lo + hi) // 2
            ts  = tl.load(tile_starts_ptr + mid)
            te  = tl.load(tile_starts_ptr + mid + 1)
            lo  = tl.where(tile_id >= te, mid + 1, lo)
            hi  = tl.where(tile_id < ts,  mid - 1, hi)
            lo  = tl.where((tile_id >= ts) & (tile_id < te), mid, lo)
            hi  = tl.where((tile_id >= ts) & (tile_id < te), mid, hi)
        expert_id       = lo
        tile_in_expert  = tile_id - tl.load(tile_starts_ptr + expert_id)
        expert_start    = tl.load(expert_starts_ptr + expert_id)
        expert_count    = tl.load(counts_ptr + expert_id)

        offs_m = tile_in_expert * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < expert_count
        tok_ids = tl.load(sorted_ids_ptr + expert_start + offs_m, mask=mask_m, other=0)

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        W13_e_ptr       = W13_ptr       + expert_id * (2 * I * H)
        W13_scale_e_ptr = W13_scale_ptr + expert_id * (2 * I // Q_BLOCK) * (H // Q_BLOCK)

        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H
            # Load activation tile: [BLOCK_M, BLOCK_K] fp8
            a_tile = tl.load(
                A_ptr + tok_ids[:, None] * H + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float8e4nv)
            # Gate rows [0..BLOCK_N)
            offs_gate = tl.arange(0, BLOCK_N)
            w_gate = tl.load(
                W13_e_ptr + offs_gate[:, None] * H + offs_k[None, :],
                mask=(offs_gate[:, None] < I) & mask_k[None, :], other=0.0,
            ).to(tl.float8e4nv)
            # Up rows [I..I+BLOCK_N)
            offs_up = I + tl.arange(0, BLOCK_N)
            w_up = tl.load(
                W13_e_ptr + offs_up[:, None] * H + offs_k[None, :],
                mask=(offs_up[:, None] < 2 * I) & mask_k[None, :], other=0.0,
            ).to(tl.float8e4nv)
            acc_gate = tl.dot(a_tile, tl.trans(w_gate), acc_gate)
            acc_up   = tl.dot(a_tile, tl.trans(w_up),   acc_up)
            # Apply block scales every Q_BLOCK K-steps
            if (k_start + BLOCK_K) % Q_BLOCK == 0:
                k_blk   = k_start // Q_BLOCK
                a_scale = tl.load(A_scale_ptr + k_blk * T + tok_ids,
                                  mask=mask_m, other=1.0)
                w_scale_gate = tl.load(
                    W13_scale_e_ptr + (tl.arange(0, BLOCK_N) // Q_BLOCK) * (H // Q_BLOCK) + k_blk,
                    mask=offs_gate < I, other=1.0)
                w_scale_up = tl.load(
                    W13_scale_e_ptr + ((I + tl.arange(0, BLOCK_N)) // Q_BLOCK) * (H // Q_BLOCK) + k_blk,
                    mask=offs_up < 2 * I, other=1.0)
                acc_gate = acc_gate * a_scale[:, None] * w_scale_gate[None, :]
                acc_up   = acc_up   * a_scale[:, None] * w_scale_up[None, :]

        # SwiGLU: silu(gate) * up  (fused, in FP32 registers)
        swiglu = (acc_gate * tl.sigmoid(acc_gate)) * acc_up

        offs_n_out = tl.arange(0, BLOCK_N)
        out_pos    = expert_start + tile_in_expert * BLOCK_M + offs_m
        tl.store(
            Out_ptr + out_pos[:, None] * I + offs_n_out[None, :],
            swiglu,
            mask=mask_m[:, None] & (offs_n_out[None, :] < I),
        )
        tile_id += num_sms
'''


# ---------------------------------------------------------------------------
# Seed-stage GEMM1-only template (used in first Triton extraction iteration)
# ---------------------------------------------------------------------------

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

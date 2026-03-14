"""
Fused MoE Seed Kernel — DeepSeek-V3 / R1 architecture.

FP8 block-scale MoE: routing (sigmoid + grouped top-k) → GEMM1 → SwiGLU → GEMM2
with per-token weighted accumulation.

This is a correct-first PyTorch implementation that serves as the seed for
KernelForge's optimization loop.  Performance will be improved iteratively
by the mutation engine.

LLM edit guardrails:
- Preserve the exact `kernel(...)` signature.
- This seed kernel is value-returning; `destination_passing_style = false`.
- Keep routing semantics and geometry constants intact; optimize one region at a time.
"""

import torch
import triton
import triton.language as tl

BLOCK_M = 128

DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 3

# ----- Constants (DeepSeek-V3 / R1 geometry) -----
H = 7168             # hidden_size
I = 2048             # intermediate_size
E_GLOBAL = 256       # total experts
E_LOCAL = 32         # experts per rank (EP=8)
BLOCK = 128          # quantization block size
TOP_K = 8            # experts per token
N_GROUP = 8          # routing groups
TOPK_GROUP = 4       # groups kept
GROUP_SIZE = E_GLOBAL // N_GROUP  # 32


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
    # Seed-stage Tritonization point:
    # only replace this helper first, and keep routing/SwiGLU/GEMM2/index_add_
    # in PyTorch until the micro-kernel is correct and benchmarked.
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


def kernel(
    routing_logits,         # [T, 256]  float32
    routing_bias,           # [256]     bfloat16
    hidden_states,          # [T, 7168] float8_e4m3fn
    hidden_states_scale,    # [56, T]   float32
    gemm1_weights,          # [32, 4096, 7168] float8_e4m3fn
    gemm1_weights_scale,    # [32, 32, 56]     float32
    gemm2_weights,          # [32, 7168, 2048]  float8_e4m3fn
    gemm2_weights_scale,    # [32, 56, 16]      float32
    local_expert_offset,    # int32 scalar
    routed_scaling_factor,  # float32 scalar
):
    T = routing_logits.shape[0]
    device = hidden_states.device
    # Robust scalar extraction
    if isinstance(local_expert_offset, torch.Tensor):
        local_start = local_expert_offset.item()
    else:
        local_start = int(local_expert_offset)

    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = routed_scaling_factor.item()
    else:
        routed_scaling_factor = float(routed_scaling_factor)

    # ----------------------------------------------------------------
    # 1) FP8 block-scale dequantisation  (block_size = 128)
    # ----------------------------------------------------------------
    # hidden_states: [T, H], scale: [H/128, T]  (transposed block layout)
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)          # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()          # [T, H/128]
    A_scale_expanded = (
        A_scale_TH
        .unsqueeze(-1)
        .expand(T, H // BLOCK, BLOCK)
        .reshape(T, H)
    )
    A = A_fp32 * A_scale_expanded                            # [T, H]

    # W13: [E_local, 2I, H]  scale: [E_local, 2I/128, H/128]
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_exp = (
        S13
        .unsqueeze(-1).expand(*S13.shape, BLOCK)             # repeat along H/128 → H
        .reshape(E_LOCAL, S13.shape[1], H)
    )
    S13_exp = (
        S13_exp
        .unsqueeze(2).expand(E_LOCAL, S13.shape[1], BLOCK, H)
        .reshape(E_LOCAL, 2 * I, H)
    )
    # Simpler: use repeat_interleave (clearer, same result)
    S13_r = torch.repeat_interleave(S13, BLOCK, dim=1)       # [E, 2I, H/128]
    S13_r = torch.repeat_interleave(S13_r, BLOCK, dim=2)     # [E, 2I, H]
    W13 = W13_fp32 * S13_r                                   # [E, 2I, H]

    # W2: [E_local, H, I]  scale: [E_local, H/128, I/128]
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_r = torch.repeat_interleave(S2, BLOCK, dim=1)         # [E, H, I/128]
    S2_r = torch.repeat_interleave(S2_r, BLOCK, dim=2)       # [E, H, I]
    W2 = W2_fp32 * S2_r                                      # [E, H, I]

    # ----------------------------------------------------------------
    # 2) DeepSeek-V3 no-aux routing
    # ----------------------------------------------------------------
    logits = routing_logits.to(torch.float32)                 # [T, E_GLOBAL]
    bias = routing_bias.to(torch.float32).reshape(-1)         # [E_GLOBAL]

    s = torch.sigmoid(logits)                                 # [T, E]
    s_with_bias = s + bias                                    # [T, E]

    # Group → top-2 per group → sum → pick top-4 groups
    s_wb_g = s_with_bias.view(T, N_GROUP, GROUP_SIZE)         # [T, 8, 32]
    top2_vals, _ = torch.topk(s_wb_g, k=2, dim=2,
                              largest=True, sorted=False)     # [T, 8, 2]
    group_scores = top2_vals.sum(dim=2)                       # [T, 8]

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP,
                              dim=1, largest=True,
                              sorted=False)                   # [T, 4]
    group_mask = torch.zeros_like(group_scores)               # [T, 8]
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask
        .unsqueeze(2)
        .expand(T, N_GROUP, GROUP_SIZE)
        .reshape(T, E_GLOBAL)
    )

    # Global top-k within kept groups
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1,
                             largest=True, sorted=False)      # [T, 8]

    # Combination weights (use s *without* bias, normalise, scale)
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor # [T, E]

    # ----------------------------------------------------------------
    # 3) Per-expert GEMM1 → SwiGLU → GEMM2 → weighted accumulate
    # ----------------------------------------------------------------
    accum = torch.zeros((T, H), dtype=torch.float32, device=device)

    for le in range(E_LOCAL):
        ge = local_start + le
        if ge < 0 or ge >= E_GLOBAL:
            continue

        sel = (topk_idx == ge).any(dim=1)                     # [T] bool
        if not sel.any():
            continue

        token_idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
        if token_idx.numel() == 0:
            continue

        A_e = A.index_select(0, token_idx)                    # [Tk, H]
        W13_e = W13[le]                                       # [2I, H]
        W2_e = W2[le]                                         # [H, I]

        # GEMM1 extraction point for incremental Tritonization.
        G1 = _gemm1_subpath(A_e, W13_e)

        # SwiGLU
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        C = torch.nn.functional.silu(X2) * X1                # [Tk, I]

        # GEMM2:  [Tk, I] @ [I, H] = [Tk, H]
        O = C.matmul(W2_e.t())

        # Weighted accumulation
        w_tok = weights.index_select(0, token_idx)[:, ge]     # [Tk]
        accum.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return accum.to(torch.bfloat16)

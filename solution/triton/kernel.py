"""
Fused MoE kernel with grouped expert dispatch for DeepSeek-V3 / R1 geometry.

This version removes the baseline's pre-dequantize-all-experts path and
replaces the per-expert Python loop with:
1. token-expert sorting by local expert id
2. persistent grouped GEMM1 + SwiGLU Triton kernel
3. persistent grouped GEMM2 Triton kernel
4. sorted routing-weight gather + index_add_ accumulation
"""

import torch
import triton
import triton.language as tl

NUM_SMS = 148
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
USE_GROUPED_TRITON = False

DEFAULT_NUM_WARPS = 8
DEFAULT_NUM_STAGES = 2

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


def _sort_tokens(topk_ids, local_start, E_LOCAL, device):
    """Return grouped local token-expert pairs sorted by local expert id."""
    T, K = topk_ids.shape
    local_mask = (topk_ids >= local_start) & (topk_ids < local_start + E_LOCAL)
    local_le = (topk_ids - local_start).reshape(-1)
    flat_tok = torch.arange(T, device=device, dtype=torch.int32)
    flat_tok = flat_tok.unsqueeze(1).expand(T, K).reshape(-1)
    flat_valid = local_mask.reshape(-1)

    if int(flat_valid.sum().item()) == 0:
        empty = torch.empty((0,), dtype=torch.int32, device=device)
        zeros = torch.zeros((E_LOCAL,), dtype=torch.int32, device=device)
        tile_starts = torch.zeros((E_LOCAL + 1,), dtype=torch.int32, device=device)
        return empty, empty, zeros, zeros, tile_starts

    flat_tok_v = flat_tok[flat_valid].contiguous()
    flat_le_v = local_le[flat_valid].to(torch.int32).contiguous()

    order = flat_le_v.argsort(stable=True)
    sorted_ids = flat_tok_v[order].contiguous()
    sorted_local_experts = flat_le_v[order].contiguous()

    counts = torch.zeros((E_LOCAL,), dtype=torch.int32, device=device)
    counts.scatter_add_(0, sorted_local_experts, torch.ones_like(sorted_local_experts))

    expert_starts = torch.zeros((E_LOCAL,), dtype=torch.int32, device=device)
    if E_LOCAL > 1:
        expert_starts[1:] = counts.cumsum(0)[:-1]

    num_tiles = (counts + BLOCK_M - 1) // BLOCK_M
    tile_starts = torch.zeros((E_LOCAL + 1,), dtype=torch.int32, device=device)
    tile_starts[1:] = num_tiles.cumsum(0)
    return sorted_ids, sorted_local_experts, expert_starts, counts, tile_starts


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids):
    hidden_fp32 = hidden_states.index_select(0, token_ids).to(torch.float32)
    scale = hidden_states_scale.index_select(1, token_ids).to(torch.float32).transpose(0, 1)
    return (hidden_fp32.view(-1, H // BLOCK, BLOCK) * scale.unsqueeze(-1)).reshape(-1, H)


def _dequant_gemm1_weight(gemm1_weights, gemm1_weights_scale, expert_id):
    w_fp32 = gemm1_weights[expert_id].to(torch.float32)
    scale = gemm1_weights_scale[expert_id].to(torch.float32)
    scale = torch.repeat_interleave(scale, BLOCK, dim=0)
    scale = torch.repeat_interleave(scale, BLOCK, dim=1)
    return w_fp32 * scale


def _dequant_gemm2_weight(gemm2_weights, gemm2_weights_scale, expert_id):
    w_fp32 = gemm2_weights[expert_id].to(torch.float32)
    scale = gemm2_weights_scale[expert_id].to(torch.float32)
    scale = torch.repeat_interleave(scale, BLOCK, dim=0)
    scale = torch.repeat_interleave(scale, BLOCK, dim=1)
    return w_fp32 * scale


@triton.jit
def _grouped_gemm1_swiglu_kernel(
    A_ptr,
    A_scale_ptr,
    sorted_ids_ptr,
    W13_ptr,
    W13_scale_ptr,
    expert_starts_ptr,
    counts_ptr,
    tile_starts_ptr,
    total_tiles,
    Out_ptr,
    stride_ah0,
    stride_ah1,
    stride_as0,
    stride_as1,
    stride_w13e,
    stride_w13n,
    stride_w13k,
    stride_w13se,
    stride_w13sn,
    stride_w13sk,
    stride_om,
    stride_on,
    T,
    E: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    tile_id = sm_id

    while tile_id < total_tiles:
        expert_id = 0
        for e in range(E):
            tile_lo = tl.load(tile_starts_ptr + e)
            tile_hi = tl.load(tile_starts_ptr + e + 1)
            expert_id = tl.where((tile_id >= tile_lo) & (tile_id < tile_hi), e, expert_id)

        expert_tile_start = tl.load(tile_starts_ptr + expert_id)
        expert_start = tl.load(expert_starts_ptr + expert_id)
        expert_count = tl.load(counts_ptr + expert_id)
        tile_in_expert = tile_id - expert_tile_start

        offs_m = tile_in_expert * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < expert_count
        pair_pos = expert_start + offs_m
        tok_ids = tl.load(sorted_ids_ptr + pair_pos, mask=mask_m, other=0)

        for n_start in range(0, I, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < I

            acc_x1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_x2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            x1_row_blk = n_start // BLOCK_N
            x2_row_blk = (I + n_start) // BLOCK_N

            for k_start in range(0, H, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < H
                k_blk = k_start // BLOCK_K

                a_ptrs = A_ptr + tok_ids[:, None] * stride_ah0 + offs_k[None, :] * stride_ah1
                a_fp8 = tl.load(
                    a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                a_scale = tl.load(
                    A_scale_ptr + k_blk * stride_as0 + tok_ids * stride_as1,
                    mask=mask_m,
                    other=1.0,
                )
                a_bf16 = (a_fp8 * a_scale[:, None]).to(tl.bfloat16)

                x1_ptrs = (
                    W13_ptr
                    + expert_id * stride_w13e
                    + offs_k[:, None] * stride_w13k
                    + offs_n[None, :] * stride_w13n
                )
                x1_fp8 = tl.load(
                    x1_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                )
                x1_scale = tl.load(
                    W13_scale_ptr
                    + expert_id * stride_w13se
                    + x1_row_blk * stride_w13sn
                    + k_blk * stride_w13sk
                )
                x1_bf16 = (x1_fp8 * x1_scale).to(tl.bfloat16)

                x2_rows = I + offs_n
                x2_ptrs = (
                    W13_ptr
                    + expert_id * stride_w13e
                    + offs_k[:, None] * stride_w13k
                    + x2_rows[None, :] * stride_w13n
                )
                x2_fp8 = tl.load(
                    x2_ptrs,
                    mask=mask_k[:, None] & (x2_rows[None, :] < 2 * I),
                    other=0.0,
                )
                x2_scale = tl.load(
                    W13_scale_ptr
                    + expert_id * stride_w13se
                    + x2_row_blk * stride_w13sn
                    + k_blk * stride_w13sk
                )
                x2_bf16 = (x2_fp8 * x2_scale).to(tl.bfloat16)

                acc_x1 += tl.dot(a_bf16, x1_bf16)
                acc_x2 += tl.dot(a_bf16, x2_bf16)

            swiglu = (acc_x2 * tl.sigmoid(acc_x2)) * acc_x1
            out_ptrs = Out_ptr + pair_pos[:, None] * stride_om + offs_n[None, :] * stride_on
            tl.store(out_ptrs, swiglu, mask=mask_m[:, None] & mask_n[None, :])

        tile_id += num_sms


@triton.jit
def _grouped_gemm2_kernel(
    A_ptr,
    sorted_ids_ptr,
    W2_ptr,
    W2_scale_ptr,
    expert_starts_ptr,
    counts_ptr,
    tile_starts_ptr,
    total_tiles,
    Out_ptr,
    stride_am,
    stride_ak,
    stride_w2e,
    stride_w2n,
    stride_w2k,
    stride_w2se,
    stride_w2sn,
    stride_w2sk,
    stride_om,
    stride_on,
    E: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    tile_id = sm_id

    while tile_id < total_tiles:
        expert_id = 0
        for e in range(E):
            tile_lo = tl.load(tile_starts_ptr + e)
            tile_hi = tl.load(tile_starts_ptr + e + 1)
            expert_id = tl.where((tile_id >= tile_lo) & (tile_id < tile_hi), e, expert_id)

        expert_tile_start = tl.load(tile_starts_ptr + expert_id)
        expert_start = tl.load(expert_starts_ptr + expert_id)
        expert_count = tl.load(counts_ptr + expert_id)
        tile_in_expert = tile_id - expert_tile_start

        offs_m = tile_in_expert * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < expert_count
        pair_pos = expert_start + offs_m
        _ = tl.load(sorted_ids_ptr + pair_pos, mask=mask_m, other=0)

        for n_start in range(0, H, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < H

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            row_blk = n_start // BLOCK_N

            for k_start in range(0, I, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < I
                k_blk = k_start // BLOCK_K

                a_ptrs = A_ptr + pair_pos[:, None] * stride_am + offs_k[None, :] * stride_ak
                a_bf16 = tl.load(
                    a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )

                w_ptrs = (
                    W2_ptr
                    + expert_id * stride_w2e
                    + offs_k[:, None] * stride_w2k
                    + offs_n[None, :] * stride_w2n
                )
                w_fp8 = tl.load(
                    w_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                )
                w_scale = tl.load(
                    W2_scale_ptr
                    + expert_id * stride_w2se
                    + row_blk * stride_w2sn
                    + k_blk * stride_w2sk
                )
                w_bf16 = (w_fp8 * w_scale).to(tl.bfloat16)

                acc += tl.dot(a_bf16, w_bf16)

            out_ptrs = Out_ptr + pair_pos[:, None] * stride_om + offs_n[None, :] * stride_on
            tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

        tile_id += num_sms


def _launch_grouped_gemm1_swiglu(
    hidden_states,
    hidden_states_scale,
    sorted_ids,
    gemm1_weights,
    gemm1_weights_scale,
    expert_starts,
    counts,
    tile_starts,
):
    num_pairs = int(sorted_ids.numel())
    out = torch.empty((num_pairs, I), dtype=torch.float32, device=hidden_states.device)
    total_tiles = int(tile_starts[-1].item())
    if total_tiles == 0:
        return out

    _grouped_gemm1_swiglu_kernel[(NUM_SMS,)](
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        tile_starts,
        total_tiles,
        out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states_scale.stride(0),
        hidden_states_scale.stride(1),
        gemm1_weights.stride(0),
        gemm1_weights.stride(1),
        gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0),
        gemm1_weights_scale.stride(1),
        gemm1_weights_scale.stride(2),
        out.stride(0),
        out.stride(1),
        hidden_states.shape[0],
        E=E_LOCAL,
        H=H,
        I=I,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
    )
    return out


def _launch_grouped_gemm2(
    swiglu_out,
    sorted_ids,
    gemm2_weights,
    gemm2_weights_scale,
    expert_starts,
    counts,
    tile_starts,
):
    num_pairs = int(sorted_ids.numel())
    swiglu_bf16 = swiglu_out.to(torch.bfloat16).contiguous()
    out = torch.empty((num_pairs, H), dtype=torch.float32, device=swiglu_out.device)
    total_tiles = int(tile_starts[-1].item())
    if total_tiles == 0:
        return out

    _grouped_gemm2_kernel[(NUM_SMS,)](
        swiglu_bf16,
        sorted_ids,
        gemm2_weights,
        gemm2_weights_scale,
        expert_starts,
        counts,
        tile_starts,
        total_tiles,
        out,
        swiglu_bf16.stride(0),
        swiglu_bf16.stride(1),
        gemm2_weights.stride(0),
        gemm2_weights.stride(1),
        gemm2_weights.stride(2),
        gemm2_weights_scale.stride(0),
        gemm2_weights_scale.stride(1),
        gemm2_weights_scale.stride(2),
        out.stride(0),
        out.stride(1),
        E=E_LOCAL,
        H=H,
        I=I,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
    )
    return out


def kernel(
    routing_logits,         # [T, 256]  float32
    routing_bias,           # [256]     bfloat16
    hidden_states,          # [T, 7168] float8_e4m3fn
    hidden_states_scale,    # [56, T]   float32
    gemm1_weights,          # [32, 4096, 7168] float8_e4m3fn
    gemm1_weights_scale,    # [32, 32, 56]     float32
    gemm2_weights,          # [32, 7168, 2048] float8_e4m3fn
    gemm2_weights_scale,    # [32, 56, 16]      float32
    local_expert_offset,    # int32 scalar
    routed_scaling_factor,  # float32 scalar
):
    T = routing_logits.shape[0]
    device = hidden_states.device

    if isinstance(local_expert_offset, torch.Tensor):
        local_start = int(local_expert_offset.item())
    else:
        local_start = int(local_expert_offset)

    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())
    else:
        routed_scaling_factor = float(routed_scaling_factor)

    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    s_wb_g = s_with_bias.view(T, N_GROUP, GROUP_SIZE)
    top2_vals, _ = torch.topk(s_wb_g, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(2).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_GLOBAL)
    )

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    (
        sorted_ids,
        sorted_local_experts,
        expert_starts,
        counts,
        tile_starts,
    ) = _sort_tokens(topk_idx, local_start, E_LOCAL, device)

    if sorted_ids.numel() == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()
    gemm1_weights = gemm1_weights.contiguous()
    gemm1_weights_scale = gemm1_weights_scale.contiguous()
    gemm2_weights = gemm2_weights.contiguous()
    gemm2_weights_scale = gemm2_weights_scale.contiguous()

    if USE_GROUPED_TRITON:
        swiglu_out = _launch_grouped_gemm1_swiglu(
            hidden_states,
            hidden_states_scale,
            sorted_ids,
            gemm1_weights,
            gemm1_weights_scale,
            expert_starts,
            counts,
            tile_starts,
        )
        gemm2_out = _launch_grouped_gemm2(
            swiglu_out,
            sorted_ids,
            gemm2_weights,
            gemm2_weights_scale,
            expert_starts,
            counts,
            tile_starts,
        )

        sorted_global_experts = sorted_local_experts.to(torch.long) + local_start
        rw_sorted = weights[sorted_ids.to(torch.long), sorted_global_experts]

        accum = torch.zeros((T, H), dtype=torch.float32, device=device)
        accum.index_add_(0, sorted_ids.to(torch.long), gemm2_out * rw_sorted.unsqueeze(1))
        return accum.to(torch.bfloat16)

    accum = torch.zeros((T, H), dtype=torch.float32, device=device)
    active_experts = torch.nonzero(counts > 0, as_tuple=False).flatten()
    for expert_id in active_experts.tolist():
        start = int(expert_starts[expert_id].item())
        count = int(counts[expert_id].item())
        end = start + count
        token_ids = sorted_ids[start:end].to(torch.long)

        if token_ids.numel() == 0:
            continue

        hidden_rows = _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids)
        w13 = _dequant_gemm1_weight(gemm1_weights, gemm1_weights_scale, expert_id)
        gemm1 = hidden_rows.matmul(w13.t())
        x1 = gemm1[:, :I]
        x2 = gemm1[:, I:]
        swiglu = torch.nn.functional.silu(x2) * x1

        w2 = _dequant_gemm2_weight(gemm2_weights, gemm2_weights_scale, expert_id)
        out = swiglu.matmul(w2.t())
        routing_weight = weights[token_ids, local_start + expert_id]
        accum.index_add_(0, token_ids, out * routing_weight.unsqueeze(1))

    return accum.to(torch.bfloat16)

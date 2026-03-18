"""
Fused MoE kernel with grouped expert dispatch for DeepSeek-V3 / R1 geometry.

This version removes the baseline's pre-dequantize-all-experts path and
replaces the per-expert Python loop with:
1. token-expert sorting by local expert id
2. persistent grouped GEMM1 + SwiGLU Triton kernel
3. persistent grouped GEMM2 Triton kernel
4. sorted routing-weight gather + index_add_ accumulation
"""

import os
import torch
import triton
import triton.language as tl

NUM_SMS = 148
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
DOT_BLOCK_K = 64
GEMM2_A_SCALE_BLOCK = DOT_BLOCK_K
GROUPED_TILE_ORDER_GROUP = 8
USE_GROUPED_TRITON = os.environ.get("KF_USE_GROUPED_TRITON", "1") == "1"
USE_GROUPED_TRITON_GEMM1_ONLY = os.environ.get("KF_USE_GROUPED_TRITON_GEMM1_ONLY", "0") == "1"
USE_NATIVE_FP8_GEMM1 = os.environ.get("KF_USE_NATIVE_FP8_GEMM1", "1") == "1"
USE_NATIVE_FP8_GEMM2 = os.environ.get("KF_USE_NATIVE_FP8_GEMM2", "0") == "1"
USE_BF16_GEMM2 = os.environ.get("KF_USE_BF16_GEMM2", "0") == "1"
USE_GROUPED_TILE_ORDER = os.environ.get("KF_USE_GROUPED_TILE_ORDER", "1") == "1"

DEFAULT_NUM_WARPS = int(os.environ.get("KF_NUM_WARPS", "8"))
DEFAULT_NUM_STAGES = int(os.environ.get("KF_NUM_STAGES", "4"))

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
        return empty, empty, zeros, zeros, tile_starts, empty

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
    tile_to_expert = torch.repeat_interleave(
        torch.arange(E_LOCAL, dtype=torch.int32, device=device),
        num_tiles.to(torch.long),
    ).contiguous()
    return sorted_ids, sorted_local_experts, expert_starts, counts, tile_starts, tile_to_expert


def _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids, bypass_scales=False):
    hidden_fp32 = hidden_states.index_select(0, token_ids).to(torch.float32)
    if bypass_scales:
        return hidden_fp32
    scale = hidden_states_scale.index_select(1, token_ids).to(torch.float32).transpose(0, 1)
    return (hidden_fp32.view(-1, H // BLOCK, BLOCK) * scale.unsqueeze(-1)).reshape(-1, H)


def _dequant_gemm1_weight(gemm1_weights, gemm1_weights_scale, expert_id, bypass_scales=False):
    w_fp32 = gemm1_weights[expert_id].to(torch.float32)
    if bypass_scales:
        return w_fp32
    scale = gemm1_weights_scale[expert_id].to(torch.float32)
    scale = torch.repeat_interleave(scale, BLOCK, dim=0)
    scale = torch.repeat_interleave(scale, BLOCK, dim=1)
    return w_fp32 * scale


def _dequant_gemm1_weight_transposed_scale(
    gemm1_weights,
    gemm1_weights_scale,
    expert_id,
    bypass_scales=False,
):
    w_fp32 = gemm1_weights[expert_id].to(torch.float32)
    if bypass_scales:
        return w_fp32
    scale = gemm1_weights_scale[expert_id].to(torch.float32).transpose(0, 1)
    scale = torch.repeat_interleave(scale, BLOCK, dim=0)
    scale = torch.repeat_interleave(scale, BLOCK, dim=1)
    return w_fp32 * scale.transpose(0, 1)


def _dequant_gemm2_weight(gemm2_weights, gemm2_weights_scale, expert_id):
    w_fp32 = gemm2_weights[expert_id].to(torch.float32)
    scale = gemm2_weights_scale[expert_id].to(torch.float32)
    scale = torch.repeat_interleave(scale, BLOCK, dim=0)
    scale = torch.repeat_interleave(scale, BLOCK, dim=1)
    return w_fp32 * scale


def _quantize_fp8_block_rows(x_fp32, block_size=BLOCK):
    num_rows, width = x_fp32.shape
    if num_rows == 0:
        q = torch.empty_like(x_fp32, dtype=torch.float8_e4m3fn)
        scale = torch.empty((width // block_size, 0), dtype=torch.float32, device=x_fp32.device)
        return q, scale

    x_blocks = x_fp32.view(num_rows, width // block_size, block_size)
    absmax = x_blocks.abs().amax(dim=2)
    scale = torch.where(
        absmax > 0,
        absmax / torch.finfo(torch.float8_e4m3fn).max,
        torch.ones_like(absmax),
    )
    q = (x_blocks / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(num_rows, width)
    return q.contiguous(), scale.transpose(0, 1).contiguous()


def _dequant_fp8_block_rows(x_fp8, x_scale, block_size=BLOCK):
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.float32)
    scale = x_scale.transpose(0, 1).to(torch.float32)
    return (x_fp8.to(torch.float32).view(-1, x_fp8.shape[1] // block_size, block_size) * scale.unsqueeze(-1)).reshape(
        x_fp8.shape[0], x_fp8.shape[1]
    )


def _build_launch_tile_metadata(counts, tile_starts, tile_to_expert):
    total_tiles = int(tile_to_expert.numel())
    if total_tiles == 0:
        empty = torch.empty((0,), dtype=torch.int32, device=counts.device)
        return empty, empty

    if (not USE_GROUPED_TILE_ORDER) or total_tiles <= NUM_SMS:
        launch_tile_experts = tile_to_expert
        launch_tile_offsets = (
            torch.arange(total_tiles, dtype=torch.int32, device=counts.device)
            - tile_starts.index_select(0, launch_tile_experts.to(torch.long))
        ).to(torch.int32)
        return launch_tile_experts, launch_tile_offsets

    num_tiles = (counts + BLOCK_M - 1) // BLOCK_M
    max_tiles = int(num_tiles.max().item())
    if max_tiles == 0:
        empty = torch.empty((0,), dtype=torch.int32, device=counts.device)
        return empty, empty

    group = GROUPED_TILE_ORDER_GROUP
    num_chunks = (max_tiles + group - 1) // group
    padded_tiles = num_chunks * group

    expert_order = torch.argsort(num_tiles.to(torch.int64), descending=True, stable=True)
    tile_offsets = torch.arange(padded_tiles, dtype=torch.int32, device=counts.device)
    tile_offsets = tile_offsets.unsqueeze(0).expand(E_LOCAL, padded_tiles)
    valid = tile_offsets < num_tiles.unsqueeze(1)

    tile_offsets = tile_offsets.index_select(0, expert_order)
    valid = valid.index_select(0, expert_order)
    launch_tile_experts = expert_order.unsqueeze(1).expand(E_LOCAL, padded_tiles)

    launch_tile_experts = (
        launch_tile_experts.view(E_LOCAL, num_chunks, group)
        .permute(1, 0, 2)
        .reshape(-1)
    )
    launch_tile_offsets = (
        tile_offsets.view(E_LOCAL, num_chunks, group)
        .permute(1, 0, 2)
        .reshape(-1)
    )
    valid = valid.view(E_LOCAL, num_chunks, group).permute(1, 0, 2).reshape(-1)

    return (
        launch_tile_experts[valid].to(torch.int32).contiguous(),
        launch_tile_offsets[valid].to(torch.int32).contiguous(),
    )


@triton.jit
def _grouped_gemm1_swiglu_kernel(
    A_ptr,
    A_scale_ptr,
    sorted_ids_ptr,
    W13_ptr,
    W13_scale_ptr,
    expert_starts_ptr,
    counts_ptr,
    launch_tile_experts_ptr,
    launch_tile_offsets_ptr,
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
    DOT_BLOCK_K: tl.constexpr,
    USE_NATIVE_FP8_DOT: tl.constexpr,
):
    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    tile_id = sm_id

    while tile_id < total_tiles:
        expert_id = tl.load(launch_tile_experts_ptr + tile_id)
        expert_start = tl.load(expert_starts_ptr + expert_id)
        expert_count = tl.load(counts_ptr + expert_id)
        tile_in_expert = tl.load(launch_tile_offsets_ptr + tile_id)

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
                k_blk = k_start // BLOCK_K
                a_scale = tl.load(
                    A_scale_ptr + k_blk * stride_as0 + tok_ids * stride_as1,
                    mask=mask_m,
                    other=1.0,
                )
                x1_scale = tl.load(
                    W13_scale_ptr
                    + expert_id * stride_w13se
                    + x1_row_blk * stride_w13sn
                    + k_blk * stride_w13sk
                )
                x2_scale = tl.load(
                    W13_scale_ptr
                    + expert_id * stride_w13se
                    + x2_row_blk * stride_w13sn
                    + k_blk * stride_w13sk
                )

                for k_sub in range(0, BLOCK_K, DOT_BLOCK_K):
                    offs_k = k_start + k_sub + tl.arange(0, DOT_BLOCK_K)
                    mask_k = offs_k < H

                    a_ptrs = A_ptr + tok_ids[:, None] * stride_ah0 + offs_k[None, :] * stride_ah1
                    a_fp8 = tl.load(
                        a_ptrs,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    a_fp32 = a_fp8.to(tl.float32)

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
                    x1_fp32 = x1_fp8.to(tl.float32)

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
                    if USE_NATIVE_FP8_DOT:
                        acc_x1 += tl.dot(a_fp8, x1_fp8) * a_scale[:, None] * x1_scale
                        acc_x2 += tl.dot(a_fp8, x2_fp8) * a_scale[:, None] * x2_scale
                    else:
                        x2_fp32 = x2_fp8.to(tl.float32)
                        acc_x1 += (
                            tl.dot(a_fp32, x1_fp32, input_precision="ieee")
                            * a_scale[:, None]
                            * x1_scale
                        )
                        acc_x2 += (
                            tl.dot(a_fp32, x2_fp32, input_precision="ieee")
                            * a_scale[:, None]
                            * x2_scale
                        )

            swiglu = (acc_x2 * tl.sigmoid(acc_x2)) * acc_x1
            out_ptrs = Out_ptr + pair_pos[:, None] * stride_om + offs_n[None, :] * stride_on
            tl.store(out_ptrs, swiglu, mask=mask_m[:, None] & mask_n[None, :])

        tile_id += num_sms


@triton.jit
def _grouped_gemm1_preact_kernel(
    A_ptr,
    A_scale_ptr,
    sorted_ids_ptr,
    W13_ptr,
    W13_scale_ptr,
    expert_starts_ptr,
    counts_ptr,
    launch_tile_experts_ptr,
    launch_tile_offsets_ptr,
    total_tiles,
    X1_ptr,
    X2_ptr,
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
    stride_x1m,
    stride_x1n,
    stride_x2m,
    stride_x2n,
    T,
    E: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_BLOCK_K: tl.constexpr,
    BYPASS_SCALES: tl.constexpr,
    MAX_K_TILES: tl.constexpr,
    USE_NATIVE_FP8_DOT: tl.constexpr,
):
    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    tile_id = sm_id

    while tile_id < total_tiles:
        expert_id = tl.load(launch_tile_experts_ptr + tile_id)
        expert_start = tl.load(expert_starts_ptr + expert_id)
        expert_count = tl.load(counts_ptr + expert_id)
        tile_in_expert = tl.load(launch_tile_offsets_ptr + tile_id)

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

            k_iters = H // BLOCK_K if MAX_K_TILES == 0 else MAX_K_TILES
            for k_idx in range(0, k_iters):
                k_start = k_idx * BLOCK_K
                k_blk = k_idx

                if BYPASS_SCALES:
                    a_scale = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
                    x1_scale = 1.0
                    x2_scale = 1.0
                else:
                    a_scale = tl.load(
                        A_scale_ptr + k_blk * stride_as0 + tok_ids * stride_as1,
                        mask=mask_m,
                        other=1.0,
                    )
                    x1_scale = tl.load(
                        W13_scale_ptr
                        + expert_id * stride_w13se
                        + x1_row_blk * stride_w13sn
                        + k_blk * stride_w13sk
                    )
                    x2_scale = tl.load(
                        W13_scale_ptr
                        + expert_id * stride_w13se
                        + x2_row_blk * stride_w13sn
                        + k_blk * stride_w13sk
                    )

                for k_sub in range(0, BLOCK_K, DOT_BLOCK_K):
                    offs_k = k_start + k_sub + tl.arange(0, DOT_BLOCK_K)
                    mask_k = offs_k < H

                    a_ptrs = A_ptr + tok_ids[:, None] * stride_ah0 + offs_k[None, :] * stride_ah1
                    a_fp8 = tl.load(
                        a_ptrs,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    a_fp32 = a_fp8.to(tl.float32)

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
                    x1_fp32 = x1_fp8.to(tl.float32)

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
                    if USE_NATIVE_FP8_DOT:
                        acc_x1 += tl.dot(a_fp8, x1_fp8) * a_scale[:, None] * x1_scale
                        acc_x2 += tl.dot(a_fp8, x2_fp8) * a_scale[:, None] * x2_scale
                    else:
                        x2_fp32 = x2_fp8.to(tl.float32)
                        acc_x1 += (
                            tl.dot(a_fp32, x1_fp32, input_precision="ieee")
                            * a_scale[:, None]
                            * x1_scale
                        )
                        acc_x2 += (
                            tl.dot(a_fp32, x2_fp32, input_precision="ieee")
                            * a_scale[:, None]
                            * x2_scale
                        )

            x1_ptrs = X1_ptr + pair_pos[:, None] * stride_x1m + offs_n[None, :] * stride_x1n
            x2_ptrs = X2_ptr + pair_pos[:, None] * stride_x2m + offs_n[None, :] * stride_x2n
            mask_out = mask_m[:, None] & mask_n[None, :]
            tl.store(x1_ptrs, acc_x1, mask=mask_out)
            tl.store(x2_ptrs, acc_x2, mask=mask_out)

        tile_id += num_sms


@triton.jit
def _grouped_gemm2_kernel(
    A_ptr,
    A_scale_ptr,
    sorted_ids_ptr,
    W2_ptr,
    W2_scale_ptr,
    expert_starts_ptr,
    counts_ptr,
    launch_tile_experts_ptr,
    launch_tile_offsets_ptr,
    total_tiles,
    Out_ptr,
    stride_am,
    stride_ak,
    stride_as0,
    stride_as1,
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
    DOT_BLOCK_K: tl.constexpr,
    A_SCALE_BLOCK: tl.constexpr,
    USE_NATIVE_FP8_DOT: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
):
    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    tile_id = sm_id

    while tile_id < total_tiles:
        expert_id = tl.load(launch_tile_experts_ptr + tile_id)
        expert_start = tl.load(expert_starts_ptr + expert_id)
        expert_count = tl.load(counts_ptr + expert_id)
        tile_in_expert = tl.load(launch_tile_offsets_ptr + tile_id)

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
                k_blk = k_start // BLOCK_K
                w_scale = tl.load(
                    W2_scale_ptr
                    + expert_id * stride_w2se
                    + row_blk * stride_w2sn
                    + k_blk * stride_w2sk
                )

                for k_sub in range(0, BLOCK_K, DOT_BLOCK_K):
                    offs_k = k_start + k_sub + tl.arange(0, DOT_BLOCK_K)
                    mask_k = offs_k < I
                    a_scale_blk = (k_start + k_sub) // A_SCALE_BLOCK

                    a_ptrs = A_ptr + pair_pos[:, None] * stride_am + offs_k[None, :] * stride_ak
                    w_ptrs = (
                        W2_ptr
                        + expert_id * stride_w2e
                        + offs_k[:, None] * stride_w2k
                        + offs_n[None, :] * stride_w2n
                    )
                    if USE_NATIVE_FP8_DOT:
                        a_scale = tl.load(
                            A_scale_ptr + a_scale_blk * stride_as0 + pair_pos * stride_as1,
                            mask=mask_m,
                            other=1.0,
                        )
                        a_fp8 = tl.load(
                            a_ptrs,
                            mask=mask_m[:, None] & mask_k[None, :],
                            other=0.0,
                        )
                        w_fp8 = tl.load(
                            w_ptrs,
                            mask=mask_k[:, None] & mask_n[None, :],
                            other=0.0,
                        )
                        acc += tl.dot(a_fp8, w_fp8) * a_scale[:, None] * w_scale
                    elif USE_BF16_DOT:
                        a_bf16 = tl.load(
                            a_ptrs,
                            mask=mask_m[:, None] & mask_k[None, :],
                            other=0.0,
                        ).to(tl.bfloat16)
                        w_bf16 = tl.load(
                            w_ptrs,
                            mask=mask_k[:, None] & mask_n[None, :],
                            other=0.0,
                        ).to(tl.bfloat16)
                        acc += tl.dot(a_bf16, w_bf16) * w_scale
                    else:
                        a_fp32 = tl.load(
                            a_ptrs,
                            mask=mask_m[:, None] & mask_k[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        w_fp32 = tl.load(
                            w_ptrs,
                            mask=mask_k[:, None] & mask_n[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        acc += tl.dot(a_fp32, w_fp32, input_precision="ieee") * w_scale

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
    tile_to_expert,
):
    num_pairs = int(sorted_ids.numel())
    out = torch.empty((num_pairs, I), dtype=torch.float32, device=hidden_states.device)
    launch_tile_experts, launch_tile_offsets = _build_launch_tile_metadata(
        counts,
        tile_starts,
        tile_to_expert,
    )
    total_tiles = int(launch_tile_experts.numel())
    if total_tiles == 0:
        return out
    dot_block_k = DOT_BLOCK_K

    _grouped_gemm1_swiglu_kernel[(NUM_SMS,)](
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        launch_tile_experts,
        launch_tile_offsets,
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
        DOT_BLOCK_K=dot_block_k,
        USE_NATIVE_FP8_DOT=USE_NATIVE_FP8_GEMM1,
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
    tile_to_expert,
):
    num_pairs = int(sorted_ids.numel())
    if USE_NATIVE_FP8_GEMM2:
        swiglu_fp8, swiglu_scale = _quantize_fp8_block_rows(
            swiglu_out.contiguous(),
            block_size=GEMM2_A_SCALE_BLOCK,
        )
        swiglu_input = swiglu_fp8
    elif USE_BF16_GEMM2:
        swiglu_input = swiglu_out.to(torch.bfloat16).contiguous()
        swiglu_scale = torch.empty((1, 1), dtype=torch.float32, device=swiglu_out.device)
    else:
        swiglu_input = swiglu_out.contiguous()
        swiglu_scale = torch.empty((1, 1), dtype=torch.float32, device=swiglu_out.device)
    out = torch.empty((num_pairs, H), dtype=torch.float32, device=swiglu_out.device)
    launch_tile_experts, launch_tile_offsets = _build_launch_tile_metadata(
        counts,
        tile_starts,
        tile_to_expert,
    )
    total_tiles = int(launch_tile_experts.numel())
    if total_tiles == 0:
        return out

    _grouped_gemm2_kernel[(NUM_SMS,)](
        swiglu_input,
        swiglu_scale,
        sorted_ids,
        gemm2_weights,
        gemm2_weights_scale,
        expert_starts,
        counts,
        launch_tile_experts,
        launch_tile_offsets,
        total_tiles,
        out,
        swiglu_input.stride(0),
        swiglu_input.stride(1),
        swiglu_scale.stride(0),
        swiglu_scale.stride(1),
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
        DOT_BLOCK_K=DOT_BLOCK_K,
        A_SCALE_BLOCK=GEMM2_A_SCALE_BLOCK,
        USE_NATIVE_FP8_DOT=USE_NATIVE_FP8_GEMM2,
        USE_BF16_DOT=USE_BF16_GEMM2,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
    )
    return out


def _launch_grouped_gemm1_preact(
    hidden_states,
    hidden_states_scale,
    sorted_ids,
    gemm1_weights,
    gemm1_weights_scale,
    expert_starts,
    counts,
    tile_starts,
    tile_to_expert,
    *,
    bypass_scales=False,
    max_k_tiles=0,
):
    num_pairs = int(sorted_ids.numel())
    x1_out = torch.empty((num_pairs, I), dtype=torch.float32, device=hidden_states.device)
    x2_out = torch.empty((num_pairs, I), dtype=torch.float32, device=hidden_states.device)
    launch_tile_experts, launch_tile_offsets = _build_launch_tile_metadata(
        counts,
        tile_starts,
        tile_to_expert,
    )
    total_tiles = int(launch_tile_experts.numel())
    if total_tiles == 0:
        return x1_out, x2_out
    dot_block_k = DOT_BLOCK_K

    _grouped_gemm1_preact_kernel[(NUM_SMS,)](
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        launch_tile_experts,
        launch_tile_offsets,
        total_tiles,
        x1_out,
        x2_out,
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
        x1_out.stride(0),
        x1_out.stride(1),
        x2_out.stride(0),
        x2_out.stride(1),
        hidden_states.shape[0],
        E=E_LOCAL,
        H=H,
        I=I,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        DOT_BLOCK_K=dot_block_k,
        BYPASS_SCALES=bypass_scales,
        MAX_K_TILES=max_k_tiles,
        USE_NATIVE_FP8_DOT=USE_NATIVE_FP8_GEMM1,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
    )
    return x1_out, x2_out


def _grouped_gemm2_reference(
    swiglu_out,
    gemm2_weights,
    gemm2_weights_scale,
    expert_starts,
    counts,
):
    num_pairs = swiglu_out.shape[0]
    out = torch.empty((num_pairs, H), dtype=torch.float32, device=swiglu_out.device)
    active_experts = torch.nonzero(counts > 0, as_tuple=False).flatten()
    for expert_id in active_experts.tolist():
        start = int(expert_starts[expert_id].item())
        count = int(counts[expert_id].item())
        end = start + count
        if count == 0:
            continue
        w2 = _dequant_gemm2_weight(gemm2_weights, gemm2_weights_scale, expert_id)
        out[start:end] = swiglu_out[start:end].matmul(w2.t())
    return out


def _grouped_gemm1_preact_reference(
    hidden_states,
    hidden_states_scale,
    sorted_ids,
    gemm1_weights,
    gemm1_weights_scale,
    expert_starts,
    counts,
    bypass_scales=False,
    transpose_scale=False,
    max_k_tiles=0,
):
    num_pairs = int(sorted_ids.numel())
    x1_out = torch.empty((num_pairs, I), dtype=torch.float32, device=hidden_states.device)
    x2_out = torch.empty((num_pairs, I), dtype=torch.float32, device=hidden_states.device)
    active_experts = torch.nonzero(counts > 0, as_tuple=False).flatten()
    for expert_id in active_experts.tolist():
        start = int(expert_starts[expert_id].item())
        count = int(counts[expert_id].item())
        end = start + count
        token_ids = sorted_ids[start:end].to(torch.long)
        if token_ids.numel() == 0:
            continue
        hidden_rows = _dequant_hidden_rows(
            hidden_states,
            hidden_states_scale,
            token_ids,
            bypass_scales=bypass_scales,
        )
        if transpose_scale:
            w13 = _dequant_gemm1_weight_transposed_scale(
                gemm1_weights,
                gemm1_weights_scale,
                expert_id,
                bypass_scales=bypass_scales,
            )
        else:
            w13 = _dequant_gemm1_weight(
                gemm1_weights,
                gemm1_weights_scale,
                expert_id,
                bypass_scales=bypass_scales,
            )
        if max_k_tiles > 0:
            k_limit = min(H, max_k_tiles * BLOCK_K)
            gemm1 = hidden_rows[:, :k_limit].matmul(w13[:, :k_limit].t())
        else:
            gemm1 = hidden_rows.matmul(w13.t())
        x1_out[start:end] = gemm1[:, :I]
        x2_out[start:end] = gemm1[:, I:]
    return x1_out, x2_out


def _grouped_gemm1_swiglu_reference(
    hidden_states,
    hidden_states_scale,
    sorted_ids,
    gemm1_weights,
    gemm1_weights_scale,
    expert_starts,
    counts,
    swap_gate_up=False,
    bypass_scales=False,
    transpose_scale=False,
):
    x1_out, x2_out = _grouped_gemm1_preact_reference(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        bypass_scales=bypass_scales,
        transpose_scale=transpose_scale,
    )
    return _apply_swiglu_halves(
        x1_out,
        x2_out,
        gate_is_second=not swap_gate_up,
    )


def _tile_row_bounds(tile_id, expert_starts, counts, tile_starts, tile_to_expert):
    expert_id = int(tile_to_expert[tile_id].item())
    expert_tile_start = int(tile_starts[expert_id].item())
    expert_start = int(expert_starts[expert_id].item())
    expert_count = int(counts[expert_id].item())
    tile_in_expert = tile_id - expert_tile_start
    row_start = expert_start + tile_in_expert * BLOCK_M
    row_end = min(row_start + BLOCK_M, expert_start + expert_count)
    return expert_id, row_start, row_end


def _top_error_columns(diff, limit=8):
    if diff.numel() == 0:
        return []
    col_max = diff.max(dim=0).values
    k = min(limit, int(col_max.numel()))
    values, indices = torch.topk(col_max, k)
    cols = []
    for idx, value in zip(indices.tolist(), values.tolist()):
        if value <= 0:
            continue
        cols.append({"col": int(idx), "max_abs_error": float(value)})
    return cols


def _apply_swiglu_halves(first_half, second_half, *, gate_is_second=True):
    if gate_is_second:
        return torch.nn.functional.silu(second_half) * first_half
    return torch.nn.functional.silu(first_half) * second_half


def _route_topk_ids_and_weights(routing_logits, routing_bias, routed_scaling_factor):
    T = routing_logits.shape[0]
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
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_GLOBAL)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * float(routed_scaling_factor)
    return topk_idx, weights


def _route_topk_ids(routing_logits, routing_bias):
    topk_idx, _ = _route_topk_ids_and_weights(
        routing_logits,
        routing_bias,
        routed_scaling_factor=1.0,
    )
    return topk_idx


def debug_grouped_gemm1_preactivation(
    topk_ids,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    local_expert_offset,
    *,
    bypass_scales=False,
    max_k_tiles=0,
    atol=1e-4,
    rtol=1e-4,
    max_mismatched_tiles=8,
):
    device = hidden_states.device
    local_start = (
        int(local_expert_offset.item())
        if isinstance(local_expert_offset, torch.Tensor)
        else int(local_expert_offset)
    )
    (
        sorted_ids,
        _sorted_local_experts,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
    ) = _sort_tokens(topk_ids, local_start, E_LOCAL, device)

    if sorted_ids.numel() == 0:
        return {
            "ok": True,
            "bypass_scales": bool(bypass_scales),
            "num_pairs": 0,
            "total_tiles": 0,
            "x1_max_abs_error": 0.0,
            "x2_max_abs_error": 0.0,
            "x1_top_error_columns": [],
            "x2_top_error_columns": [],
            "mismatched_tiles": [],
        }

    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()
    gemm1_weights = gemm1_weights.contiguous()
    gemm1_weights_scale = gemm1_weights_scale.contiguous()

    x1_ref, x2_ref = _grouped_gemm1_preact_reference(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        bypass_scales=bypass_scales,
        max_k_tiles=max_k_tiles,
    )
    x1_candidate, x2_candidate = _launch_grouped_gemm1_preact(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
        bypass_scales=bypass_scales,
        max_k_tiles=max_k_tiles,
    )

    x1_diff = (x1_candidate - x1_ref).abs()
    x2_diff = (x2_candidate - x2_ref).abs()
    x1_max_abs_error = float(x1_diff.max().item()) if x1_diff.numel() else 0.0
    x2_max_abs_error = float(x2_diff.max().item()) if x2_diff.numel() else 0.0
    mismatched_tiles = []
    total_tiles = int(tile_to_expert.numel())
    for tile_id in range(total_tiles):
        expert_id, row_start, row_end = _tile_row_bounds(
            tile_id,
            expert_starts,
            counts,
            tile_starts,
            tile_to_expert,
        )
        if row_start >= row_end:
            continue
        tile_x1_diff = x1_diff[row_start:row_end]
        tile_x2_diff = x2_diff[row_start:row_end]
        tile_x1_ok = torch.allclose(
            x1_candidate[row_start:row_end],
            x1_ref[row_start:row_end],
            atol=atol,
            rtol=rtol,
        )
        tile_x2_ok = torch.allclose(
            x2_candidate[row_start:row_end],
            x2_ref[row_start:row_end],
            atol=atol,
            rtol=rtol,
        )
        if tile_x1_ok and tile_x2_ok:
            continue
        mismatched_tiles.append(
            {
                "tile_id": tile_id,
                "expert_id": expert_id,
                "pair_start": row_start,
                "pair_end": row_end,
                "x1_max_abs_error": float(tile_x1_diff.max().item()) if tile_x1_diff.numel() else 0.0,
                "x2_max_abs_error": float(tile_x2_diff.max().item()) if tile_x2_diff.numel() else 0.0,
                "x1_top_error_columns": _top_error_columns(tile_x1_diff, limit=4),
                "x2_top_error_columns": _top_error_columns(tile_x2_diff, limit=4),
            }
        )
        if len(mismatched_tiles) >= max_mismatched_tiles:
            break

    return {
        "ok": torch.allclose(x1_candidate, x1_ref, atol=atol, rtol=rtol)
        and torch.allclose(x2_candidate, x2_ref, atol=atol, rtol=rtol),
        "bypass_scales": bool(bypass_scales),
        "max_k_tiles": int(max_k_tiles),
        "num_pairs": int(sorted_ids.numel()),
        "total_tiles": total_tiles,
        "x1_max_abs_error": x1_max_abs_error,
        "x2_max_abs_error": x2_max_abs_error,
        "x1_top_error_columns": _top_error_columns(x1_diff),
        "x2_top_error_columns": _top_error_columns(x2_diff),
        "mismatched_tiles": mismatched_tiles,
    }


def debug_grouped_gemm1_correctness(
    topk_ids,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    local_expert_offset,
    *,
    atol=1e-4,
    rtol=1e-4,
    max_mismatched_tiles=8,
):
    device = hidden_states.device
    local_start = (
        int(local_expert_offset.item())
        if isinstance(local_expert_offset, torch.Tensor)
        else int(local_expert_offset)
    )
    (
        sorted_ids,
        _sorted_local_experts,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
    ) = _sort_tokens(topk_ids, local_start, E_LOCAL, device)

    if sorted_ids.numel() == 0:
        return {
            "ok": True,
            "num_pairs": 0,
            "total_tiles": 0,
            "max_abs_error": 0.0,
            "mismatched_tiles": [],
        }

    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()
    gemm1_weights = gemm1_weights.contiguous()
    gemm1_weights_scale = gemm1_weights_scale.contiguous()

    reference = _grouped_gemm1_swiglu_reference(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
    )
    swapped_reference = _grouped_gemm1_swiglu_reference(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        swap_gate_up=True,
    )
    transposed_scale_reference = _grouped_gemm1_swiglu_reference(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        transpose_scale=True,
    )
    candidate = _launch_grouped_gemm1_swiglu(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
    )

    diff = (candidate - reference).abs()
    max_abs_error = float(diff.max().item()) if diff.numel() else 0.0
    swapped_diff = (candidate - swapped_reference).abs()
    swapped_gate_up_max_abs_error = float(swapped_diff.max().item()) if swapped_diff.numel() else 0.0
    transposed_scale_diff = (candidate - transposed_scale_reference).abs()
    transposed_scale_max_abs_error = (
        float(transposed_scale_diff.max().item()) if transposed_scale_diff.numel() else 0.0
    )
    mismatched_tiles = []
    total_tiles = int(tile_to_expert.numel())
    for tile_id in range(total_tiles):
        expert_id, row_start, row_end = _tile_row_bounds(
            tile_id,
            expert_starts,
            counts,
            tile_starts,
            tile_to_expert,
        )
        if row_start >= row_end:
            continue
        tile_candidate = candidate[row_start:row_end]
        tile_reference = reference[row_start:row_end]
        if torch.allclose(tile_candidate, tile_reference, atol=atol, rtol=rtol):
            continue
        tile_diff = (tile_candidate - tile_reference).abs()
        mismatched_tiles.append(
            {
                "tile_id": tile_id,
                "expert_id": expert_id,
                "pair_start": row_start,
                "pair_end": row_end,
                "max_abs_error": float(tile_diff.max().item()),
            }
        )
        if len(mismatched_tiles) >= max_mismatched_tiles:
            break

    return {
        "ok": torch.allclose(candidate, reference, atol=atol, rtol=rtol),
        "num_pairs": int(sorted_ids.numel()),
        "total_tiles": total_tiles,
        "max_abs_error": max_abs_error,
        "swapped_gate_up_max_abs_error": swapped_gate_up_max_abs_error,
        "transposed_scale_max_abs_error": transposed_scale_max_abs_error,
        "mismatched_tiles": mismatched_tiles,
    }


def debug_grouped_gemm1_correctness_from_inputs(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
    *,
    atol=1e-4,
    rtol=1e-4,
    max_mismatched_tiles=8,
):
    del gemm2_weights, gemm2_weights_scale

    if isinstance(local_expert_offset, torch.Tensor):
        local_start = int(local_expert_offset.item())
    else:
        local_start = int(local_expert_offset)

    return debug_grouped_gemm1_correctness(
        _route_topk_ids(routing_logits, routing_bias),
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        local_start,
        atol=atol,
        rtol=rtol,
        max_mismatched_tiles=max_mismatched_tiles,
    )


def debug_grouped_gemm1_preactivation_from_inputs(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
    *,
    bypass_scales=False,
    max_k_tiles=0,
    atol=1e-4,
    rtol=1e-4,
    max_mismatched_tiles=8,
):
    del gemm2_weights, gemm2_weights_scale, routed_scaling_factor

    return debug_grouped_gemm1_preactivation(
        _route_topk_ids(routing_logits, routing_bias),
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        local_expert_offset,
        bypass_scales=bypass_scales,
        max_k_tiles=max_k_tiles,
        atol=atol,
        rtol=rtol,
        max_mismatched_tiles=max_mismatched_tiles,
    )


def debug_grouped_gemm1_swiglu_stages(
    topk_ids,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    local_expert_offset,
):
    device = hidden_states.device
    local_start = (
        int(local_expert_offset.item())
        if isinstance(local_expert_offset, torch.Tensor)
        else int(local_expert_offset)
    )
    (
        sorted_ids,
        _sorted_local_experts,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
    ) = _sort_tokens(topk_ids, local_start, E_LOCAL, device)

    if sorted_ids.numel() == 0:
        return {
            "ok": True,
            "num_pairs": 0,
            "fused_vs_reference_max_abs_error": 0.0,
            "recomputed_vs_reference_max_abs_error": 0.0,
            "fused_vs_recomputed_max_abs_error": 0.0,
            "fused_vs_swapped_reference_max_abs_error": 0.0,
        }

    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()
    gemm1_weights = gemm1_weights.contiguous()
    gemm1_weights_scale = gemm1_weights_scale.contiguous()

    x1_ref, x2_ref = _grouped_gemm1_preact_reference(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
    )
    x1_candidate, x2_candidate = _launch_grouped_gemm1_preact(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
    )
    swiglu_reference = _apply_swiglu_halves(x1_ref, x2_ref, gate_is_second=True)
    swiglu_swapped_reference = _apply_swiglu_halves(x1_ref, x2_ref, gate_is_second=False)
    swiglu_recomputed = _apply_swiglu_halves(x1_candidate, x2_candidate, gate_is_second=True)
    swiglu_fused = _launch_grouped_gemm1_swiglu(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
    )
    fused_vs_reference = (swiglu_fused - swiglu_reference).abs()
    recomputed_vs_reference = (swiglu_recomputed - swiglu_reference).abs()
    fused_vs_recomputed = (swiglu_fused - swiglu_recomputed).abs()
    fused_vs_swapped_reference = (swiglu_fused - swiglu_swapped_reference).abs()
    return {
        "ok": torch.allclose(swiglu_fused, swiglu_reference, atol=1e-4, rtol=1e-4),
        "num_pairs": int(sorted_ids.numel()),
        "fused_vs_reference_max_abs_error": float(fused_vs_reference.max().item()),
        "recomputed_vs_reference_max_abs_error": float(recomputed_vs_reference.max().item()),
        "fused_vs_recomputed_max_abs_error": float(fused_vs_recomputed.max().item()),
        "fused_vs_swapped_reference_max_abs_error": float(
            fused_vs_swapped_reference.max().item()
        ),
    }


def debug_grouped_gemm1_swiglu_stages_from_inputs(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    del gemm2_weights, gemm2_weights_scale, routed_scaling_factor

    return debug_grouped_gemm1_swiglu_stages(
        _route_topk_ids(routing_logits, routing_bias),
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        local_expert_offset,
    )


def debug_grouped_gemm2_correctness(
    topk_ids,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    *,
    atol=1e-4,
    rtol=1e-4,
    max_mismatched_tiles=8,
):
    device = hidden_states.device
    local_start = (
        int(local_expert_offset.item())
        if isinstance(local_expert_offset, torch.Tensor)
        else int(local_expert_offset)
    )
    (
        sorted_ids,
        _sorted_local_experts,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
    ) = _sort_tokens(topk_ids, local_start, E_LOCAL, device)

    if sorted_ids.numel() == 0:
        return {
            "ok": True,
            "num_pairs": 0,
            "total_tiles": 0,
            "max_abs_error": 0.0,
            "top_error_columns": [],
            "mismatched_tiles": [],
        }

    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()
    gemm1_weights = gemm1_weights.contiguous()
    gemm1_weights_scale = gemm1_weights_scale.contiguous()
    gemm2_weights = gemm2_weights.contiguous()
    gemm2_weights_scale = gemm2_weights_scale.contiguous()

    swiglu_reference = _grouped_gemm1_swiglu_reference(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
    )
    reference = _grouped_gemm2_reference(
        swiglu_reference,
        gemm2_weights,
        gemm2_weights_scale,
        expert_starts,
        counts,
    )
    swiglu_input_max_abs = float(swiglu_reference.abs().max().item()) if swiglu_reference.numel() else 0.0
    bf16_activation_transform_max_abs_error = float(
        (swiglu_reference.to(torch.bfloat16).to(torch.float32) - swiglu_reference)
        .abs()
        .max()
        .item()
    ) if swiglu_reference.numel() else 0.0
    fp16_activation_transform_max_abs_error = float(
        (swiglu_reference.to(torch.float16).to(torch.float32) - swiglu_reference)
        .abs()
        .max()
        .item()
    ) if swiglu_reference.numel() else 0.0
    activation_transform_max_abs_error = 0.0
    if USE_NATIVE_FP8_GEMM2:
        swiglu_q, swiglu_q_scale = _quantize_fp8_block_rows(
            swiglu_reference,
            block_size=GEMM2_A_SCALE_BLOCK,
        )
        activation_transform_max_abs_error = float(
            (
                _dequant_fp8_block_rows(
                    swiglu_q,
                    swiglu_q_scale,
                    block_size=GEMM2_A_SCALE_BLOCK,
                )
                - swiglu_reference
            )
            .abs()
            .max()
            .item()
        )
    elif USE_BF16_GEMM2:
        activation_transform_max_abs_error = float(
            (swiglu_reference.to(torch.bfloat16).to(torch.float32) - swiglu_reference)
            .abs()
            .max()
            .item()
        )
    candidate = _launch_grouped_gemm2(
        swiglu_reference,
        sorted_ids,
        gemm2_weights,
        gemm2_weights_scale,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
    )

    diff = (candidate - reference).abs()
    mismatched_tiles = []
    total_tiles = int(tile_to_expert.numel())
    for tile_id in range(total_tiles):
        expert_id, row_start, row_end = _tile_row_bounds(
            tile_id,
            expert_starts,
            counts,
            tile_starts,
            tile_to_expert,
        )
        if row_start >= row_end:
            continue
        tile_candidate = candidate[row_start:row_end]
        tile_reference = reference[row_start:row_end]
        if torch.allclose(tile_candidate, tile_reference, atol=atol, rtol=rtol):
            continue
        tile_diff = (tile_candidate - tile_reference).abs()
        mismatched_tiles.append(
            {
                "tile_id": tile_id,
                "expert_id": expert_id,
                "pair_start": row_start,
                "pair_end": row_end,
                "max_abs_error": float(tile_diff.max().item()),
                "top_error_columns": _top_error_columns(tile_diff, limit=4),
            }
        )
        if len(mismatched_tiles) >= max_mismatched_tiles:
            break

    return {
        "ok": torch.allclose(candidate, reference, atol=atol, rtol=rtol),
        "num_pairs": int(sorted_ids.numel()),
        "total_tiles": total_tiles,
        "swiglu_input_max_abs": swiglu_input_max_abs,
        "bf16_activation_transform_max_abs_error": bf16_activation_transform_max_abs_error,
        "fp16_activation_transform_max_abs_error": fp16_activation_transform_max_abs_error,
        "activation_transform_max_abs_error": activation_transform_max_abs_error,
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "top_error_columns": _top_error_columns(diff),
        "mismatched_tiles": mismatched_tiles,
    }


def debug_grouped_gemm2_correctness_from_inputs(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
    *,
    atol=1e-4,
    rtol=1e-4,
    max_mismatched_tiles=8,
):
    del routed_scaling_factor

    return debug_grouped_gemm2_correctness(
        _route_topk_ids(routing_logits, routing_bias),
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        local_expert_offset,
        atol=atol,
        rtol=rtol,
        max_mismatched_tiles=max_mismatched_tiles,
    )


def debug_grouped_end_to_end_components_from_inputs(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    device = hidden_states.device
    local_start = (
        int(local_expert_offset.item())
        if isinstance(local_expert_offset, torch.Tensor)
        else int(local_expert_offset)
    )
    routed_scaling_factor = (
        float(routed_scaling_factor.item())
        if isinstance(routed_scaling_factor, torch.Tensor)
        else float(routed_scaling_factor)
    )
    topk_idx, weights = _route_topk_ids_and_weights(
        routing_logits,
        routing_bias,
        routed_scaling_factor,
    )
    (
        sorted_ids,
        sorted_local_experts,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
    ) = _sort_tokens(topk_idx, local_start, E_LOCAL, device)

    if sorted_ids.numel() == 0:
        return {
            "ok": True,
            "num_pairs": 0,
            "swiglu_fused_vs_reference_max_abs_error": 0.0,
            "swiglu_recomputed_vs_reference_max_abs_error": 0.0,
            "gemm2_fused_vs_reference_max_abs_error": 0.0,
            "gemm2_recomputed_vs_reference_max_abs_error": 0.0,
            "accum_fused_vs_reference_max_abs_error": 0.0,
            "accum_recomputed_vs_reference_max_abs_error": 0.0,
            "final_bf16_fused_vs_reference_max_abs_error": 0.0,
            "final_bf16_recomputed_vs_reference_max_abs_error": 0.0,
        }

    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()
    gemm1_weights = gemm1_weights.contiguous()
    gemm1_weights_scale = gemm1_weights_scale.contiguous()
    gemm2_weights = gemm2_weights.contiguous()
    gemm2_weights_scale = gemm2_weights_scale.contiguous()

    swiglu_reference = _grouped_gemm1_swiglu_reference(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
    )
    x1_candidate, x2_candidate = _launch_grouped_gemm1_preact(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
    )
    swiglu_recomputed = _apply_swiglu_halves(x1_candidate, x2_candidate, gate_is_second=True)
    swiglu_fused = _launch_grouped_gemm1_swiglu(
        hidden_states,
        hidden_states_scale,
        sorted_ids,
        gemm1_weights,
        gemm1_weights_scale,
        expert_starts,
        counts,
        tile_starts,
        tile_to_expert,
    )

    gemm2_reference = _grouped_gemm2_reference(
        swiglu_reference,
        gemm2_weights,
        gemm2_weights_scale,
        expert_starts,
        counts,
    )
    gemm2_recomputed = _grouped_gemm2_reference(
        swiglu_recomputed,
        gemm2_weights,
        gemm2_weights_scale,
        expert_starts,
        counts,
    )
    gemm2_fused = _grouped_gemm2_reference(
        swiglu_fused,
        gemm2_weights,
        gemm2_weights_scale,
        expert_starts,
        counts,
    )

    sorted_global_experts = sorted_local_experts.to(torch.long) + local_start
    rw_sorted = weights[sorted_ids.to(torch.long), sorted_global_experts]

    T = routing_logits.shape[0]
    accum_reference = torch.zeros((T, H), dtype=torch.float32, device=device)
    accum_recomputed = torch.zeros((T, H), dtype=torch.float32, device=device)
    accum_fused = torch.zeros((T, H), dtype=torch.float32, device=device)
    sorted_ids_long = sorted_ids.to(torch.long)
    accum_reference.index_add_(0, sorted_ids_long, gemm2_reference * rw_sorted.unsqueeze(1))
    accum_recomputed.index_add_(0, sorted_ids_long, gemm2_recomputed * rw_sorted.unsqueeze(1))
    accum_fused.index_add_(0, sorted_ids_long, gemm2_fused * rw_sorted.unsqueeze(1))

    final_reference = accum_reference.to(torch.bfloat16)
    final_recomputed = accum_recomputed.to(torch.bfloat16)
    final_fused = accum_fused.to(torch.bfloat16)

    return {
        "ok": torch.allclose(final_fused, final_reference, atol=1e-4, rtol=1e-4),
        "num_pairs": int(sorted_ids.numel()),
        "swiglu_fused_vs_reference_max_abs_error": float(
            (swiglu_fused - swiglu_reference).abs().max().item()
        ),
        "swiglu_recomputed_vs_reference_max_abs_error": float(
            (swiglu_recomputed - swiglu_reference).abs().max().item()
        ),
        "swiglu_fused_vs_recomputed_max_abs_error": float(
            (swiglu_fused - swiglu_recomputed).abs().max().item()
        ),
        "gemm2_fused_vs_reference_max_abs_error": float(
            (gemm2_fused - gemm2_reference).abs().max().item()
        ),
        "gemm2_recomputed_vs_reference_max_abs_error": float(
            (gemm2_recomputed - gemm2_reference).abs().max().item()
        ),
        "gemm2_fused_vs_recomputed_max_abs_error": float(
            (gemm2_fused - gemm2_recomputed).abs().max().item()
        ),
        "accum_fused_vs_reference_max_abs_error": float(
            (accum_fused - accum_reference).abs().max().item()
        ),
        "accum_recomputed_vs_reference_max_abs_error": float(
            (accum_recomputed - accum_reference).abs().max().item()
        ),
        "accum_fused_vs_recomputed_max_abs_error": float(
            (accum_fused - accum_recomputed).abs().max().item()
        ),
        "final_bf16_fused_vs_reference_max_abs_error": float(
            (final_fused.to(torch.float32) - final_reference.to(torch.float32)).abs().max().item()
        ),
        "final_bf16_recomputed_vs_reference_max_abs_error": float(
            (
                final_recomputed.to(torch.float32) - final_reference.to(torch.float32)
            ).abs().max().item()
        ),
        "final_bf16_fused_vs_recomputed_max_abs_error": float(
            (
                final_fused.to(torch.float32) - final_recomputed.to(torch.float32)
            ).abs().max().item()
        ),
    }


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
        tile_to_expert,
    ) = _sort_tokens(topk_idx, local_start, E_LOCAL, device)

    if sorted_ids.numel() == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()
    gemm1_weights = gemm1_weights.contiguous()
    gemm1_weights_scale = gemm1_weights_scale.contiguous()
    gemm2_weights = gemm2_weights.contiguous()
    gemm2_weights_scale = gemm2_weights_scale.contiguous()

    if USE_GROUPED_TRITON or USE_GROUPED_TRITON_GEMM1_ONLY:
        swiglu_out = _launch_grouped_gemm1_swiglu(
            hidden_states,
            hidden_states_scale,
            sorted_ids,
            gemm1_weights,
            gemm1_weights_scale,
            expert_starts,
            counts,
            tile_starts,
            tile_to_expert,
        )
        if USE_GROUPED_TRITON:
            gemm2_out = _launch_grouped_gemm2(
                swiglu_out,
                sorted_ids,
                gemm2_weights,
                gemm2_weights_scale,
                expert_starts,
                counts,
                tile_starts,
                tile_to_expert,
            )
        else:
            gemm2_out = _grouped_gemm2_reference(
                swiglu_out,
                gemm2_weights,
                gemm2_weights_scale,
                expert_starts,
                counts,
            )

        sorted_ids_long = sorted_ids.to(torch.long)
        sorted_global_experts = sorted_local_experts.to(torch.long) + local_start
        rw_sorted = weights[sorted_ids_long, sorted_global_experts]
        accum = torch.zeros((T, H), dtype=torch.float32, device=device)
        accum.index_add_(0, sorted_ids_long, gemm2_out * rw_sorted.unsqueeze(1))
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

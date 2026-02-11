"""
Fused MoE Seed Kernel — DeepSeek-V3 / R1 architecture.

FP8 block-scale MoE: routing (sigmoid + grouped top-k) → GEMM1 → SwiGLU → GEMM2
with per-token weighted accumulation.

This is a correct-first PyTorch implementation that serves as the seed for
KernelForge's optimization loop.  Performance will be improved iteratively
by the mutation engine.
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
    local_start = int(local_expert_offset)

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

        A_e = A.index_select(0, token_idx)                    # [Tk, H]
        W13_e = W13[le]                                       # [2I, H]
        W2_e = W2[le]                                         # [H, I]

        # GEMM1:  [Tk, H] @ [H, 2I] = [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())

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

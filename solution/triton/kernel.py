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
def gemm1_kernel(
    # Input pointers
    a_ptr, w_ptr, out_ptr,
    a_scale_ptr, w_scale_ptr,
    sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_we, stride_wn, stride_wk,
    stride_om, stride_on,
    stride_asm, stride_ask,
    stride_wse, stride_wsn, stride_wsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
):
    """Fused GEMM1 kernel with CORRECT FP8 block-scale dequantization and N-tail guards."""
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Group ordering for L2 cache reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Check bounds
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    
    # Load sorted token ids
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id, mask=offs_token_id < num_tokens_post_padded, other=0)
    token_mask = offs_token < M
    
    # Get expert for this block - add bounds check
    expert_id = tl.load(expert_ids_ptr + pid_m, mask=pid_m < num_pid_m, other=0)
    
    # Initialize pointers
    offs_m = offs_token
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # CRITICAL: N-tail guard for all N-indexed operations
    n_mask = offs_n < N
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = w_ptr + (expert_id * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
    
    # Scale pointers - use modulo for N to prevent OOB
    a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
    # FIXED: Add N-tail protection for w_scale access
    offs_n_block = (offs_n % N) // QUANT_BLOCK
    w_scale_ptrs = w_scale_ptr + expert_id * stride_wse + offs_n_block * stride_wsn
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_rem = K - k * BLOCK_SIZE_K
        k_mask = offs_k < k_rem
        
        # Load input tiles (FP8) with proper masking
        a_fp8 = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        # CRITICAL: Add N-tail mask to weight loads
        w_fp8 = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Load scales for this K-block
        k_block = k * BLOCK_SIZE_K // QUANT_BLOCK
        a_scale = tl.load(a_scale_ptrs + k_block * stride_ask, mask=token_mask, other=1.0)
        # CRITICAL: Add N-tail mask to scale loads
        w_scale = tl.load(w_scale_ptrs + k_block * stride_wsk, mask=n_mask, other=1.0)

        # CORRECT FP8 dequantization pattern:
        # 1. Convert FP8 to FP32
        # 2. Apply scale in FP32 space
        # 3. Convert to FP16 for dot (avoiding fp32 in dot)
        a_fp32 = a_fp8.to(tl.float32) * a_scale[:, None]
        w_fp32 = w_fp8.to(tl.float32) * w_scale[None, :]
        
        # Convert to fp16 for dot - this is safe as we've already applied scales
        a_fp16 = a_fp32.to(tl.float16)
        w_fp16 = w_fp32.to(tl.float16)

        # Dot product on properly dequantized fp16 values
        accumulator += tl.dot(a_fp16, w_fp16)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk
    
    # Store output with N-tail guard - FIXED: use offs_token_id for pair position indexing
    offs_om = offs_token_id[:, None]
    offs_on = offs_n[None, :]
    out_ptrs = out_ptr + stride_om * offs_om + stride_on * offs_on
    # CRITICAL: Add N-tail mask to output stores
    out_mask = (offs_token_id[:, None] < M) & n_mask[None, :]
    tl.store(out_ptrs, accumulator, mask=out_mask)


@triton.jit
def gemm2_kernel(
    # Input pointers
    a_ptr, w_ptr, out_ptr,
    w_scale_ptr,
    sorted_token_ids_ptr, expert_ids_ptr,
    weights_ptr,
    num_tokens_post_padded,
    local_expert_offset,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_we, stride_wn, stride_wk,
    stride_om, stride_on,
    stride_wse, stride_wsn, stride_wsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    QUANT_BLOCK: tl.constexpr, E_GLOBAL_CONST: tl.constexpr,
):
    """Fused GEMM2 kernel with CORRECT FP8 block-scale dequantization and N-tail guards."""
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Group ordering for L2 cache reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Check bounds
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    
    # Load sorted token ids
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id, mask=offs_token_id < num_tokens_post_padded, other=0)
    token_mask = offs_token < M
    
    # Get expert for this block - add bounds check
    expert_id = tl.load(expert_ids_ptr + pid_m, mask=pid_m < num_pid_m, other=0)

    # Initialize pointers
    offs_m = offs_token_id
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # CRITICAL: N-tail guard for all N-indexed operations
    n_mask = offs_n < N

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = w_ptr + (expert_id * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
    
    # Scale pointers - use modulo for N to prevent OOB
    offs_n_block = (offs_n % N) // QUANT_BLOCK
    w_scale_ptrs = w_scale_ptr + expert_id * stride_wse + offs_n_block * stride_wsn
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_rem = K - k * BLOCK_SIZE_K
        k_mask = offs_k < k_rem
        
        # Load input tiles (a is bf16, w is FP8)
        a_bf16 = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        # CRITICAL: Add N-tail mask to weight loads
        w_fp8 = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Load scales for this K-block
        k_block = k * BLOCK_SIZE_K // QUANT_BLOCK
        # CRITICAL: Add N-tail mask to scale loads
        w_scale = tl.load(w_scale_ptrs + k_block * stride_wsk, mask=n_mask, other=1.0)

        # CORRECT FP8 dequantization pattern:
        # 1. Convert FP8 to FP32
        # 2. Apply scale in FP32 space
        # 3. Convert to BF16 for dot (avoiding fp32 in dot)
        w_fp32 = w_fp8.to(tl.float32) * w_scale[None, :]
        w_bf16 = w_fp32.to(tl.bfloat16)

        # Dot product on properly dequantized values
        accumulator += tl.dot(a_bf16, w_bf16)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk
    
    # Load routing weights
    expert_global = expert_id + local_expert_offset
    routing_weights = tl.load(weights_ptr + offs_token * E_GLOBAL_CONST + expert_global, mask=token_mask, other=0.0)
    
    # Apply routing weights
    accumulator = accumulator * routing_weights[:, None]
    
    # Store per-expert output with N-tail guard
    offs_om = offs_m[:, None]
    offs_on = offs_n[None, :]
    out_ptrs = out_ptr + stride_om * offs_om + stride_on * offs_on
    # CRITICAL: Add N-tail mask to output stores
    out_mask = (offs_m[:, None] < M) & n_mask[None, :]
    tl.store(out_ptrs, accumulator, mask=out_mask)


def kernel(
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
    T = routing_logits.shape[0]
    device = hidden_states.device
    local_start = int(local_expert_offset)

    # ----------------------------------------------------------------
    # 1) DeepSeek-V3 no-aux routing
    # ----------------------------------------------------------------
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    # Group → top-2 per group → sum → pick top-4 groups
    s_wb_g = s_with_bias.view(T, N_GROUP, GROUP_SIZE)
    top2_vals, _ = torch.topk(s_wb_g, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
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
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    # Combination weights
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # ----------------------------------------------------------------
    # 2) Pre-sort tokens by expert
    # ----------------------------------------------------------------
    token_expert_pairs = []
    for t in range(T):
        for k in range(TOP_K):
            expert_global = topk_idx[t, k].item()
            expert_local = expert_global - local_start
            if 0 <= expert_local < E_LOCAL:
                token_expert_pairs.append((t, expert_local, expert_global))
    
    if not token_expert_pairs:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)
    
    token_expert_pairs.sort(key=lambda x: x[1])
    
    sorted_tokens = torch.tensor([p[0] for p in token_expert_pairs], dtype=torch.int32, device=device)
    expert_ids = torch.tensor([p[1] for p in token_expert_pairs], dtype=torch.int32, device=device)
    expert_global_ids = torch.tensor([p[2] for p in token_expert_pairs], dtype=torch.int32, device=device)
    
    num_tokens = len(token_expert_pairs)
    expert_blocks = []
    current_expert = -1
    block_start = 0
    
    for i, (_, expert_local, _) in enumerate(token_expert_pairs):
        if expert_local != current_expert:
            if current_expert != -1:
                expert_blocks.append((current_expert, block_start, i))
            current_expert = expert_local
            block_start = i
    expert_blocks.append((current_expert, block_start, num_tokens))
    
    num_blocks_m = (num_tokens + BLOCK_M - 1) // BLOCK_M
    expert_ids_per_block = torch.zeros(num_blocks_m, dtype=torch.int32, device=device)
    for expert, start, end in expert_blocks:
        start_block = start // BLOCK_M
        end_block = (end + BLOCK_M - 1) // BLOCK_M
        expert_ids_per_block[start_block:end_block] = expert
    
    # ----------------------------------------------------------------
    # 3) Launch Triton kernel for GEMM1
    # ----------------------------------------------------------------
    gemm1_output = torch.zeros((num_tokens, 2 * I), dtype=torch.float32, device=device)
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    num_pid_m = (num_tokens + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (2 * I + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    grid = (num_pid_m * num_pid_n,)
    
    gemm1_kernel[grid](
        hidden_states, gemm1_weights, gemm1_output,
        hidden_states_scale, gemm1_weights_scale,
        sorted_tokens, expert_ids_per_block,
        num_tokens,
        T, 2 * I, H,
        H, 1,
        2 * I * H, H, 1,
        2 * I, 1,
        H // BLOCK, 1,
        (2 * I // BLOCK) * (H // BLOCK), H // BLOCK, 1,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        128,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
    )
    
    # ----------------------------------------------------------------
    # 4) SwiGLU activation
    # ----------------------------------------------------------------
    X1 = gemm1_output[:, :I]
    X2 = gemm1_output[:, I:]
    C = torch.nn.functional.silu(X2) * X1
    C_bf16 = C.to(torch.bfloat16)
    
    # ----------------------------------------------------------------
    # 5) Launch Triton kernel for GEMM2
    # ----------------------------------------------------------------
    gemm2_output = torch.zeros((num_tokens, H), dtype=torch.float32, device=device)
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    num_pid_m = (num_tokens + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (H + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    grid = (num_pid_m * num_pid_n,)
    
    gemm2_kernel[grid](
        C_bf16, gemm2_weights, gemm2_output,
        gemm2_weights_scale,
        sorted_tokens, expert_ids_per_block,
        weights,
        num_tokens,
        local_start,
        num_tokens, H, I,
        I, 1,
        H * I, I, 1,
        H, 1,
        (H // BLOCK) * (I // BLOCK), I // BLOCK, 1,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        128, 256,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
    )
    
    # Accumulate expert results
    accum = torch.zeros((T, H), dtype=torch.float32, device=device)
    accum.index_add_(0, sorted_tokens.long(), gemm2_output)
    
    return accum.to(torch.bfloat16)

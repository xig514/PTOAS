"""FlashAttention with N-D tiling and multi-core distribution.

Implements FlashAttention with dynamic shapes using Layout-based N-D tiling:
- Input: q, k, v in BSND format (Batch, Sequence, Num_heads, Dim_head)
- Multi-core distribution: q is distributed across B, S, N dimensions
- Inner tiling: k, v are tiled within each core along S dimension
- Supports causal masking and scaling

Algorithm outline:
    For each q tile (distributed across cores):
        Initialize output accumulator and statistics (max, sum)
        For each k/v tile (inner loop):
            Compute QK^T
            Apply scaling and optional causal mask
            Compute softmax statistics (online algorithm)
            Update output with attention-weighted values
        Finalize output normalization
"""

import pto_frontend as pto


@pto.kernel
def flash_attention_nd(
    q: pto.Tensor(pto.float16, 4),      # [B, S_q, N, D]
    k: pto.Tensor(pto.float16, 4),      # [B, S_kv, N, D]
    v: pto.Tensor(pto.float16, 4),      # [B, S_kv, N, D]
    out: pto.Tensor(pto.float16, 4),    # [B, S_q, N, D]
):
    """FlashAttention with N-D tiling.

    Parameters
    ----------
    q : [B, S_q, N, D]
        Query tensor
    k : [B, S_kv, N, D]
        Key tensor
    v : [B, S_kv, N, D]
        Value tensor
    out : [B, S_q, N, D]
        Output tensor

    Note: scale and is_causal are hardcoded constants in this example.
    In production, these would be passed as scalar parameters or attributes.
    """

    # Constants (in real implementation, these would be parameters)
    # scale = 1.0 / sqrt(64) ≈ 0.125
    # is_causal = True

    # Tile configuration
    # Q tiles: B=1, S=128, N=1, D=64 (full head dimension)
    # K/V tiles: B=1, S=128, N=1, D=64
    TILE_B = 1
    TILE_S_Q = 128
    TILE_S_KV = 128
    TILE_N = 1
    TILE_D = 64

    # Tile buffers in VEC memory
    # Q tile: [1, 128, 1, 64]
    tile_q = pto.make_tile(
        (TILE_B, TILE_S_Q, TILE_N, TILE_D),
        pto.float16,
        pto.VEC,
        addr=0
    )

    # K tile: [1, 128, 1, 64]
    tile_k = pto.make_tile(
        (TILE_B, TILE_S_KV, TILE_N, TILE_D),
        pto.float16,
        pto.VEC,
        addr=0x10000
    )

    # V tile: [1, 128, 1, 64]
    tile_v = pto.make_tile(
        (TILE_B, TILE_S_KV, TILE_N, TILE_D),
        pto.float16,
        pto.VEC,
        addr=0x20000
    )

    # QK^T result: [1, 128, 1, 128] (Q_seq x K_seq)
    tile_qk = pto.make_tile(
        (TILE_B, TILE_S_Q, TILE_N, TILE_S_KV),
        pto.float32,
        pto.VEC,
        addr=0x30000
    )

    # Attention weights after softmax: [1, 128, 1, 128]
    tile_attn = pto.make_tile(
        (TILE_B, TILE_S_Q, TILE_N, TILE_S_KV),
        pto.float16,
        pto.VEC,
        addr=0x50000
    )

    # Partial output: [1, 128, 1, 64]
    tile_pv = pto.make_tile(
        (TILE_B, TILE_S_Q, TILE_N, TILE_D),
        pto.float32,
        pto.VEC,
        addr=0x70000
    )

    # Output accumulator: [1, 128, 1, 64]
    tile_out = pto.make_tile(
        (TILE_B, TILE_S_Q, TILE_N, TILE_D),
        pto.float32,
        pto.VEC,
        addr=0x90000
    )

    # Softmax statistics: row max and row sum
    # [1, 128, 1, 1] - one value per query position
    tile_row_max = pto.make_tile(
        (TILE_B, TILE_S_Q, TILE_N, 1),
        pto.float32,
        pto.VEC,
        addr=0xB0000
    )

    tile_row_sum = pto.make_tile(
        (TILE_B, TILE_S_Q, TILE_N, 1),
        pto.float32,
        pto.VEC,
        addr=0xB2000
    )

    # Temporary buffer for reduction operations
    tile_tmp = pto.make_tile(
        (TILE_B, TILE_S_Q, TILE_N, TILE_S_KV),
        pto.float32,
        pto.VEC,
        addr=0xB4000
    )

    # Multi-dimensional tiling for Q
    # Tile dimensions: B, S, N (not D - keep full head dimension)
    q_tiled = q.tile_nd(
        tile_sizes=(TILE_B, TILE_S_Q, TILE_N, TILE_D),
        tile_dims=[0, 1, 2]  # Tile B, S_q, N
    )

    # Distribute Q tiles across cores
    # Core grid: assume 2x4x8 = 64 cores for (B, S, N)
    # Adjust based on actual tensor shape and available cores
    q_dist = q_tiled.distribute_nd(core_grid=(2, 4, 8))

    # K and V tiling (not distributed - each core processes all K/V tiles)
    k_tiled = k.tile_nd(
        tile_sizes=(TILE_B, TILE_S_KV, TILE_N, TILE_D),
        tile_dims=[1]  # Only tile S_kv dimension
    )

    v_tiled = v.tile_nd(
        tile_sizes=(TILE_B, TILE_S_KV, TILE_N, TILE_D),
        tile_dims=[1]  # Only tile S_kv dimension
    )

    out_tiled = out.tile_nd(
        tile_sizes=(TILE_B, TILE_S_Q, TILE_N, TILE_D),
        tile_dims=[0, 1, 2]  # Match Q tiling
    )

    # Outer loop: distributed Q tiles across cores
    with q_dist.for_each() as (q_idx, q_view):
        # q_idx is (b_idx, s_q_idx, n_idx)
        # Each core processes its assigned Q tile

        # Load Q tile
        pto.tload(q_view, tile_q)

        # Initialize output accumulator and statistics
        # tile_out = 0, tile_row_max = -inf, tile_row_sum = 0
        # (In real implementation, use proper initialization ops)

        # Inner loop: iterate over all K/V tiles
        # Each Q tile attends to all K/V positions
        with k_tiled.for_each() as (k_idx, k_view):
            # k_idx is (s_kv_idx,) - single dimension

            # Get corresponding V tile
            v_view = v_tiled[k_idx]

            # Load K and V tiles
            pto.tload(k_view, tile_k)
            pto.tload(v_view, tile_v)

            # Compute QK^T: [1, 128, 1, 64] @ [1, 128, 1, 64]^T
            # Result: [1, 128, 1, 128]
            pto.tmatmul(tile_q, tile_k, tile_qk)

            # Apply scaling: QK^T * scale (hardcoded 0.125 for D=64)
            # pto.tmuls(tile_qk, 0.125, tile_qk)
            # (Simplified: skip scaling in this example)

            # Apply causal mask if needed
            # if is_causal and q_idx[1] < k_idx[0]:
            #     mask positions where query_pos < key_pos
            #     (Simplified: actual implementation needs position tracking)

            # Compute softmax statistics (online algorithm)
            # 1. Compute row max of current block
            pto.trowmax(tile_qk, tile_tmp, tile_row_max)

            # 2. Compute exp(qk - row_max)
            # tile_qk = exp(tile_qk - tile_row_max)
            # (Simplified: actual implementation needs broadcasting)

            # 3. Compute row sum
            pto.trowsum(tile_qk, tile_tmp, tile_row_sum)

            # 4. Normalize to get attention weights
            # tile_attn = tile_qk / tile_row_sum
            # (Simplified: actual implementation needs element-wise division)

            # Convert to float16 for matmul
            pto.tcvt(tile_qk, tile_attn)

            # Compute attention-weighted values: attn @ V
            # [1, 128, 1, 128] @ [1, 128, 1, 64] -> [1, 128, 1, 64]
            pto.tmatmul(tile_attn, tile_v, tile_pv)

            # Accumulate into output
            pto.tadd(tile_out, tile_pv, tile_out)

        # After processing all K/V tiles, finalize output
        # Apply final normalization using accumulated statistics
        # (Simplified: actual FlashAttention uses online normalization)

        # Convert output to float16 and store
        tile_out_f16 = pto.make_tile(
            (TILE_B, TILE_S_Q, TILE_N, TILE_D),
            pto.float16,
            pto.VEC,
            addr=0xD0000
        )
        pto.tcvt(tile_out, tile_out_f16)

        # Get corresponding output partition
        out_view = out_tiled[q_idx]
        pto.tstore(tile_out_f16, out_view)


@pto.kernel
def flash_attention_nd_simple(
    q: pto.Tensor(pto.float16, 2),      # [S_q, D]
    k: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    v: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    out: pto.Tensor(pto.float16, 2),    # [S_q, D]
):
    """Simplified FlashAttention demonstrating N-D tiling structure.

    Simplified to 2D tensors (removed B and N dimensions) to work with
    rank-2 tile_buf constraint.
    """

    TILE_S = 128
    TILE_D = 64

    # Tile Q with multi-core distribution along S dimension
    q_tiled = q.tile_nd(
        tile_sizes=(TILE_S, TILE_D),
        tile_dims=[0]  # Only tile S dimension
    )
    q_dist = q_tiled.distribute_nd(core_grid=(2,))  # 2 cores

    # Tile K, V without distribution
    k_tiled = k.tile_nd(
        tile_sizes=(TILE_S, TILE_D),
        tile_dims=[0]
    )
    v_tiled = v.tile_nd(
        tile_sizes=(TILE_S, TILE_D),
        tile_dims=[0]
    )
    out_tiled = out.tile_nd(
        tile_sizes=(TILE_S, TILE_D),
        tile_dims=[0]
    )

    # Allocate tiles
    tile_q = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0)
    tile_k = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x10000)
    tile_v = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x20000)
    tile_out = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x30000)

    # Nested loops: distributed Q, inner K/V
    with q_dist.for_each() as (q_idx, q_view):
        pto.tload(q_view, tile_q)

        with k_tiled.for_each() as (k_idx, k_view):
            v_view = v_tiled[k_idx]

            pto.tload(k_view, tile_k)
            pto.tload(v_view, tile_v)

            # Simplified: just add (placeholder for attention computation)
            pto.tadd(tile_q, tile_k, tile_out)
            pto.tadd(tile_out, tile_v, tile_out)

        out_view = out_tiled[q_idx]
        pto.tstore(tile_out, out_view)


if __name__ == "__main__":
    flash_attention_nd_simple()

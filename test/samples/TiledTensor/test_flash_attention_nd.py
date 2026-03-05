"""FlashAttention with N-D tiling and multi-core distribution.

Demonstrates the TiledTensor N-D API with a correct FlashAttention pipeline
using 2D tile_buf (PTO rank-2 constraint).

Uses the simplified 2D kernel (single head, single batch) with:
- distribute_nd for Q tile distribution across cores
- Nested K/V loop for inner computation
- Full Cube + Vector pipeline with sync
"""

import pto_frontend as pto


@pto.kernel
def flash_attention_nd(
    q: pto.Tensor(pto.float16, 2),      # [S_q, D]
    k: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    v: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    out: pto.Tensor(pto.float16, 2),    # [S_q, D]
):
    """FlashAttention using TiledTensorND with correct pipeline.

    Parameters
    ----------
    q  : [S_q, D]  — Query tensor (single head)
    k  : [S_kv, D] — Key tensor
    v  : [S_kv, D] — Value tensor
    out: [S_q, D]  — Output tensor
    """

    TILE = 32  # Br = Bc = D = 32

    # --- L1 (MAT) buffers ---
    q_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
    k_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                          addr=TILE * TILE * 2)
    v_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                          addr=TILE * TILE * 4)
    attn_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                             addr=TILE * TILE * 6)

    # --- L0 buffers for Cube ---
    q_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
    k_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)
    s_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)
    attn_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT,
                              addr=TILE * TILE * 2)
    v_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT,
                            addr=TILE * TILE * 2)
    o_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC,
                          addr=TILE * TILE * 4)

    # --- VEC buffers for softmax ---
    s_vec = pto.make_tile((TILE, TILE), pto.float32, pto.VEC, addr=0)
    tmp_vec = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                            addr=TILE * TILE * 4)
    attn_f16 = pto.make_tile((TILE, TILE), pto.float16, pto.VEC,
                             addr=TILE * TILE * 8)

    # --- N-D tiling ---
    # Tile Q along S dimension, distribute across 2 cores
    q_tiled = q.tile_nd(
        tile_sizes=(TILE, TILE),
        tile_dims=[0]  # Tile S dimension only
    )
    q_dist = q_tiled.distribute_nd(core_grid=(2,))

    # K and V: tile S dimension, no distribution
    k_tiled = k.tile_nd(
        tile_sizes=(TILE, TILE),
        tile_dims=[0]
    )
    v_tiled = v.tile_nd(
        tile_sizes=(TILE, TILE),
        tile_dims=[0]
    )
    out_tiled = out.tile_nd(
        tile_sizes=(TILE, TILE),
        tile_dims=[0]
    )

    # Outer loop: distributed Q tiles
    with q_dist.for_each() as (q_idx, q_view):
        # Load Q → MAT → LEFT
        pto.tload(q_view, q_mat)
        pto.record_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.wait_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.tmov(q_mat, q_left)

        # Inner loop: all K/V tiles
        with k_tiled.for_each() as (k_idx, k_view):
            v_view = v_tiled[k_idx]

            # Load K → MAT → RIGHT
            pto.tload(k_view, k_mat)
            pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.tmov(k_mat, k_right)

            # Matmul S = Q @ K^T
            pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.tmatmul(q_left, k_right, s_acc)

            # S: ACC → VEC for softmax
            pto.record_event(pto.TMATMUL, pto.TMOV_M2V, pto.EVENT_ID0)
            pto.wait_event(pto.TMATMUL, pto.TMOV_M2V, pto.EVENT_ID0)
            pto.tmov(s_acc, s_vec)

            # Softmax on VEC
            pto.record_event(pto.TMOV_M2V, pto.TVEC, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2V, pto.TVEC, pto.EVENT_ID0)
            pto.trowmax(s_vec, tmp_vec, tmp_vec)
            pto.trowexpandsub(s_vec, tmp_vec, s_vec)
            pto.texp(s_vec, s_vec)
            pto.trowsum(s_vec, tmp_vec, tmp_vec)
            pto.trowexpanddiv(s_vec, tmp_vec, s_vec)

            # attn weights: VEC → MAT → LEFT
            pto.tcvt(s_vec, attn_f16)
            pto.record_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
            pto.wait_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
            pto.tmov(attn_f16, attn_mat)
            pto.record_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.tmov(attn_mat, attn_left)

            # Load V → MAT → RIGHT
            pto.tload(v_view, v_mat)
            pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID2)
            pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID2)
            pto.tmov(v_mat, v_right)

            # Matmul O = attn @ V
            pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID2)
            pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID2)
            pto.tmatmul(attn_left, v_right, o_acc)

        # Store O: ACC → GM
        pto.record_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
        pto.wait_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
        out_view = out_tiled[q_idx]
        pto.tstore(o_acc, out_view)


if __name__ == "__main__":
    flash_attention_nd()

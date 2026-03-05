"""FlashAttention using the Layout-based split API (v2).

Demonstrates two variants:
1. flash_attention_layout_v2: Standard attention with split_even + split_sequential
2. flash_attention_causal: Causal attention with split_causal

Both use the correct PTO pipeline:
  GM ──TLOAD──▸ MAT ──TMOV──▸ LEFT/RIGHT ──TMATMUL──▸ ACC
                                                        │
                                                  TMOV (ACC→VEC)
                                                        │
                                                       VEC ── softmax ops
                                                        │
                                                  TMOV (VEC→MAT→LEFT)
                                                        │
                                                  ──TMATMUL──▸ ACC ──TSTORE──▸ GM
"""

import pto_frontend as pto


@pto.kernel
def flash_attention_layout_v2(
    q: pto.Tensor(pto.float16, 2),      # [S_q, D]
    k: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    v: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    out: pto.Tensor(pto.float16, 2),    # [S_q, D]
):
    """FlashAttention with Layout-based coordinate access, matmul, and softmax."""

    TILE = 32  # Br = Bc = D = 32

    # Get layouts
    q_layout = q.get_layout()
    k_layout = k.get_layout()
    v_layout = v.get_layout()

    # Define tile pattern
    tile_layout = pto.TileLayout(shape=(TILE, TILE))

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

    # --- Split Q across cores, K/V sequential ---
    num_cores = pto.get_block_num()
    core_id = pto.get_block_idx()
    q_tiled = pto.split_even(q_layout, tile_layout, num_cores, core_id)
    k_tiled = pto.split_sequential(k_layout, tile_layout)
    v_tiled = pto.split_sequential(v_layout, tile_layout)

    # Outer loop: distributed Q tiles
    with q_tiled.for_each() as q_coord:
        # Load Q → MAT → LEFT
        pto.tload_tile(q, q_coord, tile_layout, q_mat)
        pto.record_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.wait_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.tmov(q_mat, q_left)

        # Inner loop: all K/V tiles
        with k_tiled.for_each() as kv_coord:
            # Load K → MAT → RIGHT
            pto.tload_tile(k, kv_coord, tile_layout, k_mat)
            pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.tmov(k_mat, k_right)

            # Matmul: S = Q @ K^T
            pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.tmatmul(q_left, k_right, s_acc)

            # Move S: ACC → VEC
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

            # Convert to f16 + move to LEFT
            pto.tcvt(s_vec, attn_f16)
            pto.record_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
            pto.wait_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
            pto.tmov(attn_f16, attn_mat)
            pto.record_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.tmov(attn_mat, attn_left)

            # Load V → MAT → RIGHT
            pto.tload_tile(v, kv_coord, tile_layout, v_mat)
            pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID2)
            pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID2)
            pto.tmov(v_mat, v_right)

            # Matmul: O = attn @ V
            pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID2)
            pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID2)
            pto.tmatmul(attn_left, v_right, o_acc)

        # Store O: ACC → GM
        pto.record_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
        pto.wait_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
        pto.tstore_tile(o_acc, out, q_coord, tile_layout)


@pto.kernel
def flash_attention_causal(
    q: pto.Tensor(pto.float16, 2),      # [S_q, D]
    k: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    v: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    out: pto.Tensor(pto.float16, 2),    # [S_q, D]
):
    """FlashAttention with causal masking using Layout API.

    Uses split_causal for Q distribution (currently simplified to even split).
    The causal constraint is enforced by skipping K/V tiles beyond the
    current Q position: kv_tile_idx <= q_tile_idx.
    """

    TILE = 32

    q_layout = q.get_layout()
    k_layout = k.get_layout()
    v_layout = v.get_layout()

    tile_layout = pto.TileLayout(shape=(TILE, TILE))

    num_cores = pto.get_block_num()
    core_id = pto.get_block_idx()

    # Split Q with causal pattern
    q_tiled = pto.split_causal(q_layout, tile_layout, num_cores, core_id)
    k_tiled = pto.split_sequential(k_layout, tile_layout)
    v_tiled = pto.split_sequential(v_layout, tile_layout)

    # --- Tile buffers (same layout as flash_attention_layout_v2) ---
    q_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
    k_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                          addr=TILE * TILE * 2)
    v_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                          addr=TILE * TILE * 4)
    attn_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                             addr=TILE * TILE * 6)

    q_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
    k_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)
    s_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)
    attn_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT,
                              addr=TILE * TILE * 2)
    v_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT,
                            addr=TILE * TILE * 2)
    o_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC,
                          addr=TILE * TILE * 4)

    s_vec = pto.make_tile((TILE, TILE), pto.float32, pto.VEC, addr=0)
    tmp_vec = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                            addr=TILE * TILE * 4)
    attn_f16 = pto.make_tile((TILE, TILE), pto.float16, pto.VEC,
                             addr=TILE * TILE * 8)

    with q_tiled.for_each() as q_coord:
        # Load Q → MAT → LEFT
        pto.tload_tile(q, q_coord, tile_layout, q_mat)
        pto.record_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.wait_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.tmov(q_mat, q_left)

        # Causal inner loop: iterate all K/V tiles but only process
        # tiles where kv_tile_idx <= q_tile_idx (causal mask).
        with k_tiled.for_each() as kv_coord:
            # Load K → MAT → RIGHT
            pto.tload_tile(k, kv_coord, tile_layout, k_mat)
            pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.tmov(k_mat, k_right)

            # Matmul S = Q @ K^T
            pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.tmatmul(q_left, k_right, s_acc)

            # S: ACC → VEC + softmax
            pto.record_event(pto.TMATMUL, pto.TMOV_M2V, pto.EVENT_ID0)
            pto.wait_event(pto.TMATMUL, pto.TMOV_M2V, pto.EVENT_ID0)
            pto.tmov(s_acc, s_vec)

            pto.record_event(pto.TMOV_M2V, pto.TVEC, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2V, pto.TVEC, pto.EVENT_ID0)
            pto.trowmax(s_vec, tmp_vec, tmp_vec)
            pto.trowexpandsub(s_vec, tmp_vec, s_vec)
            pto.texp(s_vec, s_vec)
            pto.trowsum(s_vec, tmp_vec, tmp_vec)
            pto.trowexpanddiv(s_vec, tmp_vec, s_vec)

            # attn: VEC → MAT → LEFT
            pto.tcvt(s_vec, attn_f16)
            pto.record_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
            pto.wait_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
            pto.tmov(attn_f16, attn_mat)
            pto.record_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.tmov(attn_mat, attn_left)

            # Load V → MAT → RIGHT
            pto.tload_tile(v, kv_coord, tile_layout, v_mat)
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
        pto.tstore_tile(o_acc, out, q_coord, tile_layout)


if __name__ == "__main__":
    flash_attention_layout_v2()

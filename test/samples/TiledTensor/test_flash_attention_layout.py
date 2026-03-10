"""FlashAttention using TileLayout for explicit tile indexing.

Demonstrates correct FlashAttention pipeline with:
- Matmul using LEFT/RIGHT/ACC address spaces via Cube unit
- Softmax using VEC address space via Vector unit
- Data movement via MAT (L1) with tmov
- Pipeline synchronisation via record_event/wait_event

Pipeline flow:
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
def flash_attention_2d_layout(
    q: pto.Tensor(pto.float16, 2),      # [S_q, D]
    k: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    v: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    out: pto.Tensor(pto.float16, 2),    # [S_q, D]
):
    """FlashAttention with correct matmul, softmax, and pipeline sync.

    Uses TiledTensor.distribute for Q scheduling and for_each for K/V.

    Tile sizes: Br=Bc=D=32 (square for simplicity).
    Pipeline: GM → MAT → LEFT/RIGHT → ACC → VEC → MAT → LEFT → ACC → GM
    """

    TILE = 32  # Br = Bc = D = 32

    # --- L1 (MAT) buffers for DMA from GM ---
    q_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
    k_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                          addr=TILE * TILE * 2)
    v_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                          addr=TILE * TILE * 4)
    attn_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                             addr=TILE * TILE * 6)

    # --- L0 buffers for Cube matmul ---
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

    # --- Tiling & distribution ---
    q_tiled = q.tile(dim=0, tile_sizes=(TILE, TILE))
    k_tiled = k.tile(dim=0, tile_sizes=(TILE, TILE))
    v_tiled = v.tile(dim=0, tile_sizes=(TILE, TILE))
    out_tiled = out.tile(dim=0, tile_sizes=(TILE, TILE))

    q_dist = q_tiled.distribute()

    # Outer loop: this core's Q tiles
    with q_dist.for_each() as (q_tile_idx, q_view):
        # Step 1: Load Q from GM → MAT
        pto.tload(q_mat, q_view)
        pto.record_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.wait_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)

        # Step 2: Move Q from MAT → LEFT
        pto.tmov(q_left, q_mat)

        # Inner loop: all K/V tiles
        with k_tiled.for_each() as (kv_tile_idx, k_view):
            v_view = v_tiled[kv_tile_idx]

            # Step 3: Load K from GM → MAT → RIGHT
            pto.tload(k_mat, k_view)
            pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.tmov(k_right, k_mat)

            # Step 4: Matmul S = Q @ K^T via Cube unit
            pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.tmatmul(s_acc, q_left, k_right)

            # Step 5: Move S from ACC → VEC for softmax
            pto.record_event(pto.TMATMUL, pto.TMOV_M2V, pto.EVENT_ID0)
            pto.wait_event(pto.TMATMUL, pto.TMOV_M2V, pto.EVENT_ID0)
            pto.tmov(s_vec, s_acc)

            # Step 6: Softmax on VEC unit
            pto.record_event(pto.TMOV_M2V, pto.TVEC, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2V, pto.TVEC, pto.EVENT_ID0)

            pto.trowmax(tmp_vec, s_vec, tmp_vec)
            pto.trowexpandsub(s_vec, s_vec, tmp_vec)
            pto.texp(s_vec, s_vec)
            pto.trowsum(tmp_vec, s_vec, tmp_vec)
            pto.trowexpanddiv(s_vec, s_vec, tmp_vec)

            # Step 7: Convert attention weights to f16 for Cube matmul
            pto.tcvt(attn_f16, s_vec)

            # Step 8: Move attn weights VEC → MAT → LEFT
            pto.record_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
            pto.wait_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
            pto.tmov(attn_mat, attn_f16)

            pto.record_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.tmov(attn_left, attn_mat)

            # Step 9: Load V from GM → MAT → RIGHT
            pto.tload(v_mat, v_view)
            pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID2)
            pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID2)
            pto.tmov(v_right, v_mat)

            # Step 10: Matmul O = attn @ V via Cube unit
            pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID2)
            pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID2)
            pto.tmatmul(o_acc, attn_left, v_right)

        # Step 11: Store O from ACC → GM
        pto.record_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
        pto.wait_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
        out_view = out_tiled[q_tile_idx]
        pto.tstore(out_view, o_acc)


if __name__ == "__main__":
    flash_attention_2d_layout()

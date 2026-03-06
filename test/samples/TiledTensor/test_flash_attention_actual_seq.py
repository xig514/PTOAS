"""FlashAttention with actual_sequence_length using per-batch iteration.

Demonstrates two scenes:
1. coord.select() + coord_combine(): Cross-tensor coordinate composition for PSE
2. actual_sequence_length with per-batch Q/K range computation:
   - Q/K tensors packed as [total_tokens, D]
   - actual_seq_len_q/k: [B+1] inclusive prefix sum with leading 0
     e.g. batch lengths [50, 150, 100] -> actual_seq_len = [0, 50, 200, 300]
   - Each core loops over all B batches
   - Within each batch, Q tiles distributed round-robin across cores
   - K tiles iterated using cumsum boundaries directly as loop bounds

Pipeline:
  GM --TLOAD--> MAT --TMOV--> LEFT/RIGHT --TMATMUL--> ACC
                                                       |
                                                TMOV (ACC->VEC)
                                                       |
                                                      VEC -- softmax
                                                       |
                                                TMOV (VEC->MAT->LEFT)
                                                       |
                                               --TMATMUL--> ACC --TSTORE--> GM
"""

import pto_frontend as pto


# Helper: convert ScalarValue to index type for arithmetic
def _to_index(val):
    """Convert a ScalarValue (possibly i64) to index type."""
    from pto_frontend._utils import ensure_index_ssa
    from pto_frontend._scalar import ScalarValue
    return ScalarValue(ensure_index_ssa(val))


# ============================================================================
# Scene 1: coord.select() + coord_combine() — PSE with 4D tensors
# ============================================================================

@pto.kernel
def flash_attention_with_pse(
    q:   pto.Tensor(pto.float16, 4),   # [B, N, Sq, D]
    k:   pto.Tensor(pto.float16, 4),   # [B, N, Sk, D]
    v:   pto.Tensor(pto.float16, 4),   # [B, N, Sk, D]
    pse: pto.Tensor(pto.float16, 4),   # [B, N, Sq, Sk]
    out: pto.Tensor(pto.float16, 4),   # [B, N, Sq, D]
):
    """FlashAttention with Position-Shift Encoding (PSE).

    PSE is indexed by [B, N, Sq, Sk]. During the Q*K loop we need to
    compose a PSE tile coordinate from Q-coord axes [B, N, Sq] and
    K-coord axis [Sk].
    """
    TILE = 32

    # Layouts
    q_layout   = q.get_layout()
    k_layout   = k.get_layout()
    pse_layout = pse.get_layout()

    # Tile layouts: iterate B, N as size-1 tiles; Sq/Sk/D as TILE
    q_tile   = pto.TileLayout(shape=(1, 1, TILE, TILE))
    k_tile   = pto.TileLayout(shape=(1, 1, TILE, TILE))
    pse_tile = pto.TileLayout(shape=(1, 1, TILE, TILE))

    num_cores = pto.get_block_num()
    core_id   = pto.get_block_idx()

    # Q: split across cores on dim-0 (B); K: sequential inner loop
    q_tiled = pto.split_even(q_layout, q_tile, num_cores, core_id)
    k_tiled = pto.split_sequential(k_layout, k_tile)

    # --- Tile buffers ---
    q_mat     = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
    k_mat     = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 2)
    v_mat     = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 4)
    attn_mat  = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 6)

    q_left    = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
    k_right   = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)
    s_acc     = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)
    attn_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT,
                              addr=TILE * TILE * 2)
    v_right   = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT,
                              addr=TILE * TILE * 2)
    o_acc     = pto.make_tile((TILE, TILE), pto.float32, pto.ACC,
                              addr=TILE * TILE * 4)

    s_vec     = pto.make_tile((TILE, TILE), pto.float32, pto.VEC, addr=0)
    pse_f16   = pto.make_tile((TILE, TILE), pto.float16, pto.VEC,
                              addr=TILE * TILE * 4)
    pse_f32   = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                              addr=TILE * TILE * 8)
    tmp_vec   = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                              addr=TILE * TILE * 12)
    attn_f16  = pto.make_tile((TILE, TILE), pto.float16, pto.VEC,
                              addr=TILE * TILE * 16)

    with q_tiled.for_each() as q_coord:
        # q_coord = (b_idx, n_idx, sq_idx, d_idx)

        # Load Q tile -> MAT -> LEFT
        pto.tload_tile(q, q_coord, q_tile, q_mat)
        pto.record_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.wait_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.tmov(q_mat, q_left)

        with k_tiled.for_each() as k_coord:
            # k_coord = (b_idx2, n_idx2, sk_idx, d_idx2)

            # ========== coord.select() + coord_combine() ==========
            # Compose PSE coordinate from Q-axes [B,N,Sq] + K-axis [Sk]
            pse_coord = pto.coord_combine(
                q_coord.select(0, 1, 2),   # [B, N, Sq] from Q
                k_coord.select(2),          # [Sk]       from K
            )
            # pse_coord = (b_idx, n_idx, sq_idx, sk_idx) -- matches PSE [B,N,Sq,Sk]

            # Load PSE tile -> VEC
            pto.tload_tile(pse, pse_coord, pse_tile, pse_f16)

            # Load K tile -> MAT -> RIGHT
            pto.tload_tile(k, k_coord, k_tile, k_mat)
            pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.tmov(k_mat, k_right)

            # S = Q @ K^T
            pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.tmatmul(q_left, k_right, s_acc)

            # S: ACC -> VEC
            pto.record_event(pto.TMATMUL, pto.TMOV_M2V, pto.EVENT_ID0)
            pto.wait_event(pto.TMATMUL, pto.TMOV_M2V, pto.EVENT_ID0)
            pto.tmov(s_acc, s_vec)

            # S += PSE (add position shift encoding)
            pto.record_event(pto.TMOV_M2V, pto.TVEC, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2V, pto.TVEC, pto.EVENT_ID0)
            pto.tcvt(pse_f16, pse_f32)
            pto.tadd(s_vec, pse_f32, s_vec)

            # Softmax
            pto.trowmax(s_vec, tmp_vec, tmp_vec)
            pto.trowexpandsub(s_vec, tmp_vec, s_vec)
            pto.texp(s_vec, s_vec)
            pto.trowsum(s_vec, tmp_vec, tmp_vec)
            pto.trowexpanddiv(s_vec, tmp_vec, s_vec)

            # attn: VEC -> MAT -> LEFT
            pto.tcvt(s_vec, attn_f16)
            pto.record_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
            pto.wait_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
            pto.tmov(attn_f16, attn_mat)
            pto.record_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.tmov(attn_mat, attn_left)

            # Load V (reuse k_coord for V since V is [B,N,Sk,D])
            pto.tload_tile(v, k_coord, k_tile, v_mat)
            pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID2)
            pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID2)
            pto.tmov(v_mat, v_right)

            # O = attn @ V
            pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID2)
            pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID2)
            pto.tmatmul(attn_left, v_right, o_acc)

        # Store O
        pto.record_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
        pto.wait_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
        pto.tstore_tile(o_acc, out, q_coord, tile_layout=q_tile)


# ============================================================================
# Scene 2: actual_sequence_length — per-batch iteration
# ============================================================================

@pto.kernel
def flash_attention_actual_seq(
    q:   pto.Tensor(pto.float16, 2),   # [total_q_tokens, D]  -- packed
    k:   pto.Tensor(pto.float16, 2),   # [total_k_tokens, D]  -- packed
    v:   pto.Tensor(pto.float16, 2),   # [total_k_tokens, D]  -- packed
    out: pto.Tensor(pto.float16, 2),   # [total_q_tokens, D]
    actual_seq_len_q: pto.Tensor(pto.int32, 1),  # [B+1] prefix sum with leading 0
    actual_seq_len_k: pto.Tensor(pto.int32, 1),  # [B+1] prefix sum with leading 0
):
    """FlashAttention with variable-length sequences (actual_sequence_length).

    Q/K/V are packed into flat [total_tokens, D] tensors. Sequence
    boundaries are given by prefix-sum arrays with a leading zero:

        actual_seq_len_q = [0, cumsum_1, ..., cumsum_B]   (B+1 elements)

    For batch b (0-indexed):
        q_start = actual_seq_len_q[b]
        q_end   = actual_seq_len_q[b + 1]

    Example:
        Batch lengths = [50, 150, 100]
        actual_seq_len_q = [0, 50, 200, 300]  (B+1 = 4 elements)
        B = 3

    Each core loops over all B batches. Within each batch, Q tiles are
    distributed round-robin across cores using element-offset stepping.
    K tiles are iterated with cumsum boundaries directly as loop bounds.
    """
    TILE = 32

    num_cores = pto.get_block_num()
    core_id   = pto.get_block_idx()

    # Convert to index type for arithmetic with index-typed get_value results
    num_cores_idx = _to_index(num_cores)
    core_id_idx   = _to_index(core_id)

    # B = len(actual_seq_len_q) - 1
    B = actual_seq_len_q.shape[0] - 1

    # --- Tile buffers ---
    q_mat     = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
    k_mat     = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 2)
    v_mat     = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 4)
    attn_mat  = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 6)

    q_left    = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
    k_right   = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)
    s_acc     = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)
    attn_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT,
                              addr=TILE * TILE * 2)
    v_right   = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT,
                              addr=TILE * TILE * 2)
    o_acc     = pto.make_tile((TILE, TILE), pto.float32, pto.ACC,
                              addr=TILE * TILE * 4)

    s_vec     = pto.make_tile((TILE, TILE), pto.float32, pto.VEC, addr=0)
    tmp_vec   = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                              addr=TILE * TILE * 4)
    attn_f16  = pto.make_tile((TILE, TILE), pto.float16, pto.VEC,
                              addr=TILE * TILE * 8)

    # Outer loop: iterate over batches
    with pto.for_range(0, B) as b:
        b_next = b + 1

        # Load Q/K token boundaries for this batch
        q_start = pto.get_value(actual_seq_len_q, b)
        q_end   = pto.get_value(actual_seq_len_q, b_next)
        k_start = pto.get_value(actual_seq_len_k, b)
        k_end   = pto.get_value(actual_seq_len_k, b_next)

        # Q tiles: round-robin distribution across cores
        # Core i handles element offsets:
        #   q_start + i*TILE, q_start + (i+N)*TILE, q_start + (i+2N)*TILE, ...
        q_loop_start = q_start + core_id_idx * TILE
        q_loop_step  = num_cores_idx * TILE

        with pto.for_range(q_loop_start, q_end, q_loop_step) as q_elem:
            # Load Q tile via explicit partition (handles non-aligned offsets)
            q_pv = q.partition(offsets=[q_elem, 0], sizes=[TILE, TILE])
            pto.tload(q_pv, q_mat)
            pto.record_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.wait_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
            pto.tmov(q_mat, q_left)

            # K tile loop: iterate element offsets with TILE step
            # k_start and k_end used directly as loop bounds
            with pto.for_range(k_start, k_end, TILE) as k_elem:
                # Load K -> MAT -> RIGHT
                k_pv = k.partition(offsets=[k_elem, 0], sizes=[TILE, TILE])
                pto.tload(k_pv, k_mat)
                pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
                pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
                pto.tmov(k_mat, k_right)

                # S = Q @ K^T
                pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
                pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
                pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
                pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
                pto.tmatmul(q_left, k_right, s_acc)

                # S: ACC -> VEC
                pto.record_event(pto.TMATMUL, pto.TMOV_M2V, pto.EVENT_ID0)
                pto.wait_event(pto.TMATMUL, pto.TMOV_M2V, pto.EVENT_ID0)
                pto.tmov(s_acc, s_vec)

                # Softmax
                pto.record_event(pto.TMOV_M2V, pto.TVEC, pto.EVENT_ID0)
                pto.wait_event(pto.TMOV_M2V, pto.TVEC, pto.EVENT_ID0)
                pto.trowmax(s_vec, tmp_vec, tmp_vec)
                pto.trowexpandsub(s_vec, tmp_vec, s_vec)
                pto.texp(s_vec, s_vec)
                pto.trowsum(s_vec, tmp_vec, tmp_vec)
                pto.trowexpanddiv(s_vec, tmp_vec, s_vec)

                # attn: VEC -> MAT -> LEFT
                pto.tcvt(s_vec, attn_f16)
                pto.record_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
                pto.wait_event(pto.TVEC, pto.TMOV_V2M, pto.EVENT_ID0)
                pto.tmov(attn_f16, attn_mat)
                pto.record_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
                pto.wait_event(pto.TMOV_V2M, pto.TMOV_M2L, pto.EVENT_ID0)
                pto.tmov(attn_mat, attn_left)

                # Load V (same element offset as K)
                v_pv = v.partition(offsets=[k_elem, 0], sizes=[TILE, TILE])
                pto.tload(v_pv, v_mat)
                pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID2)
                pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID2)
                pto.tmov(v_mat, v_right)

                # O = attn @ V
                pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
                pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
                pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID2)
                pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID2)
                pto.tmatmul(attn_left, v_right, o_acc)

            # Store O tile
            pto.record_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
            pto.wait_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
            out_pv = out.partition(offsets=[q_elem, 0], sizes=[TILE, TILE])
            pto.tstore(out_pv, o_acc)


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Scene 1: FlashAttention with PSE (coord select + combine)")
    print("=" * 60)
    ir1 = flash_attention_with_pse.emit_ir()
    print(ir1)

    print()
    print("=" * 60)
    print("Scene 2: FlashAttention actual_seq (per-batch iteration)")
    print("=" * 60)
    ir2 = flash_attention_actual_seq.emit_ir()
    print(ir2)

    # Write IR files for ptoas verification
    with open("/tmp/fa_pse.pto", "w") as f:
        f.write(ir1)
    with open("/tmp/fa_actual_seq.pto", "w") as f:
        f.write(ir2)

    print()
    print("IR files written to /tmp/fa_pse.pto and /tmp/fa_actual_seq.pto")
    print("Verify with: ptoas /tmp/fa_pse.pto --pto-level=level3")
    print("             ptoas /tmp/fa_actual_seq.pto --pto-level=level3")

    # Example host-side conversion:
    # actual_seq_lengths = [50, 150, 100]  (per-batch lengths)
    # actual_seq_len_q = [0] + list(itertools.accumulate(actual_seq_lengths))
    #                  = [0, 50, 200, 300]  (B+1 elements, passed to kernel)

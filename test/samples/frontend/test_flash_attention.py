"""FlashAttention implementation using PTOAS Python frontend.

Step-by-step verification against C++ reference (fa_performance_kernel.cpp):
  Step 1: QK matmul only — S = Q @ K^T  (Cube section)
  Step 2: Full streaming FlashAttention with cross-core Cube+Vector pipeline

Key insight on matmul layout:
  tmatmul(acc, left, right) computes standard C = A @ B.
  To get Q @ K^T, we pass K^T[D, Skv] as input (host transposes K),
  or use tload with layout="DN" to load K[Skv, D] via a DN tensor view into
  a ZN MAT tile so the hardware performs the transpose during DMA.
  For P @ V, standard matmul P[Sq, Skv] @ V[Skv, D] works directly.

Algorithm (FA2.0 streaming):
  For each Q tile (distributed across cores):
    For each KV tile j:
      [Cube]   S = Q @ K_j^T                           (QK matmul)
      [Vector] FlashSoftmax: P = exp((S - max) * scale) (no /sum!)
               maintain running max & sum
      [Cube]   PV = P @ V_j                            (PV matmul)
      [Vector] GlobalUpdate: O = O * exp_corr + PV     (running rescale + accumulate)
    [Vector] O = O / global_sum                         (final normalization)

Usage:
    python3 test_flash_attention.py                    # Full NPU test
    python3 test_flash_attention.py --ir-only          # IR inspection only
    python3 test_flash_attention.py --step1            # QK matmul only
    python3 test_flash_attention.py --full             # Full FA only
    python3 test_flash_attention.py --ktranspose       # K auto-transpose experiment
    python3 test_flash_attention.py --multicore        # Multi-core FA (Sq=8192)
    python3 test_flash_attention.py --double-buffer    # Double-buffered FA
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend"))

import pto_frontend as pto

TILE = 64  # Tile size for all dimensions (S0, S1, D)

# Cross-core sync event IDs (FFTS flags)
QK_READY = 0
P_READY = 1
PV_READY = 2


# ===================================================================
#  Kernel factory functions — each creates fresh DynVars to avoid
#  cross-kernel tracing interference from module-level @pto.kernel
# ===================================================================

def _make_qk_kernel():
    """Create QK matmul kernel: S[Sq, Skv] = Q[Sq, D] @ Kt[D, Skv]."""
    Sq = pto.DynVar("Sq")
    Skv = pto.DynVar("Skv")
    D = pto.DynVar("D")

    @pto.kernel
    def flash_attn_qk_kernel(
        q: pto.Tensor[[Sq, D], pto.float16],
        kt: pto.Tensor[[D, Skv], pto.float16],
        s: pto.Tensor[[Sq, Skv], pto.float16],
    ):
        with pto.section_cube():
            q_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
            k_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                                  addr=TILE * TILE * 2)
            q_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
            k_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)
            s_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)

            sq_tiles = (Sq + (TILE - 1)) // TILE
            skv_tiles = (Skv + (TILE - 1)) // TILE
            d_tiles = (D + (TILE - 1)) // TILE

            for i in pto.range(sq_tiles):
                for j in pto.range(skv_tiles):
                    sq_off = i * TILE
                    skv_off = j * TILE

                    # First D tile
                    pto.tload(q_mat, q, offsets=[sq_off, 0])
                    pto.tload(k_mat, kt, offsets=[0, skv_off])
                    pto.tmov(q_left, q_mat)
                    pto.tmov(k_right, k_mat)
                    pto.tmatmul(s_acc, q_left, k_right)

                    # Remaining D tiles
                    for d in pto.range(1, d_tiles):
                        d_off = d * TILE
                        pto.tload(q_mat, q, offsets=[sq_off, d_off])
                        pto.tload(k_mat, kt, offsets=[d_off, skv_off])
                        pto.tmov(q_left, q_mat)
                        pto.tmov(k_right, k_mat)
                        pto.tmatmul_acc(s_acc, s_acc, q_left, k_right)

                    pto.tstore(s, s_acc, offsets=[sq_off, skv_off])

    return flash_attn_qk_kernel


def _make_fa_kernel():
    """Create full streaming FlashAttention kernel."""
    Sq = pto.DynVar("Sq")
    Skv = pto.DynVar("Skv")
    D = pto.DynVar("D")

    @pto.kernel
    def flash_attention_kernel(
        q: pto.Tensor[[Sq, D], pto.float16],
        kt: pto.Tensor[[D, Skv], pto.float16],
        v: pto.Tensor[[Skv, D], pto.float16],
        o: pto.Tensor[[Sq, D], pto.float16],
        qk_buf: pto.Tensor[[Sq, Skv], pto.float32],
        p_buf: pto.Tensor[[Sq, Skv], pto.float16],
        pv_buf: pto.Tensor[[Sq, D], pto.float32],
    ):
        scale = 1.0 / math.sqrt(TILE)
        skv_tiles = (Skv + (TILE - 1)) // TILE

        # =================== CUBE SECTION ===================
        with pto.section_cube():
            q_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
            k_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                                  addr=TILE * TILE * 2)
            p_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                                  addr=TILE * TILE * 4)
            v_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                                  addr=TILE * TILE * 6)
            q_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
            k_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT,
                                    addr=0)
            qk_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)
            p_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT,
                                   addr=TILE * TILE * 2)
            v_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT,
                                    addr=TILE * TILE * 2)
            pv_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC,
                                   addr=TILE * TILE * 4)

            core_id = pto.get_block_idx()
            sq_off = core_id * TILE

            # Load Q once
            pto.tload(q_mat, q, offsets=[sq_off, 0])
            pto.tmov(q_left, q_mat)

            # ---- First KV tile (j=0) ----
            # QK_0 = Q @ Kt_0
            pto.tload(k_mat, kt, offsets=[0, 0])
            pto.tmov(k_right, k_mat)
            pto.tmatmul(qk_acc, q_left, k_right)
            pto.tstore(qk_buf, qk_acc, offsets=[sq_off, 0])
            pto.sync_set(pto.PIPE_FIX, QK_READY)

            pto.sync_wait(pto.PIPE_M, P_READY)

            # PV_0 = P_0 @ V_0
            pto.tload(p_mat, p_buf, offsets=[sq_off, 0])
            pto.tmov(p_left, p_mat)
            pto.tload(v_mat, v, offsets=[0, 0])
            pto.tmov(v_right, v_mat)
            pto.tmatmul(pv_acc, p_left, v_right)
            pto.tstore(pv_buf, pv_acc, offsets=[sq_off, 0])
            pto.sync_set(pto.PIPE_FIX, PV_READY)

            # ---- Remaining KV tiles ----
            for j in pto.range(1, skv_tiles):
                skv_off = j * TILE
                pto.tload(k_mat, kt, offsets=[0, skv_off])
                pto.tmov(k_right, k_mat)
                pto.tmatmul(qk_acc, q_left, k_right)
                pto.tstore(qk_buf, qk_acc, offsets=[sq_off, skv_off])
                pto.sync_set(pto.PIPE_FIX, QK_READY)

                pto.sync_wait(pto.PIPE_M, P_READY)

                pto.tload(p_mat, p_buf, offsets=[sq_off, skv_off])
                pto.tmov(p_left, p_mat)
                pto.tload(v_mat, v, offsets=[skv_off, 0])
                pto.tmov(v_right, v_mat)
                pto.tmatmul(pv_acc, p_left, v_right)
                pto.tstore(pv_buf, pv_acc, offsets=[sq_off, 0])
                pto.sync_set(pto.PIPE_FIX, PV_READY)

        # =================== VECTOR SECTION ===================
        # Each sub-block processes HALF rows (sub0: rows [0,HALF), sub1: rows [HALF,TILE))
        # Both sub-blocks participate in sync_wait/sync_set (required to avoid deadlock)
        HALF = TILE // 2
        with pto.section_vector():
            qk_vec = pto.make_tile((HALF, TILE), pto.float32, pto.VEC, addr=0)
            tmp_vec = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                    addr=HALF * TILE * 4)
            p_f16 = pto.make_tile((HALF, TILE), pto.float16, pto.VEC,
                                   addr=HALF * TILE * 8)
            reduce_dst = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                        addr=HALF * TILE * 10,
                                        valid_shape=(HALF, 1))
            global_max = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                        addr=HALF * TILE * 14)
            global_sum = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                        addr=HALF * TILE * 18)
            running_o = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                       addr=HALF * TILE * 22)
            exp_corr = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                      addr=HALF * TILE * 26)

            core_id = pto.get_block_idx()
            sq_off = core_id * TILE
            sub_id = pto.get_subblock_idx()
            row_off = sub_id * HALF  # sub0→row 0, sub1→row HALF

            # ---- First KV tile: FlashSoftmax INIT + GU INIT ----
            pto.sync_wait(pto.PIPE_V, QK_READY)
            pto.tload(qk_vec, qk_buf, offsets=[sq_off + row_off, 0])

            pto.trowmax(reduce_dst, qk_vec, tmp_vec)
            pto.trowexpand(global_max, reduce_dst)
            pto.tsub(tmp_vec, qk_vec, global_max)
            pto.tmuls(tmp_vec, tmp_vec, scale)
            pto.texp(qk_vec, tmp_vec)
            pto.trowsum(reduce_dst, qk_vec, tmp_vec)
            pto.trowexpand(global_sum, reduce_dst)
            pto.tcvt(p_f16, qk_vec)

            pto.tstore(p_buf, p_f16, offsets=[sq_off + row_off, 0])
            pto.sync_set(pto.PIPE_MTE3, P_READY)

            pto.sync_wait(pto.PIPE_V, PV_READY)
            pto.tload(running_o, pv_buf, offsets=[sq_off + row_off, 0])

            # ---- Remaining KV tiles: FlashSoftmax UPDATE + GU ----
            for j in pto.range(1, skv_tiles):
                skv_off = j * TILE
                pto.sync_wait(pto.PIPE_V, QK_READY)
                pto.tload(qk_vec, qk_buf,
                          offsets=[sq_off + row_off, skv_off])

                pto.trowmax(reduce_dst, qk_vec, tmp_vec)
                pto.trowexpand(tmp_vec, reduce_dst)
                pto.tmax(tmp_vec, tmp_vec, global_max)
                pto.tsub(exp_corr, global_max, tmp_vec)
                pto.tmuls(exp_corr, exp_corr, scale)
                pto.texp(exp_corr, exp_corr)
                pto.tmul(global_sum, global_sum, exp_corr)
                pto.tmul(running_o, running_o, exp_corr)
                pto.tmuls(global_max, tmp_vec, 1.0)
                pto.tsub(tmp_vec, qk_vec, global_max)
                pto.tmuls(tmp_vec, tmp_vec, scale)
                pto.texp(qk_vec, tmp_vec)
                pto.trowsum(reduce_dst, qk_vec, tmp_vec)
                pto.trowexpand(tmp_vec, reduce_dst)
                pto.tadd(global_sum, global_sum, tmp_vec)
                pto.tcvt(p_f16, qk_vec)
                pto.tstore(p_buf, p_f16,
                           offsets=[sq_off + row_off, skv_off])
                pto.sync_set(pto.PIPE_MTE3, P_READY)

                # GlobalUpdate: O = O * exp_correction + PV_j
                pto.sync_wait(pto.PIPE_V, PV_READY)
                pto.tload(qk_vec, pv_buf,
                          offsets=[sq_off + row_off, 0])
                pto.tadd(running_o, running_o, qk_vec)

            # ---- Final: O = running_o / global_sum ----
            pto.tdiv(running_o, running_o, global_sum)
            pto.tcvt(p_f16, running_o)
            pto.tstore(o, p_f16, offsets=[sq_off + row_off, 0])

    return flash_attention_kernel


def _make_k_autotranspose_kernel():
    """Create QK matmul kernel that avoids host K transpose.

    K[Skv, D] stored row-major in global memory is the same byte layout as
    K^T[D, Skv] in column-major. We create a DN (column-major) tensor view
    of shape [D, Skv] so that TLOAD can do DN→ZN conversion into the ZN
    MAT tile. The matmul hardware then interprets ZN data as the transposed
    matrix, yielding Q @ K^T.

    Hardware constraint discovered:
    - TLOAD MAT only supports ND→NZ or DN→ZN (not ND→ZN).
    - The ZN tile (BLayout=RowMajor, SLayout=ColMajor) requires DN GlobalTensor.
    - K[Skv,D] row-major IS K^T[D,Skv] col-major (same bytes), so a DN view works.
    - However, the ptoas EmitC lowering currently always emits Layout::ND for
      GlobalTensors. Supporting Layout::DN requires changes to the lowering pass
      (lib/PTO/Transforms) — specifically, the stride pattern [1, D] for shape
      [D, Skv] must be recognized as column-major and emitted as Layout::DN.

    Status: IR generation works; compile fails until EmitC supports DN layout.
    """
    from mlir.dialects import pto as _pto

    Sq = pto.DynVar("Sq")
    Skv = pto.DynVar("Skv")
    D = pto.DynVar("D")

    @pto.kernel
    def k_autotranspose_kernel(
        q: pto.Tensor[[Sq, D], pto.float16],
        k: pto.Tensor[[Skv, D], pto.float16],   # K, NOT K^T
        s: pto.Tensor[[Sq, Skv], pto.float16],
    ):
        with pto.section_cube():
            q_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
            # ZN layout: BLayout=RowMajor, SLayout=ColMajor (matches RIGHT)
            k_mat_zn = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                                     addr=TILE * TILE * 2,
                                     blayout=_pto.BLayout.RowMajor,
                                     slayout=_pto.SLayout.ColMajor)
            q_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
            k_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)
            s_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)

            sq_tiles = (Sq + (TILE - 1)) // TILE
            skv_tiles = (Skv + (TILE - 1)) // TILE
            d_tiles = (D + (TILE - 1)) // TILE

            for i in pto.range(sq_tiles):
                for j in pto.range(skv_tiles):
                    sq_off = i * TILE
                    skv_off = j * TILE

                    # Load Q normally (ND→NZ, standard path)
                    pto.tload(q_mat, q, offsets=[sq_off, 0])

                    # Load K[skv_off:, 0:] via DN tensor view → ZN MAT tile.
                    # K[Skv,D] row-major = K^T[D,Skv] col-major (DN layout).
                    # Create partition from a DN view of K as [D, Skv].
                    _k_tload_dn(k_mat_zn, k, d_off=0, skv_off=skv_off)

                    pto.tmov(q_left, q_mat)
                    pto.tmov(k_right, k_mat_zn)
                    pto.tmatmul(s_acc, q_left, k_right)

                    # Remaining D tiles
                    for d in pto.range(1, d_tiles):
                        d_off = d * TILE
                        pto.tload(q_mat, q, offsets=[sq_off, d_off])
                        _k_tload_dn(k_mat_zn, k, d_off=d_off, skv_off=skv_off)
                        pto.tmov(q_left, q_mat)
                        pto.tmov(k_right, k_mat_zn)
                        pto.tmatmul_acc(s_acc, s_acc, q_left, k_right)

                    pto.tstore(s, s_acc, offsets=[sq_off, skv_off])

    return k_autotranspose_kernel


def _k_tload_dn(tile, k_tensor, d_off, skv_off):
    """Load a TILE x TILE block from K[Skv,D] using a DN (column-major) view.

    K[Skv, D] row-major has the same byte layout as K^T[D, Skv] col-major.
    We create a tensor view of shape [D, Skv] with col-major strides [1, D]
    which gives DN layout. Then TLOAD does DN→ZN into the ZN MAT tile.

    The partition offsets are [d_off, skv_off] in the transposed [D, Skv] view.

    NOTE: This currently generates correct IR but the ptoas EmitC lowering
    emits Layout::ND instead of Layout::DN. Needs lowering pass update.
    """
    from mlir.dialects import pto as _pto
    from pto_frontend._utils import ensure_index_ssa
    from pto_frontend._sync_tracker import get_sync_tracker
    from pto_frontend._ir_builder import get_builder
    from mlir.dialects.pto import PIPE

    builder = get_builder()

    # K has shape [Skv, D], strides row-major: [D, 1]
    # We view it as [D, Skv] with col-major strides: [1, D]
    # This is the same memory layout — just reinterpreted as DN.
    skv_ssa = k_tensor._shape_ssas[0]  # Skv
    d_ssa = k_tensor._shape_ssas[1]    # D

    # Transposed shape: [D, Skv]
    tv_shape = [d_ssa, skv_ssa]
    # Column-major strides for [D, Skv]: stride = [1, D]
    tv_strides = [builder.constant_index(1), d_ssa]

    tv_type = _pto.TensorViewType.get(2, k_tensor.dtype.to_mlir())
    tv = _pto.MakeTensorViewOp(
        tv_type, k_tensor.ptr_ssa, tv_shape, tv_strides
    ).result

    # Partition at [d_off, skv_off] with sizes [TILE, TILE]
    off_ssas = [ensure_index_ssa(d_off), ensure_index_ssa(skv_off)]
    sz_ssas = [builder.constant_index(TILE), builder.constant_index(TILE)]
    static_sizes = [TILE, TILE]

    pv_type = _pto.PartitionTensorViewType.get(
        static_sizes, k_tensor.dtype.to_mlir()
    )
    pv = _pto.PartitionViewOp(
        pv_type, tv, offsets=off_ssas, sizes=sz_ssas
    ).result

    # Record sync and emit TLOAD
    tracker = get_sync_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_MTE2, reads=[], writes=[tile])
    _pto.TLoadOp(None, pv, tile.ssa)


def _make_multicore_fa_kernel():
    """Create multi-core streaming FlashAttention kernel.

    Distributes Sq tiles across Cube cores with strided outer loop:
        for i in range(core_id, sq_tiles, num_cores)

    Each core processes one Sq tile at a time, iterating through all KV tiles.
    Scratch buffers (qk_buf, p_buf, pv_buf) are indexed by core_id to avoid
    cross-core conflicts.
    """
    Sq = pto.DynVar("Sq")
    Skv = pto.DynVar("Skv")
    D = pto.DynVar("D")
    ScratchSq = pto.DynVar("ScratchSq")  # = num_cores * TILE

    @pto.kernel
    def multicore_fa_kernel(
        q: pto.Tensor[[Sq, D], pto.float16],
        kt: pto.Tensor[[D, Skv], pto.float16],
        v: pto.Tensor[[Skv, D], pto.float16],
        o: pto.Tensor[[Sq, D], pto.float16],
        qk_buf: pto.Tensor[[ScratchSq, Skv], pto.float32],
        p_buf: pto.Tensor[[ScratchSq, Skv], pto.float16],
        pv_buf: pto.Tensor[[ScratchSq, D], pto.float32],
    ):
        scale = 1.0 / math.sqrt(TILE)
        sq_tiles = (Sq + (TILE - 1)) // TILE
        skv_tiles = (Skv + (TILE - 1)) // TILE
        num_cores = pto.get_block_num()

        # =================== CUBE SECTION ===================
        with pto.section_cube():
            q_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
            k_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                                  addr=TILE * TILE * 2)
            p_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                                  addr=TILE * TILE * 4)
            v_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                                  addr=TILE * TILE * 6)
            q_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
            k_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT,
                                    addr=0)
            qk_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)
            p_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT,
                                   addr=TILE * TILE * 2)
            v_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT,
                                    addr=TILE * TILE * 2)
            pv_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC,
                                   addr=TILE * TILE * 4)

            core_id = pto.get_block_idx()
            scratch_row = core_id * TILE  # per-core row in scratch buffers

            # Strided outer loop: each core handles tiles [core_id, core_id+num_cores, ...]
            for i in pto.range(core_id, sq_tiles, num_cores):
                sq_off = i * TILE

                # Reload Q for this Sq tile
                pto.tload(q_mat, q, offsets=[sq_off, 0])
                pto.tmov(q_left, q_mat)

                # ---- First KV tile (j=0) ----
                pto.tload(k_mat, kt, offsets=[0, 0])
                pto.tmov(k_right, k_mat)
                pto.tmatmul(qk_acc, q_left, k_right)
                pto.tstore(qk_buf, qk_acc, offsets=[scratch_row, 0])
                pto.sync_set(pto.PIPE_FIX, QK_READY)

                pto.sync_wait(pto.PIPE_M, P_READY)

                # PV_0 = P_0 @ V_0
                pto.tload(p_mat, p_buf, offsets=[scratch_row, 0])
                pto.tmov(p_left, p_mat)
                pto.tload(v_mat, v, offsets=[0, 0])
                pto.tmov(v_right, v_mat)
                pto.tmatmul(pv_acc, p_left, v_right)
                pto.tstore(pv_buf, pv_acc, offsets=[scratch_row, 0])
                pto.sync_set(pto.PIPE_FIX, PV_READY)

                # ---- Remaining KV tiles ----
                for j in pto.range(1, skv_tiles):
                    skv_off = j * TILE
                    pto.tload(k_mat, kt, offsets=[0, skv_off])
                    pto.tmov(k_right, k_mat)
                    pto.tmatmul(qk_acc, q_left, k_right)
                    pto.tstore(qk_buf, qk_acc,
                               offsets=[scratch_row, skv_off])
                    pto.sync_set(pto.PIPE_FIX, QK_READY)

                    pto.sync_wait(pto.PIPE_M, P_READY)

                    pto.tload(p_mat, p_buf,
                              offsets=[scratch_row, skv_off])
                    pto.tmov(p_left, p_mat)
                    pto.tload(v_mat, v, offsets=[skv_off, 0])
                    pto.tmov(v_right, v_mat)
                    pto.tmatmul(pv_acc, p_left, v_right)
                    pto.tstore(pv_buf, pv_acc, offsets=[scratch_row, 0])
                    pto.sync_set(pto.PIPE_FIX, PV_READY)

        # =================== VECTOR SECTION ===================
        HALF = TILE // 2
        with pto.section_vector():
            qk_vec = pto.make_tile((HALF, TILE), pto.float32, pto.VEC, addr=0)
            tmp_vec = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                    addr=HALF * TILE * 4)
            p_f16 = pto.make_tile((HALF, TILE), pto.float16, pto.VEC,
                                   addr=HALF * TILE * 8)
            reduce_dst = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                        addr=HALF * TILE * 10,
                                        valid_shape=(HALF, 1))
            global_max = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                        addr=HALF * TILE * 14)
            global_sum = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                        addr=HALF * TILE * 18)
            running_o = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                       addr=HALF * TILE * 22)
            exp_corr = pto.make_tile((HALF, TILE), pto.float32, pto.VEC,
                                      addr=HALF * TILE * 26)

            core_id = pto.get_block_idx()
            scratch_row = core_id * TILE
            sub_id = pto.get_subblock_idx()
            row_off = sub_id * HALF

            for i in pto.range(core_id, sq_tiles, num_cores):
                sq_off = i * TILE

                # ---- First KV tile: FlashSoftmax INIT + GU INIT ----
                pto.sync_wait(pto.PIPE_V, QK_READY)
                pto.tload(qk_vec, qk_buf,
                          offsets=[scratch_row + row_off, 0])

                pto.trowmax(reduce_dst, qk_vec, tmp_vec)
                pto.trowexpand(global_max, reduce_dst)
                pto.tsub(tmp_vec, qk_vec, global_max)
                pto.tmuls(tmp_vec, tmp_vec, scale)
                pto.texp(qk_vec, tmp_vec)
                pto.trowsum(reduce_dst, qk_vec, tmp_vec)
                pto.trowexpand(global_sum, reduce_dst)
                pto.tcvt(p_f16, qk_vec)

                pto.tstore(p_buf, p_f16,
                           offsets=[scratch_row + row_off, 0])
                pto.sync_set(pto.PIPE_MTE3, P_READY)

                pto.sync_wait(pto.PIPE_V, PV_READY)
                pto.tload(running_o, pv_buf,
                          offsets=[scratch_row + row_off, 0])

                # ---- Remaining KV tiles: FlashSoftmax UPDATE + GU ----
                for j in pto.range(1, skv_tiles):
                    skv_off = j * TILE
                    pto.sync_wait(pto.PIPE_V, QK_READY)
                    pto.tload(qk_vec, qk_buf,
                              offsets=[scratch_row + row_off, skv_off])

                    pto.trowmax(reduce_dst, qk_vec, tmp_vec)
                    pto.trowexpand(tmp_vec, reduce_dst)
                    pto.tmax(tmp_vec, tmp_vec, global_max)
                    pto.tsub(exp_corr, global_max, tmp_vec)
                    pto.tmuls(exp_corr, exp_corr, scale)
                    pto.texp(exp_corr, exp_corr)
                    pto.tmul(global_sum, global_sum, exp_corr)
                    pto.tmul(running_o, running_o, exp_corr)
                    pto.tmuls(global_max, tmp_vec, 1.0)
                    pto.tsub(tmp_vec, qk_vec, global_max)
                    pto.tmuls(tmp_vec, tmp_vec, scale)
                    pto.texp(qk_vec, tmp_vec)
                    pto.trowsum(reduce_dst, qk_vec, tmp_vec)
                    pto.trowexpand(tmp_vec, reduce_dst)
                    pto.tadd(global_sum, global_sum, tmp_vec)
                    pto.tcvt(p_f16, qk_vec)
                    pto.tstore(p_buf, p_f16,
                               offsets=[scratch_row + row_off, skv_off])
                    pto.sync_set(pto.PIPE_MTE3, P_READY)

                    # GlobalUpdate: O = O * exp_correction + PV_j
                    pto.sync_wait(pto.PIPE_V, PV_READY)
                    pto.tload(qk_vec, pv_buf,
                              offsets=[scratch_row + row_off, 0])
                    pto.tadd(running_o, running_o, qk_vec)

                # ---- Final: O = running_o / global_sum ----
                pto.tdiv(running_o, running_o, global_sum)
                pto.tcvt(p_f16, running_o)
                pto.tstore(o, p_f16, offsets=[sq_off + row_off, 0])

    return multicore_fa_kernel


def _make_fa_double_buffer_kernel():
    """Create double-buffered streaming FlashAttention kernel.

    Follows the ``matmul_kernel_double_buffer_no_sync`` pattern:
    - All tile buffers that participate in the KV loop are duplicated
      (buf0 / buf1) and wrapped in ``TileGroup``.
    - Buffer index alternates via ``j % 2`` each KV iteration.
    - No manual sync — ``auto_sync=True`` inserts all ``set_flag`` /
      ``wait_flag`` and uses ``EventIdGroup`` internally.

    K is passed as K[Skv, D] (not pre-transposed).  The kernel uses
    ``tload(..., layout="DN")`` with a DN tensor view and ZN MAT tiles so the
    hardware performs the transpose during the DN → ZN DMA conversion.

    Double-buffered resources:
      Cube:  k_mat(ZN), k_right, p_mat, p_left, v_mat, v_right, qk_acc, pv_acc
      Vector: qk_vec, p_f16  (DMA overlap with PIPE_V compute)

    Single-buffered (Q loaded once, running state):
      Cube:  q_mat, q_left
      Vector: tmp_vec, reduce_dst, global_max, global_sum, running_o, exp_corr
    """
    Sq = pto.DynVar("Sq")
    Skv = pto.DynVar("Skv")
    D = pto.DynVar("D")

    @pto.kernel
    def fa_double_buffer_kernel(
        q: pto.Tensor[[Sq, D], pto.float16],
        k: pto.Tensor[[Skv, D], pto.float16],       # K (not transposed)
        v: pto.Tensor[[Skv, D], pto.float16],
        o: pto.Tensor[[Sq, D], pto.float16],
        qk_buf: pto.Tensor[[Sq, Skv], pto.float32],
        p_buf: pto.Tensor[[Sq, Skv], pto.float16],
        pv_buf: pto.Tensor[[Sq, D], pto.float32],
    ):
        scale = 1.0 / math.sqrt(TILE)
        skv_tiles = (Skv + (TILE - 1)) // TILE

        # =================== CUBE SECTION ===================
        with pto.section_cube():
            # Tile type descriptors
            mat_f16 = pto.TileType((TILE, TILE), pto.float16, pto.MAT)
            mat_f16_zn = pto.TileType((TILE, TILE), pto.float16, pto.MAT,
                                      blayout=pto.BLayout.RowMajor,
                                      slayout=pto.SLayout.ColMajor)
            left_f16 = pto.TileType((TILE, TILE), pto.float16, pto.LEFT)
            right_f16 = pto.TileType((TILE, TILE), pto.float16, pto.RIGHT)
            acc_f32 = pto.TileType((TILE, TILE), pto.float32, pto.ACC)

            # ---- MAT (L1) buffers ----
            q_mat = pto.make_tile(mat_f16, addr=0)
            # K uses ZN layout for DN→ZN transposed load
            k_mat0 = pto.make_tile(mat_f16_zn, addr=TILE * TILE * 2)
            k_mat1 = pto.make_tile(mat_f16_zn, addr=TILE * TILE * 4)
            k_mat_group = pto.TileGroup([k_mat0, k_mat1])
            p_mat0 = pto.make_tile(mat_f16, addr=TILE * TILE * 6)
            p_mat1 = pto.make_tile(mat_f16, addr=TILE * TILE * 8)
            p_mat_group = pto.TileGroup([p_mat0, p_mat1])
            v_mat0 = pto.make_tile(mat_f16, addr=TILE * TILE * 10)
            v_mat1 = pto.make_tile(mat_f16, addr=TILE * TILE * 12)
            v_mat_group = pto.TileGroup([v_mat0, v_mat1])

            # ---- LEFT (L0A) buffers ----
            q_left = pto.make_tile(left_f16, addr=0)
            p_left0 = pto.make_tile(left_f16, addr=TILE * TILE * 2)
            p_left1 = pto.make_tile(left_f16, addr=TILE * TILE * 4)
            p_left_group = pto.TileGroup([p_left0, p_left1])

            # ---- RIGHT (L0B) buffers ----
            k_right0 = pto.make_tile(right_f16, addr=0)
            k_right1 = pto.make_tile(right_f16, addr=TILE * TILE * 2)
            k_right_group = pto.TileGroup([k_right0, k_right1])
            v_right0 = pto.make_tile(right_f16, addr=TILE * TILE * 4)
            v_right1 = pto.make_tile(right_f16, addr=TILE * TILE * 6)
            v_right_group = pto.TileGroup([v_right0, v_right1])

            # ---- ACC (L0C) buffers ----
            qk_acc0 = pto.make_tile(acc_f32, addr=0)
            qk_acc1 = pto.make_tile(acc_f32, addr=TILE * TILE * 4)
            qk_acc_group = pto.TileGroup([qk_acc0, qk_acc1])
            pv_acc0 = pto.make_tile(acc_f32, addr=TILE * TILE * 8)
            pv_acc1 = pto.make_tile(acc_f32, addr=TILE * TILE * 12)
            pv_acc_group = pto.TileGroup([pv_acc0, pv_acc1])

            core_id = pto.get_block_idx()
            sq_off = core_id * TILE

            # Load Q once (single buffer)
            pto.tload(q_mat, q, offsets=[sq_off, 0])
            pto.tmov(q_left, q_mat)

            # ---- KV loop: unified with double buffering ----
            for j in pto.range(skv_tiles):
                buf = j % 2
                skv_off = j * TILE

                # QK: S_j = Q @ K_j^T  (transposed load: K[Skv,D] → K^T[D,Skv])
                pto.tload(k_mat_group[buf], k,
                                     offsets=[skv_off, 0], layout="DN")
                pto.tmov(k_right_group[buf], k_mat_group[buf])
                pto.tmatmul(qk_acc_group[buf], q_left, k_right_group[buf])
                pto.tstore(qk_buf, qk_acc_group[buf],
                           offsets=[sq_off, skv_off])
                pto.sync_set(pto.PIPE_FIX, QK_READY)

                pto.sync_wait(pto.PIPE_M, P_READY)

                # PV: PV_j = P_j @ V_j
                pto.tload(p_mat_group[buf], p_buf,
                          offsets=[sq_off, skv_off])
                pto.tmov(p_left_group[buf], p_mat_group[buf])
                pto.tload(v_mat_group[buf], v, offsets=[skv_off, 0])
                pto.tmov(v_right_group[buf], v_mat_group[buf])
                pto.tmatmul(pv_acc_group[buf], p_left_group[buf],
                            v_right_group[buf])
                pto.tstore(pv_buf, pv_acc_group[buf],
                           offsets=[sq_off, 0])
                pto.sync_set(pto.PIPE_FIX, PV_READY)

        # =================== VECTOR SECTION ===================
        HALF = TILE // 2
        with pto.section_vector():
            # Tile type descriptors
            vec_f32 = pto.TileType((HALF, TILE), pto.float32, pto.VEC)
            vec_f16 = pto.TileType((HALF, TILE), pto.float16, pto.VEC)
            vec_reduce = pto.TileType((HALF, TILE), pto.float32, pto.VEC,
                                      valid_shape=(HALF, 1))

            # Double-buffered VEC tiles for DMA/compute overlap
            qk_vec0 = pto.make_tile(vec_f32, addr=0)
            qk_vec1 = pto.make_tile(vec_f32, addr=HALF * TILE * 4)
            qk_vec_group = pto.TileGroup([qk_vec0, qk_vec1])

            tmp_vec = pto.make_tile(vec_f32, addr=HALF * TILE * 8)

            p_f16_0 = pto.make_tile(vec_f16, addr=HALF * TILE * 12)
            p_f16_1 = pto.make_tile(vec_f16, addr=HALF * TILE * 14)
            p_f16_group = pto.TileGroup([p_f16_0, p_f16_1])

            reduce_dst = pto.make_tile(vec_reduce, addr=HALF * TILE * 16)
            global_max = pto.make_tile(vec_f32, addr=HALF * TILE * 20)
            global_sum = pto.make_tile(vec_f32, addr=HALF * TILE * 24)
            running_o = pto.make_tile(vec_f32, addr=HALF * TILE * 28)
            exp_corr = pto.make_tile(vec_f32, addr=HALF * TILE * 32)

            core_id = pto.get_block_idx()
            sq_off = core_id * TILE
            sub_id = pto.get_subblock_idx()
            row_off = sub_id * HALF

            # ---- First KV tile (j=0): FlashSoftmax INIT ----
            pto.sync_wait(pto.PIPE_V, QK_READY)
            pto.tload(qk_vec_group[0], qk_buf,
                      offsets=[sq_off + row_off, 0])

            pto.trowmax(reduce_dst, qk_vec_group[0], tmp_vec)
            pto.trowexpand(global_max, reduce_dst)
            pto.tsub(tmp_vec, qk_vec_group[0], global_max)
            pto.tmuls(tmp_vec, tmp_vec, scale)
            pto.texp(qk_vec_group[0], tmp_vec)
            pto.trowsum(reduce_dst, qk_vec_group[0], tmp_vec)
            pto.trowexpand(global_sum, reduce_dst)
            pto.tcvt(p_f16_group[0], qk_vec_group[0])

            pto.tstore(p_buf, p_f16_group[0],
                       offsets=[sq_off + row_off, 0])
            pto.sync_set(pto.PIPE_MTE3, P_READY)

            pto.sync_wait(pto.PIPE_V, PV_READY)
            pto.tload(running_o, pv_buf,
                      offsets=[sq_off + row_off, 0])

            # ---- Remaining KV tiles: FlashSoftmax UPDATE + GU ----
            for j in pto.range(1, skv_tiles):
                buf = j % 2
                skv_off = j * TILE

                pto.sync_wait(pto.PIPE_V, QK_READY)
                pto.tload(qk_vec_group[buf], qk_buf,
                          offsets=[sq_off + row_off, skv_off])

                # --- FlashSoftmax UPDATE ---
                pto.trowmax(reduce_dst, qk_vec_group[buf], tmp_vec)
                pto.trowexpand(tmp_vec, reduce_dst)
                pto.tmax(tmp_vec, tmp_vec, global_max)
                pto.tsub(exp_corr, global_max, tmp_vec)
                pto.tmuls(exp_corr, exp_corr, scale)
                pto.texp(exp_corr, exp_corr)
                pto.tmul(global_sum, global_sum, exp_corr)
                pto.tmul(running_o, running_o, exp_corr)
                pto.tmuls(global_max, tmp_vec, 1.0)
                pto.tsub(tmp_vec, qk_vec_group[buf], global_max)
                pto.tmuls(tmp_vec, tmp_vec, scale)
                pto.texp(qk_vec_group[buf], tmp_vec)
                pto.trowsum(reduce_dst, qk_vec_group[buf], tmp_vec)
                pto.trowexpand(tmp_vec, reduce_dst)
                pto.tadd(global_sum, global_sum, tmp_vec)
                pto.tcvt(p_f16_group[buf], qk_vec_group[buf])
                pto.tstore(p_buf, p_f16_group[buf],
                           offsets=[sq_off + row_off, skv_off])
                pto.sync_set(pto.PIPE_MTE3, P_READY)

                # --- GlobalUpdate: O = O * exp_correction + PV_j ---
                pto.sync_wait(pto.PIPE_V, PV_READY)
                pto.tload(qk_vec_group[buf], pv_buf,
                          offsets=[sq_off + row_off, 0])
                pto.tadd(running_o, running_o, qk_vec_group[buf])

            # ---- Final normalization: O = running_o / global_sum ----
            pto.tdiv(running_o, running_o, global_sum)
            pto.tcvt(p_f16_group[0], running_o)
            pto.tstore(o, p_f16_group[0],
                       offsets=[sq_off + row_off, 0])

    return fa_double_buffer_kernel


# ===================================================================
#  Test harness
# ===================================================================

def test_ir_qk():
    ir = _make_qk_kernel().emit_ir()
    print(ir)
    assert "pto.tmatmul" in ir
    print("// Step 1 IR: QK matmul OK.", file=sys.stderr)


def test_ir_full():
    ir = _make_fa_kernel().emit_ir()
    print(ir)
    assert "pto.section.cube" in ir
    assert "pto.section.vector" in ir
    assert "pto.sync.set" in ir
    print("// Full FA IR: cross-core kernel OK.", file=sys.stderr)


def test_ir_k_transpose():
    ir = _make_k_autotranspose_kernel().emit_ir()
    print(ir)
    assert "pto.tmatmul" in ir
    print("// K auto-transpose IR: OK.", file=sys.stderr)


def test_ir_multicore():
    ir = _make_multicore_fa_kernel().emit_ir()
    print(ir)
    assert "pto.section.cube" in ir
    assert "pto.section.vector" in ir
    assert "pto.sync.set" in ir
    print("// Multi-core FA IR: OK.", file=sys.stderr)


def test_ir_double_buffer():
    ir = _make_fa_double_buffer_kernel().emit_ir()
    print(ir)
    assert "pto.section.cube" in ir
    assert "pto.section.vector" in ir
    assert "pto.sync.set" in ir
    print("// Double-buffer FA IR: OK.", file=sys.stderr)


def test_npu_qk():
    import torch
    import torch_npu

    kernel = _make_qk_kernel()

    @pto.jit
    def run():
        compiled = pto.compile(kernel, arch="a3", auto_sync=True)
        print(f"  compiled: {compiled.lib_path}", file=sys.stderr)

        device = "npu:6"
        torch.npu.set_device(device)

        for (sq, skv, d) in [(64,64,64), (128,128,64),
                              (128,128,128), (256,256,64)]:
            torch.manual_seed(42)
            q = torch.rand((sq, d), device=device, dtype=torch.float16)
            k = torch.rand((skv, d), device=device, dtype=torch.float16)
            kt = k.T.contiguous()
            s = torch.empty((sq, skv), device=device, dtype=torch.float16)

            pto.launch(compiled, q, kt, s)
            torch.npu.synchronize()

            s_ref = torch.matmul(q, k.T)
            diff = (s - s_ref).abs().max().item()
            print(f"  ({sq},{skv},{d}): max|diff|={diff:.4f}", file=sys.stderr)
            torch.testing.assert_close(s, s_ref, rtol=5e-3, atol=5e-3)
            print(f"  ({sq},{skv},{d}): PASS", file=sys.stderr)

    run()
    print("// Step 1 PASS: QK matmul verified.", file=sys.stderr)


def test_npu_full():
    import torch
    import torch_npu

    kernel = _make_fa_kernel()

    @pto.jit
    def run():
        compiled = pto.compile(kernel, arch="a3", auto_sync=True)
        print(f"  compiled: {compiled.lib_path}", file=sys.stderr)

        device = "npu:6"
        torch.npu.set_device(device)

        # Note: Skv must be >= 2*TILE for streaming path. Single-tile (Skv=TILE)
        # has a known auto_sync priming/drain conflict with cross-core sync.
        for (sq, skv, d) in [(64,128,64), (64,256,64), (64,512,64)]:
            torch.manual_seed(42)
            q = torch.rand((sq, d), device=device, dtype=torch.float16)
            k = torch.rand((skv, d), device=device, dtype=torch.float16)
            kt = k.T.contiguous()
            v = torch.rand((skv, d), device=device, dtype=torch.float16)
            o = torch.empty((sq, d), device=device, dtype=torch.float16)

            qk_buf = torch.empty((sq, skv), device=device,
                                  dtype=torch.float32)
            p_buf = torch.empty((sq, skv), device=device,
                                 dtype=torch.float16)
            pv_buf = torch.empty((sq, d), device=device, dtype=torch.float32)

            pto.launch(compiled, q, kt, v, o, qk_buf, p_buf, pv_buf,
                       block_dim=1)
            torch.npu.synchronize()

            scale_val = 1.0 / math.sqrt(d)
            qk_ref = torch.matmul(q.float(), k.float().T) * scale_val
            attn_ref = torch.softmax(qk_ref, dim=-1)
            o_ref = torch.matmul(attn_ref, v.float()).half()

            diff = (o - o_ref).abs().max().item()
            print(f"  FA ({sq},{skv},{d}): max|diff|={diff:.4f}",
                  file=sys.stderr)
            print(f"    o[0,:8]:     {o[0,:8].tolist()}", file=sys.stderr)
            print(f"    o_ref[0,:8]: {o_ref[0,:8].tolist()}", file=sys.stderr)

            torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
            print(f"  FA ({sq},{skv},{d}): PASS", file=sys.stderr)

    run()
    print("// Step 3 PASS: Full FlashAttention verified.", file=sys.stderr)


def test_npu_k_transpose():
    """Test K auto-transpose: load K[Skv,D] directly (no host transpose).

    Uses ZN MAT layout with DN tensor view so hardware does the transpose.

    CURRENT STATUS: IR generation works correctly. Compilation fails because
    the ptoas EmitC lowering always generates Layout::ND for GlobalTensors.
    Supporting DN layout requires adding stride-pattern detection to the
    lowering pass (lib/PTO/Transforms) to emit Layout::DN when strides
    indicate column-major ordering.

    For now, this test verifies IR generation only.
    """
    test_ir_k_transpose()
    print("// K auto-transpose: IR verified. "
          "Compile blocked on EmitC DN layout support.",
          file=sys.stderr)


def test_npu_multicore_fa():
    """Test multi-core FlashAttention with Sq=8192, Skv=2048, D=64.

    Distributes Sq tiles across 24 Cube cores using strided loop.
    """
    import torch
    import torch_npu

    kernel = _make_multicore_fa_kernel()

    @pto.jit
    def run():
        compiled = pto.compile(kernel, arch="a3", auto_sync=True)
        print(f"  compiled: {compiled.lib_path}", file=sys.stderr)

        device = "npu:6"
        torch.npu.set_device(device)

        block_dim = 24
        for (sq, skv, d) in [(8192, 2048, 64)]:
            torch.manual_seed(42)
            q = torch.rand((sq, d), device=device, dtype=torch.float16)
            k = torch.rand((skv, d), device=device, dtype=torch.float16)
            kt = k.T.contiguous()
            v = torch.rand((skv, d), device=device, dtype=torch.float16)
            o = torch.empty((sq, d), device=device, dtype=torch.float16)

            scratch_sq = block_dim * TILE
            qk_buf = torch.empty((scratch_sq, skv), device=device,
                                  dtype=torch.float32)
            p_buf = torch.empty((scratch_sq, skv), device=device,
                                 dtype=torch.float16)
            pv_buf = torch.empty((scratch_sq, d), device=device,
                                  dtype=torch.float32)

            pto.launch(compiled, q, kt, v, o, qk_buf, p_buf, pv_buf,
                       block_dim=block_dim)
            torch.npu.synchronize()

            # Reference: standard attention
            scale_val = 1.0 / math.sqrt(d)
            qk_ref = torch.matmul(q.float(), k.float().T) * scale_val
            attn_ref = torch.softmax(qk_ref, dim=-1)
            o_ref = torch.matmul(attn_ref, v.float()).half()

            diff = (o - o_ref).abs().max().item()
            print(f"  Multi-core FA ({sq},{skv},{d}): max|diff|={diff:.4f}",
                  file=sys.stderr)
            print(f"    o[0,:8]:     {o[0,:8].tolist()}", file=sys.stderr)
            print(f"    o_ref[0,:8]: {o_ref[0,:8].tolist()}", file=sys.stderr)

            torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
            print(f"  Multi-core FA ({sq},{skv},{d}): PASS", file=sys.stderr)

    run()
    print("// Multi-core FA PASS: verified.", file=sys.stderr)


def test_npu_double_buffer_fa():
    """Test double-buffered FlashAttention with TileGroup and auto_sync.

    Same algorithm as test_npu_full but with all KV-loop tile buffers
    double-buffered (TileGroup). auto_sync generates EventIdGroup-based
    synchronization automatically.
    """
    import torch
    import torch_npu

    kernel = _make_fa_double_buffer_kernel()

    @pto.jit
    def run():
        compiled = pto.compile(kernel, arch="a3", auto_sync=True)
        print(f"  compiled: {compiled.lib_path}", file=sys.stderr)

        device = "npu:6"
        torch.npu.set_device(device)

        for (sq, skv, d) in [(64, 128, 64), (64, 256, 64), (64, 512, 64)]:
            torch.manual_seed(42)
            q = torch.rand((sq, d), device=device, dtype=torch.float16)
            k = torch.rand((skv, d), device=device, dtype=torch.float16)
            v = torch.rand((skv, d), device=device, dtype=torch.float16)
            o = torch.empty((sq, d), device=device, dtype=torch.float16)

            qk_buf = torch.empty((sq, skv), device=device,
                                  dtype=torch.float32)
            p_buf = torch.empty((sq, skv), device=device,
                                 dtype=torch.float16)
            pv_buf = torch.empty((sq, d), device=device, dtype=torch.float32)

            pto.launch(compiled, q, k, v, o, qk_buf, p_buf, pv_buf,
                       block_dim=1)
            torch.npu.synchronize()

            scale_val = 1.0 / math.sqrt(d)
            qk_ref = torch.matmul(q.float(), k.float().T) * scale_val
            attn_ref = torch.softmax(qk_ref, dim=-1)
            o_ref = torch.matmul(attn_ref, v.float()).half()

            diff = (o - o_ref).abs().max().item()
            print(f"  DB-FA ({sq},{skv},{d}): max|diff|={diff:.4f}",
                  file=sys.stderr)
            print(f"    o[0,:8]:     {o[0,:8].tolist()}", file=sys.stderr)
            print(f"    o_ref[0,:8]: {o_ref[0,:8].tolist()}", file=sys.stderr)

            torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
            print(f"  DB-FA ({sq},{skv},{d}): PASS", file=sys.stderr)

    run()
    print("// Double-buffer FA PASS: verified.", file=sys.stderr)


if __name__ == "__main__":
    if "--ir-only" in sys.argv:
        test_ir_qk()
        print("---", file=sys.stderr)
        test_ir_full()
    elif "--step1" in sys.argv:
        try:
            test_npu_qk()
        except (ImportError, RuntimeError) as e:
            print(f"NPU not available ({e}), IR test.", file=sys.stderr)
            test_ir_qk()
    elif "--full" in sys.argv:
        try:
            test_npu_full()
        except (ImportError, RuntimeError) as e:
            print(f"NPU not available ({e}), IR test.", file=sys.stderr)
            test_ir_full()
    elif "--ktranspose" in sys.argv:
        try:
            test_npu_k_transpose()
        except (ImportError, RuntimeError) as e:
            print(f"NPU not available ({e}), IR test.", file=sys.stderr)
            test_ir_k_transpose()
    elif "--multicore" in sys.argv:
        try:
            test_npu_multicore_fa()
        except (ImportError, RuntimeError) as e:
            print(f"NPU not available ({e}), IR test.", file=sys.stderr)
            test_ir_multicore()
    elif "--double-buffer" in sys.argv:
        try:
            test_npu_double_buffer_fa()
        except (ImportError, RuntimeError) as e:
            print(f"NPU not available ({e}), IR test.", file=sys.stderr)
            test_ir_double_buffer()
    else:
        try:
            test_npu_qk()
            print("---", file=sys.stderr)
            test_npu_full()
        except (ImportError, RuntimeError) as e:
            print(f"NPU not available ({e}), IR tests.", file=sys.stderr)
            test_ir_qk()
            print("---", file=sys.stderr)
            test_ir_full()

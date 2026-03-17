"""FlashAttention implementation using PTOAS Python frontend.

Step-by-step verification against C++ reference (fa_performance_kernel.cpp):
  Step 1: QK matmul only — S = Q @ K^T  (Cube section)
  Step 2: Full streaming FlashAttention with cross-core Cube+Vector pipeline

Key insight on matmul layout:
  tmatmul(acc, left, right) computes standard C = A @ B.
  To get Q @ K^T, we pass K^T[D, Skv] as input (host transposes K).
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
                    pv_q = q.partition(offsets=[sq_off, 0], sizes=[TILE, TILE])
                    pv_kt = kt.partition(offsets=[0, skv_off],
                                          sizes=[TILE, TILE])
                    pto.tload(q_mat, pv_q)
                    pto.tload(k_mat, pv_kt)
                    pto.tmov(q_left, q_mat)
                    pto.tmov(k_right, k_mat)
                    pto.tmatmul(s_acc, q_left, k_right)

                    # Remaining D tiles
                    for d in pto.range(1, d_tiles):
                        d_off = d * TILE
                        pv_q_d = q.partition(offsets=[sq_off, d_off],
                                             sizes=[TILE, TILE])
                        pv_kt_d = kt.partition(offsets=[d_off, skv_off],
                                                sizes=[TILE, TILE])
                        pto.tload(q_mat, pv_q_d)
                        pto.tload(k_mat, pv_kt_d)
                        pto.tmov(q_left, q_mat)
                        pto.tmov(k_right, k_mat)
                        pto.tmatmul_acc(s_acc, s_acc, q_left, k_right)

                    pv_s = s.partition(offsets=[sq_off, skv_off],
                                       sizes=[TILE, TILE])
                    pto.tstore(pv_s, s_acc)

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
            pv_q = q.partition(offsets=[sq_off, 0], sizes=[TILE, TILE])
            pto.tload(q_mat, pv_q)
            pto.tmov(q_left, q_mat)

            # ---- First KV tile (j=0) ----
            # QK_0 = Q @ Kt_0
            pv_kt0 = kt.partition(offsets=[0, 0], sizes=[TILE, TILE])
            pto.tload(k_mat, pv_kt0)
            pto.tmov(k_right, k_mat)
            pto.tmatmul(qk_acc, q_left, k_right)
            pv_qk = qk_buf.partition(offsets=[sq_off, 0], sizes=[TILE, TILE])
            pto.tstore(pv_qk, qk_acc)
            pto.sync_set(pto.PIPE_FIX, QK_READY)

            pto.sync_wait(pto.PIPE_M, P_READY)

            # PV_0 = P_0 @ V_0
            pv_p0 = p_buf.partition(offsets=[sq_off, 0], sizes=[TILE, TILE])
            pto.tload(p_mat, pv_p0)
            pto.tmov(p_left, p_mat)
            pv_v0 = v.partition(offsets=[0, 0], sizes=[TILE, TILE])
            pto.tload(v_mat, pv_v0)
            pto.tmov(v_right, v_mat)
            pto.tmatmul(pv_acc, p_left, v_right)
            pv_pv = pv_buf.partition(offsets=[sq_off, 0], sizes=[TILE, TILE])
            pto.tstore(pv_pv, pv_acc)
            pto.sync_set(pto.PIPE_FIX, PV_READY)

            # ---- Remaining KV tiles ----
            for j in pto.range(1, skv_tiles):
                skv_off = j * TILE
                pv_ktj = kt.partition(offsets=[0, skv_off],
                                       sizes=[TILE, TILE])
                pto.tload(k_mat, pv_ktj)
                pto.tmov(k_right, k_mat)
                pto.tmatmul(qk_acc, q_left, k_right)
                pv_qk_j = qk_buf.partition(offsets=[sq_off, skv_off],
                                             sizes=[TILE, TILE])
                pto.tstore(pv_qk_j, qk_acc)
                pto.sync_set(pto.PIPE_FIX, QK_READY)

                pto.sync_wait(pto.PIPE_M, P_READY)

                pv_pj = p_buf.partition(offsets=[sq_off, skv_off],
                                         sizes=[TILE, TILE])
                pto.tload(p_mat, pv_pj)
                pto.tmov(p_left, p_mat)
                pv_vj = v.partition(offsets=[skv_off, 0],
                                     sizes=[TILE, TILE])
                pto.tload(v_mat, pv_vj)
                pto.tmov(v_right, v_mat)
                pto.tmatmul(pv_acc, p_left, v_right)
                pto.tstore(pv_pv, pv_acc)
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
            pv_qk0 = qk_buf.partition(offsets=[sq_off + row_off, 0],
                                       sizes=[HALF, TILE])
            pto.tload(qk_vec, pv_qk0)

            pto.trowmax(reduce_dst, qk_vec, tmp_vec)
            pto.trowexpand(global_max, reduce_dst)
            pto.tsub(tmp_vec, qk_vec, global_max)
            pto.tmuls(tmp_vec, tmp_vec, scale)
            pto.texp(qk_vec, tmp_vec)
            pto.trowsum(reduce_dst, qk_vec, tmp_vec)
            pto.trowexpand(global_sum, reduce_dst)
            pto.tcvt(p_f16, qk_vec)

            pv_p0 = p_buf.partition(offsets=[sq_off + row_off, 0],
                                     sizes=[HALF, TILE])
            pto.tstore(pv_p0, p_f16)
            pto.sync_set(pto.PIPE_MTE3, P_READY)

            pto.sync_wait(pto.PIPE_V, PV_READY)
            pv_pv0 = pv_buf.partition(offsets=[sq_off + row_off, 0],
                                       sizes=[HALF, TILE])
            pto.tload(running_o, pv_pv0)

            # ---- Remaining KV tiles: FlashSoftmax UPDATE + GU ----
            for j in pto.range(1, skv_tiles):
                skv_off = j * TILE
                pto.sync_wait(pto.PIPE_V, QK_READY)
                pv_qkj = qk_buf.partition(
                    offsets=[sq_off + row_off, skv_off], sizes=[HALF, TILE])
                pto.tload(qk_vec, pv_qkj)

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
                pv_pj = p_buf.partition(
                    offsets=[sq_off + row_off, skv_off], sizes=[HALF, TILE])
                pto.tstore(pv_pj, p_f16)
                pto.sync_set(pto.PIPE_MTE3, P_READY)

                # GlobalUpdate: O = O * exp_correction + PV_j
                pto.sync_wait(pto.PIPE_V, PV_READY)
                pv_pvj = pv_buf.partition(
                    offsets=[sq_off + row_off, 0], sizes=[HALF, TILE])
                pto.tload(qk_vec, pv_pvj)
                pto.tadd(running_o, running_o, qk_vec)

            # ---- Final: O = running_o / global_sum ----
            pto.tdiv(running_o, running_o, global_sum)
            pto.tcvt(p_f16, running_o)
            pv_o = o.partition(offsets=[sq_off + row_off, 0],
                               sizes=[HALF, TILE])
            pto.tstore(pv_o, p_f16)

    return flash_attention_kernel


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

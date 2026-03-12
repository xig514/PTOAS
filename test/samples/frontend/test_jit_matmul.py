"""End-to-end test: dynamic-shape matrix multiplication on NPU.

Demonstrates:
  - @pto.kernel with DynVar dynamic shapes (M, N, K)
  - MAT → LEFT / RIGHT → ACC tile pipeline for matmul
  - Peeled first K iteration (tmatmul) + loop accumulation (tmatmul_acc)
  - Automatic pipeline synchronization via auto_sync=True
  - pto.compile() + pto.launch() for NPU execution
  - Verification against torch.matmul reference

Usage (on NPU machine):
    python3 test_jit_matmul.py

Usage (IR-only, no NPU):
    python3 test_jit_matmul.py --ir-only
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend"))

import pto_frontend as pto

# -- Dynamic shape variables ------------------------------------------------
M = pto.DynVar("M")
N = pto.DynVar("N")
K = pto.DynVar("K")

TILE = 64


# ---------------------------------------------------------------------------
#  Matmul kernel:  C[M, N] = A[M, K] @ B[K, N]
# ---------------------------------------------------------------------------

@pto.kernel
def matmul_kernel(
    a: pto.Tensor[[M, K], pto.float16],
    b: pto.Tensor[[K, N], pto.float16],
    c: pto.Tensor[[M, N], pto.float16],
):
    with pto.section_cube():
        # -- Allocate tile buffers in each address space -----------------------
        # MAT: DMA staging buffers (L1)
        a_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
        b_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 2)

        # LEFT / RIGHT: L0 compute-input buffers
        a_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
        b_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)

        # ACC: L0-C accumulator (float32 for precision)
        c_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)

        # -- Compute tile counts -----------------------------------------------
        m_tiles = (M + (TILE - 1)) // TILE
        n_tiles = (N + (TILE - 1)) // TILE
        k_tiles = (K + (TILE - 1)) // TILE
        pto.set_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID0)
        pto.set_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID0)   
        pto.set_flag(pto.PIPE_FIX, pto.PIPE_M, pto.EVENT_ID0)
        for i in pto.range(m_tiles):
            for j in pto.range(n_tiles):
                m_off = i * TILE
                n_off = j * TILE

                # ==============================================================
                #  First K tile (k=0): tmatmul — clears ACC, then multiplies
                # ==============================================================
                pv_a0 = a.partition(offsets=[m_off, 0], sizes=[TILE, TILE])
                pv_b0 = b.partition(offsets=[0, n_off], sizes=[TILE, TILE])
                pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID0)
                # GM → MAT (MTE2 pipe)
                pto.tload(a_mat, pv_a0)
                pto.tload(b_mat, pv_b0)

                pto.set_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, pto.EVENT_ID0)
                pto.wait_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, pto.EVENT_ID0)
                pto.wait_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID0)
                # MAT → LEFT / RIGHT (MTE1 pipe)
                pto.tmov(a_left, a_mat)
                pto.tmov(b_right, b_mat)
                pto.set_flag(pto.PIPE_MTE1, pto.PIPE_M, pto.EVENT_ID0)
                pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_M, pto.EVENT_ID0)
                pto.set_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID0)
                # CUBE matmul (clears ACC first)
                pto.wait_flag(pto.PIPE_FIX, pto.PIPE_M, pto.EVENT_ID0)
                pto.tmatmul(c_acc, a_left, b_right)
                pto.set_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID0)

                # ==============================================================
                #  Remaining K tiles (k=1..k_tiles-1): tmatmul_acc — accumulates
                # ==============================================================
                for k in pto.range(1, k_tiles):
                    k_off = k * TILE
                    pv_a_k = a.partition(offsets=[m_off, k_off],
                                         sizes=[TILE, TILE])
                    pv_b_k = b.partition(offsets=[k_off, n_off],
                                         sizes=[TILE, TILE])
                    pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID0)
                    # GM → MAT (MTE2)
                    pto.tload(a_mat, pv_a_k)
                    pto.tload(b_mat, pv_b_k)
                    pto.set_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, pto.EVENT_ID0)
                    pto.wait_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, pto.EVENT_ID0)
                    pto.wait_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID0)
                    # MAT → LEFT / RIGHT (MTE1)
                    pto.tmov(a_left, a_mat)
                    pto.tmov(b_right, b_mat)
                    pto.set_flag(pto.PIPE_MTE1, pto.PIPE_M, pto.EVENT_ID0)
                    pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_M, pto.EVENT_ID0)
                    pto.set_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID0)
                    # CUBE matmul-accumulate
                    pto.tmatmul_acc(c_acc, c_acc, a_left, b_right)
                    pto.set_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID0)

                # ==============================================================
                #  Store result: ACC → GM (MTE3 pipe)
                # ==============================================================
                pv_c = c.partition(offsets=[m_off, n_off], sizes=[TILE, TILE])
                pto.set_flag(pto.PIPE_M, pto.PIPE_FIX, pto.EVENT_ID0)
                pto.wait_flag(pto.PIPE_M, pto.PIPE_FIX, pto.EVENT_ID0)
                pto.tstore(pv_c, c_acc)
                pto.set_flag(pto.PIPE_FIX, pto.PIPE_M, pto.EVENT_ID0)
        pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID0)
        pto.wait_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID0)   
        pto.wait_flag(pto.PIPE_FIX, pto.PIPE_M, pto.EVENT_ID0)


@pto.kernel
def matmul_kernel_double_buffer(
    a: pto.Tensor[[M, K], pto.float16],
    b: pto.Tensor[[K, N], pto.float16],
    c: pto.Tensor[[M, N], pto.float16],
):
    with pto.section_cube():
        # -- Allocate tile buffers in each address space -----------------------
        # MAT: DMA staging buffers (L1)
        a_mat0 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
        a_mat1 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=TILE * TILE * 2)

        a_mat_group = pto.TileGroup([a_mat0, a_mat1])

        b_mat0 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 2 * 2)
        b_mat1 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 2 * 3)
        b_mat_group = pto.TileGroup([b_mat0, b_mat1])

        # LEFT / RIGHT: L0 compute-input buffers
        a_left0 = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
        a_left1 = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=TILE * TILE * 2)
        a_left_group = pto.TileGroup([a_left0, a_left1])

        b_right0 = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)
        b_right1 = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=TILE * TILE * 2)
        b_right_group = pto.TileGroup([b_right0, b_right1])

        # ACC: L0-C accumulator (float32 for precision)
        c_acc0 = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)
        c_acc1 = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=TILE * TILE * 4)
        c_acc_group = pto.TileGroup([c_acc0, c_acc1])

        # -- Compute tile counts -----------------------------------------------
        m_tiles = (M + (TILE - 1)) // TILE
        n_tiles = (N + (TILE - 1)) // TILE
        k_tiles = (K + (TILE - 1)) // TILE

        # Create EventIdGroup for dynamic EVENT_ID selection
        event_ids = pto.EventIdGroup([pto.EVENT_ID0, pto.EVENT_ID1])

        pto.set_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID0)
        pto.set_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID0)
        pto.set_flag(pto.PIPE_FIX, pto.PIPE_M, pto.EVENT_ID0)

        pto.set_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID1)
        pto.set_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID1)
        pto.set_flag(pto.PIPE_FIX, pto.PIPE_M, pto.EVENT_ID1)
        for i in pto.range(m_tiles):
            for j in pto.range(n_tiles):
                m_off = i * TILE
                n_off = j * TILE
                buff_idx = (i * n_tiles + j) % 2
                l0c_idx = (i * n_tiles + j) % 2
                # ==============================================================
                #  First K tile (k=0): tmatmul — clears ACC, then multiplies
                # ==============================================================
                pv_a0 = a.partition(offsets=[m_off, 0], sizes=[TILE, TILE])
                pv_b0 = b.partition(offsets=[0, n_off], sizes=[TILE, TILE])
                pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, event_ids[buff_idx])
                # GM → MAT (MTE2 pipe)

                pto.tload(a_mat_group[buff_idx], pv_a0)
                pto.tload(b_mat_group[buff_idx], pv_b0)

                pto.set_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, event_ids[buff_idx])
                pto.wait_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, event_ids[buff_idx])
                pto.wait_flag(pto.PIPE_M, pto.PIPE_MTE1, event_ids[buff_idx])
                # MAT → LEFT / RIGHT (MTE1 pipe)
                pto.tmov(a_left_group[buff_idx], a_mat_group[buff_idx])
                pto.tmov(b_right_group[buff_idx], b_mat_group[buff_idx])
                pto.set_flag(pto.PIPE_MTE1, pto.PIPE_M, event_ids[buff_idx])
                pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_M, event_ids[buff_idx])
                pto.set_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, event_ids[buff_idx])
                # CUBE matmul (clears ACC first)
                pto.wait_flag(pto.PIPE_FIX, pto.PIPE_M, event_ids[l0c_idx])
                pto.tmatmul(c_acc_group[l0c_idx], a_left_group[buff_idx], b_right_group[buff_idx])
                pto.set_flag(pto.PIPE_M, pto.PIPE_MTE1, event_ids[buff_idx])

                # ==============================================================
                #  Remaining K tiles (k=1..k_tiles-1): tmatmul_acc — accumulates
                # ==============================================================
                for k in pto.range(1, k_tiles):
                    buff_idx = (buff_idx + 1) % 2
                    k_off = k * TILE
                    pv_a_k = a.partition(offsets=[m_off, k_off],
                                         sizes=[TILE, TILE])
                    pv_b_k = b.partition(offsets=[k_off, n_off],
                                         sizes=[TILE, TILE])
                    pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, event_ids[buff_idx])
                    # GM → MAT (MTE2)
                    pto.tload(a_mat_group[buff_idx], pv_a_k)
                    pto.tload(b_mat_group[buff_idx], pv_b_k)
                    pto.set_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, event_ids[buff_idx])
                    pto.wait_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, event_ids[buff_idx])
                    pto.wait_flag(pto.PIPE_M, pto.PIPE_MTE1, event_ids[buff_idx])
                    # MAT → LEFT / RIGHT (MTE1)
                    pto.tmov(a_left_group[buff_idx], a_mat_group[buff_idx])
                    pto.tmov(b_right_group[buff_idx], b_mat_group[buff_idx])
                    pto.set_flag(pto.PIPE_MTE1, pto.PIPE_M, event_ids[buff_idx])
                    pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_M, event_ids[buff_idx])
                    pto.set_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, event_ids[buff_idx])
                    # CUBE matmul-accumulate
                    pto.tmatmul_acc(c_acc_group[l0c_idx], c_acc_group[l0c_idx], a_left_group[buff_idx], b_right_group[buff_idx])
                    pto.set_flag(pto.PIPE_M, pto.PIPE_MTE1, event_ids[buff_idx])
                # ==============================================================
                #  Store result: ACC → GM (MTE3 pipe)
                # ==============================================================
                pv_c = c.partition(offsets=[m_off, n_off], sizes=[TILE, TILE])
                pto.set_flag(pto.PIPE_M, pto.PIPE_FIX, event_ids[l0c_idx])
                pto.wait_flag(pto.PIPE_M, pto.PIPE_FIX, event_ids[l0c_idx])
                pto.tstore(pv_c, c_acc_group[l0c_idx])
                pto.set_flag(pto.PIPE_FIX, pto.PIPE_M, event_ids[l0c_idx])
        pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID0)
        pto.wait_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID0)
        pto.wait_flag(pto.PIPE_FIX, pto.PIPE_M, pto.EVENT_ID0)
        pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID1)
        pto.wait_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID1)
        pto.wait_flag(pto.PIPE_FIX, pto.PIPE_M, pto.EVENT_ID1)


@pto.kernel
def matmul_kernel_double_buffer_no_sync(
    a: pto.Tensor[[M, K], pto.float16],
    b: pto.Tensor[[K, N], pto.float16],
    c: pto.Tensor[[M, N], pto.float16],
):
    with pto.section_cube():
        # -- Allocate tile buffers in each address space -----------------------
        # MAT: DMA staging buffers (L1)
        a_mat0 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
        a_mat1 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=TILE * TILE * 2)

        a_mat_group = pto.TileGroup([a_mat0, a_mat1])

        b_mat0 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 2 * 2)
        b_mat1 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 2 * 3)
        b_mat_group = pto.TileGroup([b_mat0, b_mat1])

        # LEFT / RIGHT: L0 compute-input buffers
        a_left0 = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
        a_left1 = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=TILE * TILE * 2)
        a_left_group = pto.TileGroup([a_left0, a_left1])

        b_right0 = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)
        b_right1 = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=TILE * TILE * 2)
        b_right_group = pto.TileGroup([b_right0, b_right1])

        # ACC: L0-C accumulator (float32 for precision)
        c_acc0 = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)
        c_acc1 = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=TILE * TILE * 4)
        c_acc_group = pto.TileGroup([c_acc0, c_acc1])

        # -- Compute tile counts -----------------------------------------------
        m_tiles = (M + (TILE - 1)) // TILE
        n_tiles = (N + (TILE - 1)) // TILE
        k_tiles = (K + (TILE - 1)) // TILE
        for i in pto.range(m_tiles):
            for j in pto.range(n_tiles):
                m_off = i * TILE
                n_off = j * TILE
                buff_idx = (i * n_tiles + j) % 2
                l0c_idx = (i * n_tiles + j) % 2
                # ==============================================================
                #  First K tile (k=0): tmatmul — clears ACC, then multiplies
                # ==============================================================
                pv_a0 = a.partition(offsets=[m_off, 0], sizes=[TILE, TILE])
                pv_b0 = b.partition(offsets=[0, n_off], sizes=[TILE, TILE])
                # GM → MAT (MTE2 pipe)

                pto.tload(a_mat_group[buff_idx], pv_a0)
                pto.tload(b_mat_group[buff_idx], pv_b0)
                # MAT → LEFT / RIGHT (MTE1 pipe)
                pto.tmov(a_left_group[buff_idx], a_mat_group[buff_idx])
                pto.tmov(b_right_group[buff_idx], b_mat_group[buff_idx])
                # CUBE matmul (clears ACC first)
                pto.tmatmul(c_acc_group[l0c_idx], a_left_group[buff_idx], b_right_group[buff_idx])

                # ==============================================================
                #  Remaining K tiles (k=1..k_tiles-1): tmatmul_acc — accumulates
                # ==============================================================
                for k in pto.range(1, k_tiles):
                    buff_idx = (buff_idx + 1) % 2
                    k_off = k * TILE
                    pv_a_k = a.partition(offsets=[m_off, k_off],
                                         sizes=[TILE, TILE])
                    pv_b_k = b.partition(offsets=[k_off, n_off],
                                         sizes=[TILE, TILE])
                    # GM → MAT (MTE2)
                    pto.tload(a_mat_group[buff_idx], pv_a_k)
                    pto.tload(b_mat_group[buff_idx], pv_b_k)
                    # MAT → LEFT / RIGHT (MTE1)
                    pto.tmov(a_left_group[buff_idx], a_mat_group[buff_idx])
                    pto.tmov(b_right_group[buff_idx], b_mat_group[buff_idx])
                    # CUBE matmul-accumulate
                    pto.tmatmul_acc(c_acc_group[l0c_idx], c_acc_group[l0c_idx], a_left_group[buff_idx], b_right_group[buff_idx])
                # ==============================================================
                #  Store result: ACC → GM (MTE3 pipe)
                # ==============================================================
                pv_c = c.partition(offsets=[m_off, n_off], sizes=[TILE, TILE])
                pto.tstore(pv_c, c_acc_group[l0c_idx])


# ---------------------------------------------------------------------------
#  NPU launch tests
# ---------------------------------------------------------------------------

def test_npu_launch(db = True, auto_sync = False):
    """Full compile + launch on NPU, with golden comparison."""
    import torch
    import torch_npu

    @pto.jit
    def run():
        if auto_sync == True:
            compiled = pto.compile(matmul_kernel_double_buffer_no_sync, npu_arch="dav-c220-cube",
                                    auto_sync=auto_sync)
        else:
            if db == True:
                compiled = pto.compile(matmul_kernel_double_buffer, npu_arch="dav-c220-cube",
                                    auto_sync=auto_sync)
            else:
                compiled = pto.compile(matmul_kernel, npu_arch="dav-c220-cube",
                                    auto_sync=auto_sync)
        print(f"compiled lib: {compiled.lib_path}", file=sys.stderr)

        device = "npu:6"
        torch.npu.set_device(device)

        # Test shapes: (M, N, K) — all multiples of 64
        shapes = [
            # (M, N, K)
            # (64, 64, 64),       # single tile in each dimension
            # (64, 128, 64),       # single tile in each dimension
            (128, 128, 128),    # 2×2×2 tiles
            (64, 128, 192),     # rectangular: 1×2×3 tiles
            (192, 128, 64),     # rectangular: 3×2×1 tiles
            (128, 256, 128),    # wider: 2×4×2 tiles
            (256, 256, 256),    # larger square: 4×4×4 tiles
        ]

        for (m, n, k) in shapes:
            torch.manual_seed(42)
            a = torch.rand((m, k), device=device, dtype=torch.float16)
            b = torch.rand((k, n), device=device, dtype=torch.float16)
            c = torch.empty((m, n), device=device, dtype=torch.float16)

            pto.launch(compiled, a, b, c)
            torch.npu.synchronize()

            c_ref = torch.matmul(a, b)

            # Debug: show actual vs expected values
            diff = (c - c_ref).abs()
            print(f"  shape ({m},{n},{k}):", file=sys.stderr)
            print(f"    c[0,:8]:     {c[0,:8].tolist()}", file=sys.stderr)
            print(f"    c_ref[0,:8]: {c_ref[0,:8].tolist()}", file=sys.stderr)
            print(f"    c has NaN: {c.isnan().any().item()}", file=sys.stderr)
            print(f"    c all zero: {(c==0).all().item()}", file=sys.stderr)
            print(f"    max |diff|: {diff.max().item():.4f}", file=sys.stderr)
            mismatch = 100*(1-torch.isclose(c,c_ref,rtol=5e-3,atol=5e-3).float().mean().item())
            print(f"    mismatch: {mismatch:.1f}%", file=sys.stderr)

            # float16 matmul tolerance: both use float32 accumulation
            torch.testing.assert_close(c, c_ref, rtol=5e-3, atol=5e-3)
            print(f"  shape ({m}, {n}, {k}): PASS", file=sys.stderr)

    run()
    print("// NPU matmul tests passed.", file=sys.stderr)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--ir-only" in sys.argv:
        test_ir_only()
    else:
        try:
            # test_npu_launch(db = True)
            # test_npu_launch(db = False)
            test_npu_launch(db = True, auto_sync = True)
        except (ImportError, RuntimeError) as e:
            print(f"NPU not available ({e}), falling back to IR-only test.",
                  file=sys.stderr)
            test_ir_only()

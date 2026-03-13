"""End-to-end test: compile a fused cube+vector kernel.

Demonstrates:
  - @pto.kernel with both section_cube() and section_vector()
  - Fused compilation using --cce-aicore-arch=dav-c220

Usage (on NPU machine):
    python3 test_jit_fused_kernel.py

Usage (IR-only, no NPU):
    python3 test_jit_fused_kernel.py --ir-only
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend"))

import pto_frontend as pto

M = pto.DynVar("M")
N = pto.DynVar("N")
K = pto.DynVar("K")

TILE = 64
TILE_M = 64
TILE_N = 128
# ---------------------------------------------------------------------------
#  Matmul + Add fused kernel:
#    section_cube:   C[M, N] = A[M, K] @ B[K, N]
#    section_vector: E[M, N] = D[M, N] + D[M, N]
# ---------------------------------------------------------------------------

@pto.kernel
def matmul_add_kernel(
    a: pto.Tensor[[M, K], pto.float16],
    b: pto.Tensor[[K, N], pto.float16],
    c: pto.Tensor[[M, N], pto.float16],
    d: pto.Tensor[[M, N], pto.float16],
    e: pto.Tensor[[M, N], pto.float16]
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
        for i in pto.range(m_tiles):
            for j in pto.range(n_tiles):
                m_off = i * TILE
                n_off = j * TILE

                # ==============================================================
                #  First K tile (k=0): tmatmul — clears ACC, then multiplies
                # ==============================================================
                pv_a0 = a.partition(offsets=[m_off, 0], sizes=[TILE, TILE])
                pv_b0 = b.partition(offsets=[0, n_off], sizes=[TILE, TILE])
                # GM → MAT (MTE2 pipe)
                pto.tload(a_mat, pv_a0)
                pto.tload(b_mat, pv_b0)

                # MAT → LEFT / RIGHT (MTE1 pipe)
                pto.tmov(a_left, a_mat)
                pto.tmov(b_right, b_mat)
                # CUBE matmul (clears ACC first)
                pto.tmatmul(c_acc, a_left, b_right)

                # ==============================================================
                #  Remaining K tiles (k=1..k_tiles-1): tmatmul_acc — accumulates
                # ==============================================================
                for k in pto.range(1, k_tiles):
                    k_off = k * TILE
                    pv_a_k = a.partition(offsets=[m_off, k_off],
                                         sizes=[TILE, TILE])
                    pv_b_k = b.partition(offsets=[k_off, n_off],
                                         sizes=[TILE, TILE])
                    # GM → MAT (MTE2)
                    pto.tload(a_mat, pv_a_k)
                    pto.tload(b_mat, pv_b_k)
                    # MAT → LEFT / RIGHT (MTE1)
                    pto.tmov(a_left, a_mat)
                    pto.tmov(b_right, b_mat)
                    # CUBE matmul-accumulate
                    pto.tmatmul_acc(c_acc, c_acc, a_left, b_right)

                # ==============================================================
                #  Store result: ACC → GM (MTE3 pipe)
                # ==============================================================
                pv_c = c.partition(offsets=[m_off, n_off], sizes=[TILE, TILE])
                pto.tstore(pv_c, c_acc)


    with pto.section_vector():
        tile_d = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=0)
        tile_e = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC,
                               addr=TILE_M * TILE_N * 4)

        m_loops = (M + (TILE_M - 1)) // TILE_M
        n_loops = (N + (TILE_N - 1)) // TILE_N
        for i in pto.range(m_loops):
            for j in pto.range(n_loops):
                m_offset = i * TILE_M
                n_offset = j * TILE_N

                pv_d = d.partition(offsets=[m_offset, n_offset],
                                   sizes=[TILE_M, TILE_N])
                pv_e = e.partition(offsets=[m_offset, n_offset],
                                   sizes=[TILE_M, TILE_N])
                pto.tload(tile_d, pv_d)
                pto.tadd(tile_e, tile_d, tile_d)
                pto.tstore(pv_e, tile_e)


def test_ir_only():
    """Print the generated IR for inspection."""
    ir = matmul_add_kernel.emit_ir()
    print(ir)
    assert "pto.section.cube" in ir, "Missing section_cube in IR"
    assert "pto.section.vector" in ir, "Missing section_vector in IR"
    print("// IR-only test passed: fused kernel has both cube and vector sections.",
          file=sys.stderr)


def test_compile():
    """Compile the fused kernel to .so."""
    @pto.jit
    def run():
        compiled = pto.compile(matmul_add_kernel)
        print(f"compiled lib: {compiled.lib_path}", file=sys.stderr)
        print("// Fused kernel compilation passed.", file=sys.stderr)

    run()


def test_npu_launch():
    """Full compile + launch on NPU, with golden comparison."""
    import torch
    import torch_npu

    @pto.jit
    def run():
        compiled = pto.compile(matmul_add_kernel, arch="a3", auto_sync=True)
        print(f"compiled lib: {compiled.lib_path}", file=sys.stderr)

        device = "npu:6"
        torch.npu.set_device(device)
        dtype = torch.float16

        shape_m, shape_n, shape_k = 512, 512, 512

        torch.manual_seed(42)
        a = torch.rand((shape_m, shape_k), device=device, dtype=dtype)
        b = torch.rand((shape_k, shape_n), device=device, dtype=dtype)
        c = torch.empty((shape_m, shape_n), device=device, dtype=dtype)
        d = torch.rand((shape_m, shape_n), device=device, dtype=dtype)
        e = torch.empty((shape_m, shape_n), device=device, dtype=dtype)

        pto.launch(compiled, a, b, c, d, e)
        torch.npu.synchronize()

        c_ref = a @ b
        e_ref = d + d
        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
        print("e_ref[0][0:16]", e_ref[0][0:16].cpu().numpy(), file=sys.stderr)
        print("e[0][0:16]", e[0][0:16].cpu().numpy(), file=sys.stderr)
        torch.testing.assert_close(e, e_ref)
        print(f"  fused kernel ({shape_m}x{shape_n}x{shape_k}): PASS",
              file=sys.stderr)

    run()
    print("// Fused kernel NPU launch tests passed.", file=sys.stderr)


if __name__ == "__main__":
    if "--ir-only" in sys.argv:
        test_ir_only()
    elif "--compile-only" in sys.argv:
        test_compile()
    else:
        try:
            test_npu_launch()
        except (ImportError, RuntimeError) as e:
            print(f"NPU not available ({e}), falling back to IR-only test.",
                  file=sys.stderr)
            test_ir_only()

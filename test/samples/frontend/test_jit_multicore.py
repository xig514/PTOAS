"""End-to-end test: multi-core kernels on NPU.

Demonstrates:
  - Multi-core dispatch with block_dim > 1
  - pto.get_block_idx() / pto.get_block_num() for Cube core indexing
  - pto.get_subblock_idx() for Vector sub-block indexing
  - Each core computes a different slice of the output via strided loops

Usage (on NPU machine):
    python3 test_jit_multicore.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend"))

import pto_frontend as pto

# -- Dynamic shape variables ------------------------------------------------
M = pto.DynVar("M")
N = pto.DynVar("N")
K = pto.DynVar("K")

TILE_M = 64
TILE_N = 128
TILE = 64


# ---------------------------------------------------------------------------
#  Multi-core double-buffer vector add
#  Each block processes a slice of M-tile-rows via strided loop.
#  Both vector sub-blocks within each block execute the same computation.
#  Block index = block_idx, total blocks = block_num
# ---------------------------------------------------------------------------

@pto.kernel
def multicore_double_buffer_add_kernel(
    x: pto.Tensor[[M, N], pto.float16],
    y: pto.Tensor[[M, N], pto.float16],
    z: pto.Tensor[[M, N], pto.float16],
    add_scale: bool,
    scale: float,
):
    with pto.section_vector():
        BUF = TILE_M * TILE_N * 2
        a = pto.TileGroup([
            pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=0),
            pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=BUF * 3),
        ])
        b = pto.TileGroup([
            pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=BUF),
            pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=BUF * 4),
        ])
        c = pto.TileGroup([
            pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=BUF * 2),
            pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=BUF * 5),
        ])

        # Distribute M-tiles across blocks
        core_id = pto.get_block_idx()
        num_cores = pto.get_block_num()

        m_tiles = (M + (TILE_M - 1)) // TILE_M
        n_tiles = (N + (TILE_N - 1)) // TILE_N

        # Each block processes M-tiles: core_id, core_id+num_cores, ...
        for i in pto.range(core_id, m_tiles, num_cores):
            for j in pto.range(n_tiles):
                m_off = i * TILE_M
                n_off = j * TILE_N
                pv_x = x.partition(offsets=[m_off, n_off],
                                   sizes=[TILE_M, TILE_N])
                pv_y = y.partition(offsets=[m_off, n_off],
                                   sizes=[TILE_M, TILE_N])
                pv_z = z.partition(offsets=[m_off, n_off],
                                   sizes=[TILE_M, TILE_N])

                buf_idx = (i * n_tiles + j) % 2
                pto.tload(a[buf_idx], pv_x)
                pto.tload(b[buf_idx], pv_y)
                pto.tadd(c[buf_idx], a[buf_idx], b[buf_idx])

                # Conditionally add scalar scale to result
                with pto.if_(add_scale):
                    pto.tadds(c[buf_idx], c[buf_idx], scale)

                pto.tstore(pv_z, c[buf_idx])


# ---------------------------------------------------------------------------
#  Multi-core double-buffer matmul
#  C[M, N] = A[M, K] @ B[K, N]
#  Each Cube core processes a slice of M-tile-rows via strided loop.
#  Cube core index = block_idx, total = block_num
# ---------------------------------------------------------------------------

@pto.kernel
def multicore_matmul_kernel_double_buffer(
    a: pto.Tensor[[M, K], pto.float16],
    b: pto.Tensor[[K, N], pto.float16],
    c: pto.Tensor[[M, N], pto.float16],
):
    with pto.section_cube():
        # -- Allocate tile buffers --
        a_mat0 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
        a_mat1 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=TILE * TILE * 2)
        a_mat_group = pto.TileGroup([a_mat0, a_mat1])

        b_mat0 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 2 * 2)
        b_mat1 = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                              addr=TILE * TILE * 2 * 3)
        b_mat_group = pto.TileGroup([b_mat0, b_mat1])

        a_left0 = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
        a_left1 = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=TILE * TILE * 2)
        a_left_group = pto.TileGroup([a_left0, a_left1])

        b_right0 = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)
        b_right1 = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=TILE * TILE * 2)
        b_right_group = pto.TileGroup([b_right0, b_right1])

        c_acc0 = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)
        c_acc1 = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=TILE * TILE * 4)
        c_acc_group = pto.TileGroup([c_acc0, c_acc1])

        # -- Core distribution --
        core_id = pto.get_block_idx()
        num_cores = pto.get_block_num()

        m_tiles = (M + (TILE - 1)) // TILE
        n_tiles = (N + (TILE - 1)) // TILE
        k_tiles = (K + (TILE - 1)) // TILE

        event_ids = pto.EventIdGroup([pto.EVENT_ID0, pto.EVENT_ID1])

        pto.set_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID0)
        pto.set_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID0)
        pto.set_flag(pto.PIPE_FIX, pto.PIPE_M, pto.EVENT_ID0)

        pto.set_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, pto.EVENT_ID1)
        pto.set_flag(pto.PIPE_M, pto.PIPE_MTE1, pto.EVENT_ID1)
        pto.set_flag(pto.PIPE_FIX, pto.PIPE_M, pto.EVENT_ID1)

        # Each Cube core processes M-tiles: core_id, core_id+num_cores, ...
        for i in pto.range(core_id, m_tiles, num_cores):
            for j in pto.range(n_tiles):
                m_off = i * TILE
                n_off = j * TILE
                buff_idx = (i * n_tiles + j) % 2
                l0c_idx = (i * n_tiles + j) % 2

                # First K tile (k=0): tmatmul — clears ACC
                pv_a0 = a.partition(offsets=[m_off, 0], sizes=[TILE, TILE])
                pv_b0 = b.partition(offsets=[0, n_off], sizes=[TILE, TILE])
                pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, event_ids[buff_idx])
                pto.tload(a_mat_group[buff_idx], pv_a0)
                pto.tload(b_mat_group[buff_idx], pv_b0)

                pto.set_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, event_ids[buff_idx])
                pto.wait_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, event_ids[buff_idx])
                pto.wait_flag(pto.PIPE_M, pto.PIPE_MTE1, event_ids[buff_idx])
                pto.tmov(a_left_group[buff_idx], a_mat_group[buff_idx])
                pto.tmov(b_right_group[buff_idx], b_mat_group[buff_idx])
                pto.set_flag(pto.PIPE_MTE1, pto.PIPE_M, event_ids[buff_idx])
                pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_M, event_ids[buff_idx])
                pto.set_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, event_ids[buff_idx])
                pto.wait_flag(pto.PIPE_FIX, pto.PIPE_M, event_ids[l0c_idx])
                pto.tmatmul(c_acc_group[l0c_idx], a_left_group[buff_idx], b_right_group[buff_idx])
                pto.set_flag(pto.PIPE_M, pto.PIPE_MTE1, event_ids[buff_idx])

                # Remaining K tiles: tmatmul_acc — accumulates
                for k in pto.range(1, k_tiles):
                    buff_idx = (buff_idx + 1) % 2
                    k_off = k * TILE
                    pv_a_k = a.partition(offsets=[m_off, k_off],
                                         sizes=[TILE, TILE])
                    pv_b_k = b.partition(offsets=[k_off, n_off],
                                         sizes=[TILE, TILE])
                    pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, event_ids[buff_idx])
                    pto.tload(a_mat_group[buff_idx], pv_a_k)
                    pto.tload(b_mat_group[buff_idx], pv_b_k)
                    pto.set_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, event_ids[buff_idx])
                    pto.wait_flag(pto.PIPE_MTE2, pto.PIPE_MTE1, event_ids[buff_idx])
                    pto.wait_flag(pto.PIPE_M, pto.PIPE_MTE1, event_ids[buff_idx])
                    pto.tmov(a_left_group[buff_idx], a_mat_group[buff_idx])
                    pto.tmov(b_right_group[buff_idx], b_mat_group[buff_idx])
                    pto.set_flag(pto.PIPE_MTE1, pto.PIPE_M, event_ids[buff_idx])
                    pto.wait_flag(pto.PIPE_MTE1, pto.PIPE_M, event_ids[buff_idx])
                    pto.set_flag(pto.PIPE_MTE1, pto.PIPE_MTE2, event_ids[buff_idx])
                    pto.tmatmul_acc(c_acc_group[l0c_idx], c_acc_group[l0c_idx], a_left_group[buff_idx], b_right_group[buff_idx])
                    pto.set_flag(pto.PIPE_M, pto.PIPE_MTE1, event_ids[buff_idx])

                # Store result: ACC → GM
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


# ---------------------------------------------------------------------------
#  NPU launch tests
# ---------------------------------------------------------------------------

def test_multicore_add():
    """Multi-core double-buffer add on NPU, with scalar params."""
    import torch
    import torch_npu

    BLOCK_DIM = 24

    @pto.jit
    def run():
        compiled = pto.compile(multicore_double_buffer_add_kernel, auto_sync=True)
        print(f"compiled lib (multicore add): {compiled.lib_path}",
              file=sys.stderr)

        device = "npu:6"
        torch.npu.set_device(device)
        dtype = torch.float16

        shapes = [
            (12288, 4096),
        ]

        for shape in shapes:
            # --- Test 1: add_scale=False, scale is ignored ---
            torch.manual_seed(42)
            x = torch.rand(shape, device=device, dtype=dtype)
            y = torch.rand(shape, device=device, dtype=dtype)
            z = torch.empty(shape, device=device, dtype=dtype)

            pto.launch(compiled, x, y, z, False, 0.0, block_dim=BLOCK_DIM)
            torch.npu.synchronize()

            z_ref = x + y
            torch.testing.assert_close(z, z_ref)
            print(f"  add (add_scale=False) shape {shape}: PASS",
                  file=sys.stderr)

            # --- Test 2: add_scale=True, scale=0.5 ---
            scale_val = 0.5
            z2 = torch.empty(shape, device=device, dtype=dtype)

            pto.launch(compiled, x, y, z2, True, scale_val, block_dim=BLOCK_DIM)
            torch.npu.synchronize()

            z2_ref = x + y + scale_val
            torch.testing.assert_close(z2, z2_ref, rtol=1e-3, atol=1e-3)
            print(f"  add (add_scale=True, scale={scale_val}) shape {shape}: PASS",
                  file=sys.stderr)

    run()
    print("// Multi-core add tests passed.", file=sys.stderr)


def test_multicore_matmul():
    """Multi-core double-buffer matmul on NPU."""
    import torch
    import torch_npu

    BLOCK_DIM = 4  # 4 Cube cores

    @pto.jit
    def run():
        compiled = pto.compile(multicore_matmul_kernel_double_buffer,
                               auto_sync=False)
        print(f"compiled lib (multicore matmul): {compiled.lib_path}",
              file=sys.stderr)

        device = "npu:6"
        torch.npu.set_device(device)

        shapes = [
            # (M, N, K) — M must have enough tiles to split across cores
            (256, 256, 256),
            (512, 256, 128),
            (256, 512, 256),
        ]

        for (m, n, k) in shapes:
            torch.manual_seed(42)
            a = torch.rand((m, k), device=device, dtype=torch.float16)
            b = torch.rand((k, n), device=device, dtype=torch.float16)
            c = torch.empty((m, n), device=device, dtype=torch.float16)

            pto.launch(compiled, a, b, c, block_dim=BLOCK_DIM)
            torch.npu.synchronize()

            c_ref = torch.matmul(a, b)

            diff = (c - c_ref).abs()
            print(f"  shape ({m},{n},{k}), block_dim={BLOCK_DIM}:",
                  file=sys.stderr)
            print(f"    max |diff|: {diff.max().item():.4f}", file=sys.stderr)
            mismatch = 100 * (1 - torch.isclose(c, c_ref, rtol=5e-3, atol=5e-3).float().mean().item())
            print(f"    mismatch: {mismatch:.1f}%", file=sys.stderr)

            torch.testing.assert_close(c, c_ref, rtol=5e-3, atol=5e-3)
            print(f"  shape ({m}, {n}, {k}): PASS", file=sys.stderr)

    run()
    print("// Multi-core matmul tests passed.", file=sys.stderr)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        test_multicore_add()
        # test_multicore_matmul()
    except (ImportError, RuntimeError) as e:
        print(f"NPU not available ({e}), falling back.",
              file=sys.stderr)

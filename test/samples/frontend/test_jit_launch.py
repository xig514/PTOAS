"""End-to-end test: compile + launch dynamic-shape vector add on NPU.

Demonstrates:
  - @pto.kernel with DynVar dynamic shapes
  - pto.compile() → CompiledKernel
  - pto.launch() → execute on NPU via ctypes
  - Verification against torch reference

Usage (on NPU machine):
    python3 test_jit_launch.py

Usage (IR-only, no NPU):
    python3 test_jit_launch.py --ir-only
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend"))

import pto_frontend as pto

# -- Dynamic shape variables ------------------------------------------------
M = pto.DynVar("M")
N = pto.DynVar("N")

TILE_M = 64
TILE_N = 128


@pto.kernel
def dynamic_add_kernel(
    x: pto.Tensor[[M, N], pto.float16],
    y: pto.Tensor[[M, N], pto.float16],
    z: pto.Tensor[[M, N], pto.float16],
):
    with pto.section_vector():
        tile_a = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=0)
        tile_b = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC,
                               addr=TILE_M * TILE_N * 2)
        tile_c = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC,
                               addr=TILE_M * TILE_N * 4)

        m_loops = (M + (TILE_M - 1)) // TILE_M
        n_loops = (N + (TILE_N - 1)) // TILE_N
        for i in pto.range(m_loops):
            for j in pto.range(n_loops):
                m_offset = i * TILE_M
                n_offset = j * TILE_N

                pv_x = x.partition(offsets=[m_offset, n_offset],
                                   sizes=[TILE_M, TILE_N])
                pv_y = y.partition(offsets=[m_offset, n_offset],
                                   sizes=[TILE_M, TILE_N])
                pv_z = z.partition(offsets=[m_offset, n_offset],
                                   sizes=[TILE_M, TILE_N])
                pto.tload(tile_a, pv_x)
                pto.tload(tile_b, pv_y)
                pto.tadd(tile_c, tile_a, tile_b)
                pto.tstore(pv_z, tile_c)


@pto.kernel
def double_buffer_add_kernel(
    x: pto.Tensor[[M, N], pto.float16],
    y: pto.Tensor[[M, N], pto.float16],
    z: pto.Tensor[[M, N], pto.float16],
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

        m_tiles = (M + (TILE_M - 1)) // TILE_M
        n_tiles = (N + (TILE_N - 1)) // TILE_N

        for i in pto.range(m_tiles):
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
                pto.tstore(pv_z, c[buf_idx])


def test_npu_launch():
    """Full compile + launch on NPU, with golden comparison."""
    import torch
    import torch_npu

    @pto.jit
    def run():
        compiled = pto.compile(dynamic_add_kernel, auto_sync=True)
        print(f"compiled lib: {compiled.lib_path}", file=sys.stderr)

        device = "npu:6"
        torch.npu.set_device(device)
        dtype = torch.float16

        shapes = [
            # (64, 128),
            (512, 512),
        ]

        for shape in shapes:
            torch.manual_seed(42)
            x = torch.rand(shape, device=device, dtype=dtype)
            y = torch.rand(shape, device=device, dtype=dtype)
            z = torch.empty(shape, device=device, dtype=dtype)

            pto.launch(compiled, x, y, z)
            torch.npu.synchronize()

            z_ref = x + y
            torch.testing.assert_close(z, z_ref)
            print(f"  shape {shape}: PASS", file=sys.stderr)

    run()
    print("// NPU launch tests passed.", file=sys.stderr)


def test_double_buffer_launch():
    """Compile + launch the double-buffer kernel on NPU."""
    import torch
    import torch_npu

    @pto.jit
    def run():
        compiled = pto.compile(double_buffer_add_kernel, auto_sync=True)
        print(f"compiled lib (double-buffer): {compiled.lib_path}",
              file=sys.stderr)

        device = "npu:6"
        torch.npu.set_device(device)
        dtype = torch.float16

        shapes = [
            (128, 256),
            (256, 512),
        ]

        for shape in shapes:
            torch.manual_seed(42)
            x = torch.rand(shape, device=device, dtype=dtype)
            y = torch.rand(shape, device=device, dtype=dtype)
            z = torch.empty(shape, device=device, dtype=dtype)

            pto.launch(compiled, x, y, z)
            torch.npu.synchronize()

            z_ref = x + y
            torch.testing.assert_close(z, z_ref)
            print(f"  double-buffer shape {shape}: PASS", file=sys.stderr)

    run()
    print("// Double-buffer NPU launch tests passed.", file=sys.stderr)


if __name__ == "__main__":
    if "--ir-only" in sys.argv:
        test_ir_only()
    else:
        try:
            # test_npu_launch()
            test_double_buffer_launch()
        except (ImportError, RuntimeError) as e:
            print(f"NPU not available ({e}), falling back to IR-only test.",
                  file=sys.stderr)
            test_ir_only()

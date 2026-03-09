"""Dynamic-shape vector add using Tensor[[M, N], dtype] with DynVar.

Demonstrates:
  - DynVar for dynamic shapes
  - Tensor[[M, N], dtype] annotation
  - Additional int parameter (tile_m)
  - Nested for_range loops with dynamic bounds computed from DynVar arithmetic
  - Offset computation inside loops (i * TILE_N, etc.)
  - Tile load / add / store with partition views

Usage:
    python3 test_dynamic_add.py > dynamic_add.pto
    ptoas dynamic_add.pto --pto-level=level3
"""

import sys
import os

# Ensure pto_frontend is importable
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
    # Allocate tile buffers in VEC memory
    tile_a = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=0)
    tile_b = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC,
                           addr=TILE_M * TILE_N * 2)
    tile_c = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC,
                           addr=TILE_M * TILE_N * 4)

    # Compute loop bounds from dynamic shapes
    m_loops = (M + (TILE_M - 1)) // TILE_M
    n_loops = (N + (TILE_N - 1)) // TILE_N

    with pto.for_range(0, m_loops, 1) as i:
        with pto.for_range(0, n_loops, 1) as j:
            # Offset computation inside the loop
            m_offset = i * TILE_M
            n_offset = j * TILE_N

            # Partition views at computed offsets with static tile sizes
            pv_x = x.partition(offsets=[m_offset, n_offset],
                               sizes=[TILE_M, TILE_N])
            pv_y = y.partition(offsets=[m_offset, n_offset],
                               sizes=[TILE_M, TILE_N])
            pv_z = z.partition(offsets=[m_offset, n_offset],
                               sizes=[TILE_M, TILE_N])

            # Load -> Add -> Store
            pto.tload(pv_x, tile_a)
            pto.tload(pv_y, tile_b)
            pto.tadd(tile_a, tile_b, tile_c)
            pto.tstore(pv_z, tile_c)


if __name__ == "__main__":
    ir = dynamic_add_kernel.emit_ir()
    print(ir)

    # Verify key IR patterns
    checks = [
        "scf.for",
        "pto.partition_view",
        "pto.tload",
        "pto.tadd",
        "pto.tstore",
        "arith.addi",
        "arith.muli",
        "arith.divsi",
    ]
    for pat in checks:
        assert pat in ir, f"Missing expected pattern in IR: {pat}"

    print("// All IR checks passed.", file=sys.stderr)

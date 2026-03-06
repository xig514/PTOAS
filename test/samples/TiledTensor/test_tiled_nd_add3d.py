"""Test N-D tiling for 3D tensor element-wise add (using 2D tile_buf).

Since PTO tile_buf is restricted to rank-2, we handle 3D tensors by
tiling the last 2 dimensions and using for_range for the first dimension.

Scenario: C[Z, M, N] = A[Z, M, N] + B[Z, M, N]
"""

import pto_frontend as pto


@pto.kernel
def add_3d_tiled(
    a: pto.Tensor(pto.float16, 2),  # [M, N] — 2D slice
    b: pto.Tensor(pto.float16, 2),
    c: pto.Tensor(pto.float16, 2),
    num_slices: int,  # Z dimension
):
    """Element-wise add with tiling on 2D slices, looped over Z."""

    TILE_M = 32
    TILE_N = 32

    tiled_a = a.tile_nd(
        tile_sizes=(TILE_M, TILE_N),
        tile_dims=[0, 1]
    )
    tiled_b = b.tile_nd(
        tile_sizes=(TILE_M, TILE_N),
        tile_dims=[0, 1]
    )
    tiled_c = c.tile_nd(
        tile_sizes=(TILE_M, TILE_N),
        tile_dims=[0, 1]
    )

    # Distribute the first tile dimension across cores
    dist_a = tiled_a.distribute_nd(core_grid=(4, 1))

    tile_a = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=0)
    tile_b = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC,
                           addr=TILE_M * TILE_N * 2)
    tile_c = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC,
                           addr=TILE_M * TILE_N * 4)

    # Outer loop over "Z" slices
    with pto.for_range(0, num_slices) as z:
        with dist_a.for_each() as (tile_idx, part_a):
            part_b = tiled_b[tile_idx]
            part_c = tiled_c[tile_idx]

            pto.tload(part_a, tile_a)
            pto.tload(part_b, tile_b)
            pto.tadd(tile_a, tile_b, tile_c)
            pto.tstore(part_c, tile_c)


if __name__ == "__main__":
    add_3d_tiled()

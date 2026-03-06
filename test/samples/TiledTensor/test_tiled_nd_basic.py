"""Test basic N-D tiling without distribution.

Demonstrates multi-dimensional tiling with explicit iteration ranges.
"""

import pto_frontend as pto


@pto.kernel
def basic_nd_tiling(
    x: pto.Tensor(pto.float16, 2),  # [M, N]
):
    # Tile both dimensions
    tiled = x.tile_nd(
        tile_sizes=(32, 128),
        tile_dims=[0, 1]  # Tile both dimensions
    )

    # Iterate over subset of tiles
    # Dim 0: 2 tiles (out of M/32), Dim 1: 2 tiles (out of N/128)
    with tiled.for_each(ranges=[(0, 2, 1), (0, 2, 1)]) as (tile_idx, partition):
        # tile_idx is (i, j)
        # partition is [32, 128]

        tile_buf = pto.make_tile((32, 128), pto.float16, pto.VEC, addr=0)
        pto.tload(partition, tile_buf)
        pto.tstore(partition, tile_buf)


if __name__ == "__main__":
    basic_nd_tiling()

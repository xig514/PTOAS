"""Test N-D tiling for 3D Add operator.

Scenario: 3D tensor addition with shape [2048, 2048, 2048]
- Tile all three dimensions
- Tile sizes: 256x256x256
- Multi-core distribution: 8x8x8 core grid
"""

import pto_frontend as pto


@pto.kernel
def add_3d_tiled_nd(
    a: pto.Tensor(pto.float16, 3),  # [2048, 2048, 2048]
    b: pto.Tensor(pto.float16, 3),  # [2048, 2048, 2048]
    c: pto.Tensor(pto.float16, 3),  # [2048, 2048, 2048]
):
    # Tile all three dimensions with 256x256x256 tiles
    # Core grid: 8x8x8 = 512 cores

    tiled_a = a.tile_nd(
        tile_sizes=(256, 256, 256),
        tile_dims=[0, 1, 2]  # Tile all dimensions
    )
    tiled_b = b.tile_nd(
        tile_sizes=(256, 256, 256),
        tile_dims=[0, 1, 2]
    )
    tiled_c = c.tile_nd(
        tile_sizes=(256, 256, 256),
        tile_dims=[0, 1, 2]
    )

    # Distribute across 8x8x8 core grid
    dist_a = tiled_a.distribute_nd(core_grid=(8, 8, 8))
    dist_b = tiled_b.distribute_nd(core_grid=(8, 8, 8))
    dist_c = tiled_c.distribute_nd(core_grid=(8, 8, 8))

    # Each core processes its assigned tile
    with dist_a.for_each() as (tile_idx, part_a):
        # tile_idx is (i, j, k) for the 3D grid
        # Each partition is [256, 256, 256]

        # Get corresponding partitions for b and c
        part_b = tiled_b[tile_idx]
        part_c = tiled_c[tile_idx]

        # Allocate tile buffers
        tile_a = pto.make_tile((256, 256, 256), pto.float16, pto.VEC, addr=0)
        tile_b = pto.make_tile((256, 256, 256), pto.float16, pto.VEC, addr=0x100000)
        tile_c = pto.make_tile((256, 256, 256), pto.float16, pto.VEC, addr=0x200000)

        # Load inputs
        pto.tload(part_a, tile_a)
        pto.tload(part_b, tile_b)

        # Compute: c = a + b
        pto.tadd(tile_a, tile_b, tile_c)

        # Store result
        pto.tstore(tile_c, part_c)


if __name__ == "__main__":
    add_3d_tiled_nd()

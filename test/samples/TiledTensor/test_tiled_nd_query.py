"""Test N-D tiling for BSND Query tensor.

Scenario: Query tensor with shape BSND (Batch, Sequence, Num_heads, Dim_head)
- Tile dimensions: B, S, N (indices 0, 1, 2)
- Tile sizes: B=1, S=128, N=1, D=full (64)
- Multi-core distribution: 2x4x2 core grid for (B, S, N)
"""

import pto_frontend as pto


@pto.kernel
def query_tiled_nd(
    query: pto.Tensor(pto.float16, 4),  # [B, S, N, D]
):
    # Query shape: [4, 512, 8, 64]
    # Tile: B=1, S=128, N=1, D=64 (full)
    # Core grid: 2x4x2 = 16 cores

    tiled = query.tile_nd(
        tile_sizes=(1, 128, 1, 64),
        tile_dims=[0, 1, 2]  # Tile B, S, N dimensions
    )

    # Distribute across 2x4x2 core grid
    distributed = tiled.distribute_nd(core_grid=(2, 4, 2))

    # Each core processes its assigned tiles
    with distributed.for_each() as (tile_idx, partition):
        # tile_idx is (b_idx, s_idx, n_idx)
        # partition is a view of shape [1, 128, 1, 64]

        # Allocate tile buffer
        tile_buf = pto.make_tile((1, 128, 1, 64), pto.float16, pto.VEC, addr=0)

        # Load the partition
        pto.tload(partition, tile_buf)

        # Process (example: just store back)
        pto.tstore(tile_buf, partition)


if __name__ == "__main__":
    query_tiled_nd()

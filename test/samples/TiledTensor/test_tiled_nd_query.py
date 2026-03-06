"""Test N-D tiling for multi-head attention Query tensor (2D slices).

Since PTO tile_buf is restricted to rank-2, we handle 4D BSND tensors
by using nested for_range loops for outer dimensions (B, N) and
tile_nd for the inner (S, D) dimensions.
"""

import pto_frontend as pto


@pto.kernel
def query_tiled_2d(
    query: pto.Tensor(pto.float16, 2),  # [S, D] — single head slice
):
    """Tile a 2D query tensor along the S dimension with distribution."""

    TILE_S = 128
    TILE_D = 64

    tiled = query.tile_nd(
        tile_sizes=(TILE_S, TILE_D),
        tile_dims=[0]  # Tile S dimension, D is kept whole
    )

    # Distribute across 4 cores
    distributed = tiled.distribute_nd(core_grid=(4,))

    with distributed.for_each() as (tile_idx, partition):
        tile_buf = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0)
        pto.tload(partition, tile_buf)
        pto.tstore(partition, tile_buf)


@pto.kernel
def query_batched_heads(
    query: pto.Tensor(pto.float16, 2),  # [S, D] — single batch+head
    num_batches: int,
    num_heads: int,
):
    """Simulate BSND iteration with nested for_range over B and N.

    In practice, the caller reshapes/indexes the 4D tensor to provide
    a 2D [S, D] view for each (batch, head) combination.
    """
    TILE_S = 128
    TILE_D = 64

    tiled = query.tile_nd(
        tile_sizes=(TILE_S, TILE_D),
        tile_dims=[0]
    )

    tile_buf = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0)

    # Outer loops over batch and head dimensions
    with pto.for_range(0, num_batches) as b:
        with pto.for_range(0, num_heads) as n:
            # Inner tiled loop over sequence dimension
            with tiled.for_each() as (tile_idx, partition):
                pto.tload(partition, tile_buf)
                pto.tstore(partition, tile_buf)


if __name__ == "__main__":
    query_tiled_2d()

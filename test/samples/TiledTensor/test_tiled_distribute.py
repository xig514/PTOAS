"""Test: TiledTensor.distribute — even multi-core distribution.

Covers: TiledTensor.distribute, DistributedTiledTensor.for_each,
        get_block_idx / get_block_num, index_cast, ceildiv, min clamping.
"""
import pto_frontend as pto


@pto.kernel
def distributed_add(
    x: pto.Tensor(pto.float16, 2),
    y: pto.Tensor(pto.float16, 2),
    z: pto.Tensor(pto.float16, 2),
):
    TILE = 32

    tile_x = pto.make_tile((TILE, TILE), pto.float16, pto.VEC, addr=0)
    tile_y = pto.make_tile((TILE, TILE), pto.float16, pto.VEC, addr=TILE * TILE * 2)
    tile_z = pto.make_tile((TILE, TILE), pto.float16, pto.VEC, addr=TILE * TILE * 4)

    x_dist = x.tile(dim=0, tile_sizes=(TILE, TILE)).distribute()
    y_tiled = y.tile(dim=0, tile_sizes=(TILE, TILE))
    z_tiled = z.tile(dim=0, tile_sizes=(TILE, TILE))

    with x_dist.for_each() as (i, x_view):
        y_view = y_tiled[i]
        z_view = z_tiled[i]

        pto.tload(tile_x, x_view)
        pto.tload(tile_y, y_view)
        pto.tadd(tile_z, tile_x, tile_y)
        pto.tstore(z_view, tile_z)


if __name__ == "__main__":
    distributed_add()

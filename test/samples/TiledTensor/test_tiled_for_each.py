"""Test: TiledTensor.for_each — basic tile iteration over one dimension.

Covers: _TensorProxy.tile, TiledTensor.for_each, tload/tadd/tstore with
        tile-generated partition views.
"""
import pto_frontend as pto


@pto.kernel
def tiled_add(
    x: pto.Tensor(pto.float16, 2),
    y: pto.Tensor(pto.float16, 2),
    z: pto.Tensor(pto.float16, 2),
):
    TILE = 32

    tile_x = pto.make_tile((TILE, TILE), pto.float16, pto.VEC, addr=0)
    tile_y = pto.make_tile((TILE, TILE), pto.float16, pto.VEC, addr=TILE * TILE * 2)
    tile_z = pto.make_tile((TILE, TILE), pto.float16, pto.VEC, addr=TILE * TILE * 4)

    x_tiled = x.tile(dim=0, tile_sizes=(TILE, TILE))
    y_tiled = y.tile(dim=0, tile_sizes=(TILE, TILE))
    z_tiled = z.tile(dim=0, tile_sizes=(TILE, TILE))

    with x_tiled.for_each() as (i, x_view):
        y_view = y_tiled[i]
        z_view = z_tiled[i]

        pto.tload(x_view, tile_x)
        pto.tload(y_view, tile_y)
        pto.tadd(tile_x, tile_y, tile_z)
        pto.tstore(tile_z, z_view)


if __name__ == "__main__":
    tiled_add()

"""Test: vector add using the pto_frontend high-level API.

Covers: @kernel, Tensor, make_tile, tload, tadd, tstore, slicing.
"""
import pto_frontend as pto


@pto.kernel
def vector_add(
    x: pto.Tensor(pto.float16, 2),
    y: pto.Tensor(pto.float16, 2),
    z: pto.Tensor(pto.float16, 2),
):
    tile_x = pto.make_tile((32, 32), pto.float16, pto.VEC, addr=0)
    tile_y = pto.make_tile((32, 32), pto.float16, pto.VEC, addr=32 * 32 * 2)
    tile_z = pto.make_tile((32, 32), pto.float16, pto.VEC, addr=32 * 32 * 4)

    pto.tload(x[0:32, 0:32], tile_x)
    pto.tload(y[0:32, 0:32], tile_y)
    pto.tadd(tile_x, tile_y, tile_z)
    pto.tstore(tile_z, z[0:32, 0:32])


if __name__ == "__main__":
    vector_add()

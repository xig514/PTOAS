"""Test: dynamic shapes with for_range and Tensor.partition.

Covers: Tensor.shape[i], for_range, partition (dynamic offsets, static sizes).
"""
import pto_frontend as pto


@pto.kernel
def dynamic_add(
    x: pto.Tensor(pto.float32, 2),
    y: pto.Tensor(pto.float32, 2),
    z: pto.Tensor(pto.float32, 2),
):
    M = x.shape[0]

    tile_x = pto.make_tile((32, 32), pto.float32, pto.VEC, addr=0)
    tile_y = pto.make_tile((32, 32), pto.float32, pto.VEC, addr=32 * 32 * 4)
    tile_z = pto.make_tile((32, 32), pto.float32, pto.VEC, addr=32 * 32 * 8)

    with pto.for_range(0, M, 32) as i:
        pto.tload(x.partition([i, 0], [32, 32]), tile_x)
        pto.tload(y.partition([i, 0], [32, 32]), tile_y)
        pto.tadd(tile_x, tile_y, tile_z)
        pto.tstore(tile_z, z.partition([i, 0], [32, 32]))


if __name__ == "__main__":
    dynamic_add()

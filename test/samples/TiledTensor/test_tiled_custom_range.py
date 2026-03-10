"""Test: TiledTensor.for_each with custom start/end — causal-like scenario.

Covers: for_each(start, end) with runtime-computed bounds, simulating a
        scenario where each core handles a custom range of tiles.
        Also covers ensure_index_ssa i64->index conversion for get_block_idx.
"""
import pto_frontend as pto


@pto.kernel
def custom_range_add(
    x: pto.Tensor(pto.float16, 2),
    z: pto.Tensor(pto.float16, 2),
):
    TILE = 32

    tile_x = pto.make_tile((TILE, TILE), pto.float16, pto.VEC, addr=0)
    tile_z = pto.make_tile((TILE, TILE), pto.float16, pto.VEC, addr=TILE * TILE * 2)

    x_tiled = x.tile(dim=0, tile_sizes=(TILE, TILE))
    z_tiled = z.tile(dim=0, tile_sizes=(TILE, TILE))

    # Simulate custom schedule: core handles tiles [core_id*2, core_id*2+2)
    # get_block_idx() returns i64; ScalarValue arithmetic keeps i64,
    # and for_each internally converts to index via ensure_index_ssa.
    core_id = pto.get_block_idx()
    start = core_id * 2
    end = start + 2

    with x_tiled.for_each(start=start, end=end) as (i, x_view):
        z_view = z_tiled[i]
        pto.tload(tile_x, x_view)
        pto.tadd(tile_z, tile_x, tile_x)
        pto.tstore(z_view, tile_z)


if __name__ == "__main__":
    custom_range_add()

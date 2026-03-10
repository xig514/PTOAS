"""Test new Layout-based tiling API.

Demonstrates the enhanced Layout system where:
1. Tensor carries TensorLayout (shape + stride)
2. TileLayout describes tile pattern
3. split_* functions return TiledView
4. Coordinate-based tile access
"""

import pto_frontend as pto


@pto.kernel
def layout_even_split(
    tensor: pto.Tensor(pto.float16, 2),  # [dynamic, dynamic]
    out: pto.Tensor(pto.float16, 2),
):
    """Even split across cores using Layout API."""

    # Get tensor layout (shape + stride)
    tensor_layout = tensor.get_layout()

    # Define tile pattern
    tile_layout = pto.TileLayout(shape=(64, 128))

    # Get core information
    num_cores = pto.get_block_num()
    core_id = pto.get_block_idx()

    # Split tensor evenly across cores
    tiled = pto.split_even(tensor_layout, tile_layout, num_cores, core_id)

    # Allocate tile buffer
    tile_buf = pto.make_tile((64, 128), pto.float16, pto.VEC, addr=0)

    # Iterate over assigned tiles
    with tiled.for_each() as coord:
        # coord is TileCoordinate (tile_i, tile_j)
        # Load tile at coordinate
        pto.tload_tile(tile_buf, tensor, coord, tile_layout)

        # Store tile to output
        pto.tstore_tile(out, tile_buf, coord, tile_layout)


@pto.kernel
def layout_sequential_split(
    tensor: pto.Tensor(pto.float32, 2),  # [dynamic, dynamic]
    out: pto.Tensor(pto.float32, 2),
):
    """Sequential iteration over all tiles (no distribution)."""

    tensor_layout = tensor.get_layout()
    tile_layout = pto.TileLayout(shape=(32, 64))

    # Split without distribution - iterate all tiles
    tiled = pto.split_sequential(tensor_layout, tile_layout)

    tile_buf = pto.make_tile((32, 64), pto.float32, pto.VEC, addr=0)

    with tiled.for_each() as coord:
        pto.tload_tile(tile_buf, tensor, coord, tile_layout)
        pto.tstore_tile(out, tile_buf, coord, tile_layout)


if __name__ == "__main__":
    layout_even_split()

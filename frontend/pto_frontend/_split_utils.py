"""Utility functions for tensor splitting and distribution.

Provides various strategies for splitting tensors into tiles and
distributing them across cores.
"""

from ._layout_v2 import TensorLayout, TileLayout, TiledView
from ._scalar import ScalarValue
from ._utils import ensure_index_ssa
from mlir.dialects import arith


def split_even(tensor_layout, tile_layout, num_cores, core_id):
    """Split tensor evenly across cores.

    Divides the tensor into tiles and distributes them evenly across cores.
    Each core gets approximately the same number of tiles.

    Parameters
    ----------
    tensor_layout : TensorLayout
        Layout of the tensor to split
    tile_layout : TileLayout
        Layout of each tile
    num_cores : int or ScalarValue
        Total number of cores
    core_id : int or ScalarValue
        ID of current core (0-indexed)

    Returns
    -------
    TiledView
        View containing this core's assigned tile ranges

    Example
    -------
    >>> tensor_layout = tensor.get_layout()
    >>> tile_layout = TileLayout(shape=(64, 128))
    >>> tiled = split_even(tensor_layout, tile_layout, num_cores, core_id)
    >>> with tiled.for_each() as coord:
    ...     # Process tile at coord
    """
    from ._ir_builder import get_builder
    builder = get_builder()

    assert tensor_layout.rank == tile_layout.rank

    ranges = []

    # For simplicity, split along the first dimension only
    # (Can be extended to multi-dimensional splitting)
    for dim in range(tensor_layout.rank):
        if dim == 0:
            # Split this dimension across cores
            tensor_size = tensor_layout.get_shape(dim)
            tile_size = tile_layout.shape[dim]

            # Compute number of tiles: ceildiv(tensor_size, tile_size)
            tensor_size_ssa = ensure_index_ssa(tensor_size)
            tile_size_ssa = builder.constant_index(tile_size)

            tile_size_minus_1 = builder.constant_index(tile_size - 1)
            numerator = arith.AddIOp(tensor_size_ssa, tile_size_minus_1).result
            num_tiles = arith.DivSIOp(numerator, tile_size_ssa).result

            # Compute tiles per core: ceildiv(num_tiles, num_cores)
            num_cores_ssa = ensure_index_ssa(num_cores)
            one = builder.constant_index(1)
            num_cores_minus_1 = arith.SubIOp(num_cores_ssa, one).result
            tiles_numerator = arith.AddIOp(num_tiles, num_cores_minus_1).result
            tiles_per_core = arith.DivSIOp(tiles_numerator, num_cores_ssa).result

            # Compute this core's range
            core_id_ssa = ensure_index_ssa(core_id)
            raw_start = arith.MulIOp(core_id_ssa, tiles_per_core).result
            start = _index_min(raw_start, num_tiles)

            core_id_plus_1 = arith.AddIOp(core_id_ssa, one).result
            raw_end = arith.MulIOp(core_id_plus_1, tiles_per_core).result
            end = _index_min(raw_end, num_tiles)

            ranges.append((ScalarValue(start), ScalarValue(end), 1))
        else:
            # Iterate all tiles in other dimensions
            tensor_size = tensor_layout.get_shape(dim)
            tile_size = tile_layout.shape[dim]

            tensor_size_ssa = ensure_index_ssa(tensor_size)
            tile_size_ssa = builder.constant_index(tile_size)

            tile_size_minus_1 = builder.constant_index(tile_size - 1)
            numerator = arith.AddIOp(tensor_size_ssa, tile_size_minus_1).result
            num_tiles = arith.DivSIOp(numerator, tile_size_ssa).result

            ranges.append((0, ScalarValue(num_tiles), 1))

    return TiledView(tensor_layout, tile_layout, ranges)


def split_causal(tensor_layout, tile_layout, num_cores, core_id):
    """Split tensor with causal masking pattern.

    For attention mechanisms with causal masking, where each query position
    can only attend to keys up to its position. This creates a triangular
    access pattern.

    Parameters
    ----------
    tensor_layout : TensorLayout
        Layout of the tensor to split (typically query tensor)
    tile_layout : TileLayout
        Layout of each tile
    num_cores : int or ScalarValue
        Total number of cores
    core_id : int or ScalarValue
        ID of current core (0-indexed)

    Returns
    -------
    TiledView
        View containing this core's assigned tile ranges with causal pattern

    Example
    -------
    >>> # For FlashAttention with causal masking
    >>> q_layout = query.get_layout()
    >>> tile_layout = TileLayout(shape=(128, 64))
    >>> tiled = split_causal(q_layout, tile_layout, num_cores, core_id)
    >>> with tiled.for_each() as q_coord:
    ...     # Only process K/V tiles up to q_coord[0]
    ...     for kv_idx in range(q_coord[0] + 1):
    ...         # Causal attention computation
    """
    from ._ir_builder import get_builder
    builder = get_builder()

    assert tensor_layout.rank == tile_layout.rank

    # Causal splitting: distribute tiles such that each core gets
    # a contiguous range, but the total work is balanced considering
    # the triangular pattern

    # For simplicity, use even split on first dimension
    # In practice, you'd want to balance the triangular workload
    ranges = []

    for dim in range(tensor_layout.rank):
        if dim == 0:
            # Split query tiles across cores
            tensor_size = tensor_layout.get_shape(dim)
            tile_size = tile_layout.shape[dim]

            tensor_size_ssa = ensure_index_ssa(tensor_size)
            tile_size_ssa = builder.constant_index(tile_size)

            tile_size_minus_1 = builder.constant_index(tile_size - 1)
            numerator = arith.AddIOp(tensor_size_ssa, tile_size_minus_1).result
            num_tiles = arith.DivSIOp(numerator, tile_size_ssa).result

            # Even split (can be improved with load balancing)
            num_cores_ssa = ensure_index_ssa(num_cores)
            one = builder.constant_index(1)
            num_cores_minus_1 = arith.SubIOp(num_cores_ssa, one).result
            tiles_numerator = arith.AddIOp(num_tiles, num_cores_minus_1).result
            tiles_per_core = arith.DivSIOp(tiles_numerator, num_cores_ssa).result

            core_id_ssa = ensure_index_ssa(core_id)
            raw_start = arith.MulIOp(core_id_ssa, tiles_per_core).result
            start = _index_min(raw_start, num_tiles)

            core_id_plus_1 = arith.AddIOp(core_id_ssa, one).result
            raw_end = arith.MulIOp(core_id_plus_1, tiles_per_core).result
            end = _index_min(raw_end, num_tiles)

            ranges.append((ScalarValue(start), ScalarValue(end), 1))
        else:
            # Other dimensions: full range
            tensor_size = tensor_layout.get_shape(dim)
            tile_size = tile_layout.shape[dim]

            tensor_size_ssa = ensure_index_ssa(tensor_size)
            tile_size_ssa = builder.constant_index(tile_size)

            tile_size_minus_1 = builder.constant_index(tile_size - 1)
            numerator = arith.AddIOp(tensor_size_ssa, tile_size_minus_1).result
            num_tiles = arith.DivSIOp(numerator, tile_size_ssa).result

            ranges.append((0, ScalarValue(num_tiles), 1))

    return TiledView(tensor_layout, tile_layout, ranges)


def split_sequential(tensor_layout, tile_layout):
    """Split tensor into tiles without distribution.

    Returns a view that iterates over all tiles sequentially.
    Useful for single-core processing or inner loops.

    Parameters
    ----------
    tensor_layout : TensorLayout
        Layout of the tensor to split
    tile_layout : TileLayout
        Layout of each tile

    Returns
    -------
    TiledView
        View containing all tiles

    Example
    -------
    >>> # Inner loop over all K/V tiles
    >>> kv_layout = key.get_layout()
    >>> tile_layout = TileLayout(shape=(128, 64))
    >>> tiled = split_sequential(kv_layout, tile_layout)
    >>> with tiled.for_each() as coord:
    ...     # Process each tile
    """
    from ._ir_builder import get_builder
    builder = get_builder()

    assert tensor_layout.rank == tile_layout.rank

    ranges = []

    for dim in range(tensor_layout.rank):
        tensor_size = tensor_layout.get_shape(dim)
        tile_size = tile_layout.shape[dim]

        tensor_size_ssa = ensure_index_ssa(tensor_size)
        tile_size_ssa = builder.constant_index(tile_size)

        tile_size_minus_1 = builder.constant_index(tile_size - 1)
        numerator = arith.AddIOp(tensor_size_ssa, tile_size_minus_1).result
        num_tiles = arith.DivSIOp(numerator, tile_size_ssa).result

        ranges.append((0, ScalarValue(num_tiles), 1))

    return TiledView(tensor_layout, tile_layout, ranges)


def _index_min(a, b):
    """Emit arith.select(a < b, a, b)."""
    from mlir.dialects.arith import CmpIPredicate
    cmp = arith.CmpIOp(CmpIPredicate.slt, a, b).result
    return arith.SelectOp(cmp, a, b).result

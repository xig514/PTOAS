"""Utility functions for tensor splitting and distribution.

Provides various strategies for splitting tensors into tiles and
distributing them across cores.
"""

from ._layout_v2 import TensorLayout, TileLayout, TiledView, TileCoordinate
from ._scalar import ScalarValue
from ._utils import ensure_index_ssa
from mlir.dialects import arith, scf
from mlir.ir import InsertionPoint
from contextlib import contextmanager


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

    .. note:: Current implementation uses even split as a simplification.
       True causal balancing (triangular workload distribution) is not yet
       implemented.  The caller is responsible for masking out invalid
       positions inside the iteration loop (e.g. skip K/V tiles where
       kv_tile_idx > q_tile_idx).

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


class Tiled1DView:
    """Result of splitting a 1D scalar range across cores.

    Iterates over a contiguous sub-range [start, end) of integer indices.
    Unlike TiledView, this is not tied to a tensor layout — it simply
    represents a range of scalar indices (e.g., batch indices).

    Parameters
    ----------
    start : ScalarValue
        Start index (inclusive)
    end : ScalarValue
        End index (exclusive)
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def for_each(self):
        """Iterate over each index in [start, end).

        Yields
        ------
        idx : ScalarValue
            Current index
        """
        @contextmanager
        def _iterate():
            s = ensure_index_ssa(self.start)
            e = ensure_index_ssa(self.end)
            st = ensure_index_ssa(1)

            loop = scf.ForOp(s, e, st, [])
            ip = InsertionPoint(loop.body)
            ip.__enter__()
            try:
                idx = ScalarValue(loop.induction_variable)
                yield idx
                scf.YieldOp([])
            finally:
                ip.__exit__(None, None, None)

        return _iterate()


def split_even_1d(count, num_cores, core_id):
    """Split a 1D range [0, count) evenly across cores.

    Returns a Tiled1DView representing the sub-range assigned to core_id.
    Useful for distributing batch indices or token ranges across cores.

    Parameters
    ----------
    count : int or ScalarValue
        Total number of items
    num_cores : int or ScalarValue
        Total number of cores
    core_id : int or ScalarValue
        ID of current core (0-indexed)

    Returns
    -------
    Tiled1DView
        View containing this core's assigned range [start, end)
    """
    from ._ir_builder import get_builder
    builder = get_builder()

    count_ssa = ensure_index_ssa(count)
    num_cores_ssa = ensure_index_ssa(num_cores)
    core_id_ssa = ensure_index_ssa(core_id)

    # items_per_core = ceildiv(count, num_cores)
    one = builder.constant_index(1)
    num_cores_minus_1 = arith.SubIOp(num_cores_ssa, one).result
    numerator = arith.AddIOp(count_ssa, num_cores_minus_1).result
    items_per_core = arith.DivSIOp(numerator, num_cores_ssa).result

    # start = min(core_id * items_per_core, count)
    raw_start = arith.MulIOp(core_id_ssa, items_per_core).result
    start = _index_min(raw_start, count_ssa)

    # end = min((core_id + 1) * items_per_core, count)
    core_id_plus_1 = arith.AddIOp(core_id_ssa, one).result
    raw_end = arith.MulIOp(core_id_plus_1, items_per_core).result
    end = _index_min(raw_end, count_ssa)

    return Tiled1DView(ScalarValue(start), ScalarValue(end))


def _index_min(a, b):
    """Emit arith.select(a < b, a, b)."""
    from mlir.dialects.arith import CmpIPredicate
    cmp = arith.CmpIOp(CmpIPredicate.slt, a, b).result
    return arith.SelectOp(cmp, a, b).result

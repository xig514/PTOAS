"""Enhanced Layout system for tensor tiling and distribution.

Core concepts:
1. TensorLayout: Describes tensor's shape and stride (replaces simple shape)
2. TileLayout: Describes tile shape and stride
3. TiledView: Result of splitting, contains tile coordinate ranges
4. Coordinate-based access: Use tile coordinates to access data

Example:
    tensor_layout = tensor.get_layout()  # TensorLayout with shape + stride
    tile_layout = TileLayout(shape=(64, 128))  # Tile pattern

    # Split tensor into tiles, assign to cores
    tiled = split_even(tensor_layout, tile_layout, num_cores, core_id)

    # Iterate over assigned tiles
    with tiled.for_each() as coord:  # coord is (tile_i, tile_j)
        tile_buf = make_tile((64, 128), ...)
        tload_tile(tensor, coord, tile_buf)
"""

from mlir.dialects import arith, scf
from mlir.ir import InsertionPoint
from ._scalar import ScalarValue
from ._utils import ensure_index_ssa


class TensorLayout:
    """Layout of a tensor: shape + stride.

    Represents the memory layout of a tensor, including both logical shape
    and physical stride pattern.

    Parameters
    ----------
    shape : tuple of int or ScalarValue
        Logical shape of each dimension (can be dynamic)
    stride : tuple of int, optional
        Stride for each dimension. Defaults to row-major compact.
    """

    def __init__(self, shape, stride=None):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)

        if stride is None:
            # Compute row-major compact stride
            self.stride = self._compact_stride_static(self.shape)
        else:
            if isinstance(stride, int):
                stride = (stride,)
            self.stride = tuple(stride)

        assert len(self.shape) == len(self.stride)

    @property
    def rank(self):
        """Number of dimensions."""
        return len(self.shape)

    @staticmethod
    def _compact_stride_static(shape):
        """Compute row-major compact stride (static version)."""
        if not shape:
            return ()
        # For dynamic shapes, we can't compute static strides
        # Return symbolic strides based on shape
        stride = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            if isinstance(shape[i + 1], int):
                stride[i] = stride[i + 1] * shape[i + 1]
            else:
                # Dynamic: stride[i] = stride[i+1] * shape[i+1]
                # We'll compute this at runtime when needed
                stride[i] = None  # Mark as dynamic
        return tuple(stride)

    def get_shape(self, dim):
        """Get shape of dimension (returns ScalarValue if dynamic)."""
        s = self.shape[dim]
        if isinstance(s, int):
            return s
        elif isinstance(s, ScalarValue):
            return s
        else:
            # It's an SSA value
            return ScalarValue(s)

    def get_stride(self, dim):
        """Get stride of dimension."""
        return self.stride[dim]

    def __repr__(self):
        return f"TensorLayout(shape={self.shape}, stride={self.stride})"


class TileLayout:
    """Layout of a tile: shape + stride.

    Describes the tiling pattern to apply to a tensor.

    Parameters
    ----------
    shape : tuple of int
        Tile size for each dimension (must be static)
    stride : tuple of int, optional
        Stride for each dimension. Defaults to row-major compact.
    """

    def __init__(self, shape, stride=None):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)

        # Tile shape must be static
        assert all(isinstance(s, int) for s in self.shape), \
            "TileLayout shape must be static integers"

        if stride is None:
            self.stride = self._compact_stride(self.shape)
        else:
            if isinstance(stride, int):
                stride = (stride,)
            self.stride = tuple(stride)

        assert len(self.shape) == len(self.stride)

    @property
    def rank(self):
        """Number of dimensions."""
        return len(self.shape)

    @staticmethod
    def _compact_stride(shape):
        """Compute row-major compact stride."""
        if not shape:
            return ()
        stride = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            stride[i] = stride[i + 1] * shape[i + 1]
        return tuple(stride)

    def __repr__(self):
        return f"TileLayout(shape={self.shape}, stride={self.stride})"


class PartialCoord:
    """A subset of coordinate dimensions selected from a TileCoordinate.

    Created by ``TileCoordinate.select(*dims)`` and used with
    ``coord_combine()`` to compose new coordinates from multiple sources.

    Parameters
    ----------
    values : tuple of int or ScalarValue
        The selected coordinate values
    source_dims : tuple of int
        Original dimension indices these values came from
    """

    def __init__(self, values, source_dims=None):
        self.values = tuple(values)
        self.source_dims = tuple(source_dims) if source_dims else ()

    @property
    def rank(self):
        return len(self.values)

    def __repr__(self):
        return f"PartialCoord(values={self.values}, dims={self.source_dims})"


class TileCoordinate:
    """Coordinate of a tile in the tile grid.

    Represents a multi-dimensional tile index, where each dimension
    can be static (int) or dynamic (ScalarValue).
    """

    def __init__(self, coords):
        """
        Parameters
        ----------
        coords : tuple of int or ScalarValue
            Coordinate for each dimension
        """
        self.coords = tuple(coords)

    @property
    def rank(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx]

    def __iter__(self):
        return iter(self.coords)

    def select(self, *dims):
        """Select specific dimensions from this coordinate.

        Parameters
        ----------
        *dims : int
            Dimension indices to select

        Returns
        -------
        PartialCoord
            A partial coordinate containing only the selected dimensions

        Example
        -------
        >>> coord = TileCoordinate((b, n, sq, d))
        >>> coord.select(0, 1, 2)  # -> PartialCoord with (b, n, sq)
        """
        values = [self.coords[d] for d in dims]
        return PartialCoord(values, dims)

    def offset_by(self, dim, offset):
        """Create a new coordinate with an offset added to one dimension.

        Parameters
        ----------
        dim : int
            Dimension index to offset
        offset : int or ScalarValue
            Offset to add (in tile units)

        Returns
        -------
        TileCoordinate
            New coordinate with the offset applied

        Example
        -------
        >>> local_coord = TileCoordinate((sq_local, d))
        >>> global_coord = local_coord.offset_by(dim=0, offset=q_start)
        """
        new_coords = list(self.coords)
        orig = new_coords[dim]
        if isinstance(orig, ScalarValue):
            new_coords[dim] = orig + offset
        elif isinstance(offset, ScalarValue):
            new_coords[dim] = offset + orig
        else:
            new_coords[dim] = orig + offset
        return TileCoordinate(tuple(new_coords))

    def __repr__(self):
        return f"TileCoordinate{self.coords}"


def coord_combine(*parts):
    """Combine multiple partial/full coordinates into a new TileCoordinate.

    Accepts any mix of ``PartialCoord``, ``TileCoordinate``, or raw
    ``ScalarValue`` / ``int`` values, and concatenates them in order.

    Parameters
    ----------
    *parts : PartialCoord, TileCoordinate, ScalarValue, or int
        Coordinate fragments to concatenate

    Returns
    -------
    TileCoordinate
        New coordinate with all values concatenated

    Example
    -------
    >>> pse_coord = coord_combine(
    ...     q_coord.select(0, 1, 2),   # [B, N, Sq] from Q
    ...     k_coord.select(2),          # [Sk] from K
    ... )
    """
    all_values = []
    for p in parts:
        if isinstance(p, PartialCoord):
            all_values.extend(p.values)
        elif isinstance(p, TileCoordinate):
            all_values.extend(p.coords)
        else:
            # Raw scalar or int
            all_values.append(p)
    return TileCoordinate(tuple(all_values))


class TiledView:
    """Result of splitting a tensor into tiles.

    Contains information about tile ranges for each dimension,
    allowing iteration over assigned tiles.

    Parameters
    ----------
    tensor_layout : TensorLayout
        Layout of the source tensor
    tile_layout : TileLayout
        Layout of each tile
    ranges : list of tuple (start, end, step)
        Range for each dimension (can contain ScalarValue)
    """

    def __init__(self, tensor_layout, tile_layout, ranges):
        self.tensor_layout = tensor_layout
        self.tile_layout = tile_layout
        self.ranges = ranges

        assert tensor_layout.rank == tile_layout.rank
        assert len(ranges) == tile_layout.rank

    def for_each(self):
        """Iterate over all tiles in the assigned range.

        Yields
        ------
        coord : TileCoordinate
            Coordinate of current tile
        """
        from contextlib import contextmanager

        @contextmanager
        def _iterate():
            yield from self._build_nested_loops(0, [])

        return _iterate()

    def _build_nested_loops(self, depth, indices):
        """Recursively build nested loops for tile iteration."""
        if depth == len(self.ranges):
            # Base case: yield coordinate
            coord = TileCoordinate(tuple(indices))
            yield coord
            return

        start, end, step = self.ranges[depth]
        s = ensure_index_ssa(start)
        e = ensure_index_ssa(end)
        st = ensure_index_ssa(step)

        loop = scf.ForOp(s, e, st, [])
        ip = InsertionPoint(loop.body)
        ip.__enter__()
        try:
            idx = ScalarValue(loop.induction_variable)
            yield from self._build_nested_loops(depth + 1, indices + [idx])
            scf.YieldOp([])
        finally:
            ip.__exit__(None, None, None)

    def compute_offset(self, coord):
        """Compute byte offset for a tile coordinate.

        Parameters
        ----------
        coord : TileCoordinate
            Tile coordinate

        Returns
        -------
        offset : tuple of ScalarValue
            Offset for each dimension in elements
        """
        from ._ir_builder import get_builder
        builder = get_builder()

        offsets = []
        for i, tile_idx in enumerate(coord):
            tile_size = self.tile_layout.shape[i]
            tile_idx_ssa = ensure_index_ssa(tile_idx)
            tile_size_ssa = builder.constant_index(tile_size)
            offset = arith.MulIOp(tile_idx_ssa, tile_size_ssa).result
            offsets.append(ScalarValue(offset))

        return tuple(offsets)

    def __repr__(self):
        return f"TiledView(ranges={self.ranges})"

"""Tensor proxy: type annotations, runtime proxies, and slicing."""


class _TensorSpec:
    """Annotation object returned by ``Tensor(dtype, ndim)``."""

    def __init__(self, dtype, ndim):
        self.dtype = dtype
        self.ndim = ndim


class _TensorShapeSpec:
    """Annotation object returned by ``Tensor[[M, N], dtype]``.

    *shape* is a list of ``int`` (static) or :class:`DynVar` (dynamic).
    """

    def __init__(self, shape, dtype):
        self.shape = list(shape)
        self.dtype = dtype
        self.ndim = len(shape)


class Tensor:
    """Tensor type annotation factory.

    Two forms are supported::

        # Legacy: specify dtype and rank only
        Tensor(pto.float16, 2)          -> _TensorSpec

        # New: specify shape (int or DynVar) and dtype
        Tensor[[M, N], pto.float16]     -> _TensorShapeSpec
    """

    def __new__(cls, dtype, ndim):
        """Legacy call syntax: ``Tensor(dtype, ndim)``."""
        return _TensorSpec(dtype, ndim)

    def __class_getitem__(cls, item):
        """Subscript syntax: ``Tensor[[M, N], dtype]``."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                "Tensor[...] requires exactly two arguments: "
                "Tensor[[shape...], dtype]"
            )
        shape, dtype = item
        if not isinstance(shape, (list, tuple)):
            raise TypeError(
                f"Tensor shape must be a list or tuple, got {type(shape).__name__}"
            )
        return _TensorShapeSpec(shape, dtype)


class _PartitionView:
    """Wraps a ``!pto.partition_tensor_view`` SSA value."""

    __slots__ = ("ssa", "shape", "dtype")

    def __init__(self, ssa, shape, dtype):
        self.ssa = ssa
        self.shape = shape  # tuple of ints (static sizes)
        self.dtype = dtype


class _ShapeAccessor:
    """``tensor.shape[i]`` -> ScalarValue wrapping the i-th dim SSA."""

    def __init__(self, shape_ssas):
        self._ssas = shape_ssas

    def __getitem__(self, idx):
        from ._scalar import ScalarValue

        return ScalarValue(self._ssas[idx])


class _TensorProxy:
    """Runtime proxy for a Tensor parameter during kernel tracing."""

    def __init__(self, ptr_ssa, shape_ssas, dtype, ndim):
        self.ptr_ssa = ptr_ssa
        self._shape_ssas = shape_ssas
        self.dtype = dtype
        self.ndim = ndim
        self.shape = _ShapeAccessor(shape_ssas)
        self._layout = None  # Lazy initialization

    def get_layout(self):
        """Get the TensorLayout of this tensor.

        Returns
        -------
        TensorLayout
            Layout containing shape and stride information
        """
        if self._layout is None:
            from ._layout_v2 import TensorLayout
            # Create layout with dynamic shape and row-major stride
            self._layout = TensorLayout(
                shape=tuple(self._shape_ssas),
                stride=None  # Will compute row-major
            )
        return self._layout

    # -- internal: create a fresh tensor view at the current IP --

    def _make_tensor_view(self):
        from mlir.dialects import pto as _pto
        from ._utils import compute_row_major_strides

        tv_type = _pto.TensorViewType.get(self.ndim, self.dtype.to_mlir())
        strides = compute_row_major_strides(self._shape_ssas)
        return _pto.MakeTensorViewOp(
            tv_type, self.ptr_ssa, list(self._shape_ssas), strides
        ).result

    # -- slicing: x[0:32, 0:32] -> _PartitionView --

    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)

        from ._utils import ensure_index_ssa
        from mlir.dialects import pto as _pto
        from mlir.dialects import arith as _arith

        offsets = []
        sizes = []
        static_sizes = []

        for s in slices:
            if not isinstance(s, slice):
                raise TypeError(f"Expected slice, got {type(s).__name__}")
            start = s.start if s.start is not None else 0
            stop = s.stop

            offsets.append(ensure_index_ssa(start))

            if isinstance(start, int) and isinstance(stop, int):
                sz = stop - start
                static_sizes.append(sz)
                sizes.append(ensure_index_ssa(sz))
            else:
                start_ssa = ensure_index_ssa(start)
                stop_ssa = ensure_index_ssa(stop)
                size_ssa = _arith.SubIOp(stop_ssa, start_ssa).result
                sizes.append(size_ssa)
                static_sizes.append(-1)

        tv = self._make_tensor_view()
        pv_type = _pto.PartitionTensorViewType.get(
            static_sizes, self.dtype.to_mlir()
        )
        pv = _pto.PartitionViewOp(
            pv_type, tv, offsets=offsets, sizes=sizes
        ).result
        return _PartitionView(pv, tuple(static_sizes), self.dtype)

    # -- tiling: split one dimension into fixed-size tiles --

    def tile(self, dim, tile_sizes=None, size=None):
        """Tile the tensor along *dim*.

        Parameters
        ----------
        dim : int
            The dimension to iterate over.
        tile_sizes : tuple[int, ...], optional
            Static tile size for **every** dimension.  Produces fully-static
            partition views (required when the result feeds ``tload``).
        size : int, optional
            Tile size for the iterated dimension only.  Other dimensions
            use the full dynamic extent (``-1`` in the partition type).
            Mutually exclusive with *tile_sizes*.

        Returns
        -------
        TiledTensor
        """
        from ._tiled_tensor import TiledTensor

        if tile_sizes is not None and size is not None:
            raise ValueError("Provide either tile_sizes or size, not both")
        if tile_sizes is not None:
            return TiledTensor(self, dim, tuple(tile_sizes))
        if size is not None:
            ts = [None] * self.ndim
            ts[dim] = size
            return TiledTensor(self, dim, tuple(ts))
        raise ValueError("Provide either tile_sizes or size")

    def tile_nd(self, tile_sizes, tile_dims):
        """Tile the tensor along multiple dimensions.

        Parameters
        ----------
        tile_sizes : tuple[int, ...]
            Static tile size for **every** dimension. Use None for dimensions
            that should not be tiled.
        tile_dims : list[int]
            List of dimension indices to tile. Must match non-None entries
            in tile_sizes.

        Returns
        -------
        TiledTensorND
        """
        from ._tiled_tensor_nd import TiledTensorND

        return TiledTensorND(self, tile_sizes, tile_dims)

    def partition_at_coord(self, tile_coord, tile_layout):
        """Create a partition view at the given tile coordinate.

        Parameters
        ----------
        tile_coord : TileCoordinate
            Coordinate of the tile
        tile_layout : TileLayout
            Layout of the tile

        Returns
        -------
        _PartitionView
            Partition view for the tile
        """
        from ._utils import ensure_index_ssa
        from mlir.dialects import pto as _pto, arith
        from ._ir_builder import get_builder

        builder = get_builder()

        offsets = []
        sizes = []
        static_sizes = []

        for i in range(self.ndim):
            tile_idx = tile_coord[i]
            tile_size = tile_layout.shape[i]

            idx_ssa = ensure_index_ssa(tile_idx)
            tile_size_ssa = builder.constant_index(tile_size)
            offset = arith.MulIOp(idx_ssa, tile_size_ssa).result

            offsets.append(offset)
            sizes.append(tile_size_ssa)
            static_sizes.append(tile_size)

        tv = self._make_tensor_view()
        pv_type = _pto.PartitionTensorViewType.get(
            static_sizes, self.dtype.to_mlir()
        )
        pv = _pto.PartitionViewOp(
            pv_type, tv, offsets=offsets, sizes=sizes
        ).result
        return _PartitionView(pv, tuple(static_sizes), self.dtype)

    # -- explicit partition with known static sizes --

    def partition(self, offsets, sizes):
        """Create a partition view with explicit offsets and sizes.

        ``offsets`` may be ints or ScalarValues (dynamic).
        ``sizes`` may be ints (static) or ScalarValues (dynamic → -1 in type).
        """
        from ._utils import ensure_index_ssa
        from mlir.dialects import pto as _pto
        from ._scalar import ScalarValue

        off_ssas = [ensure_index_ssa(o) for o in offsets]
        sz_ssas = [ensure_index_ssa(s) for s in sizes]
        # For static sizes use the int value; for dynamic (ScalarValue) use -1
        static_sizes = [
            -1 if isinstance(s, ScalarValue) else s
            for s in sizes
        ]

        tv = self._make_tensor_view()
        pv_type = _pto.PartitionTensorViewType.get(
            static_sizes, self.dtype.to_mlir()
        )
        pv = _pto.PartitionViewOp(
            pv_type, tv, offsets=off_ssas, sizes=sz_ssas
        ).result
        return _PartitionView(pv, tuple(static_sizes), self.dtype)

"""Tensor proxy: type annotations, runtime proxies, and slicing."""


class _TensorSpec:
    """Annotation object returned by ``Tensor(dtype, ndim)``."""

    def __init__(self, dtype, ndim):
        self.dtype = dtype
        self.ndim = ndim


def Tensor(dtype, ndim):
    """Create a Tensor type annotation for a kernel parameter.

    Usage::

        @pto.kernel
        def my_kernel(x: pto.Tensor(pto.float16, 2), ...):
            ...
    """
    return _TensorSpec(dtype, ndim)


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

    # -- explicit partition with known static sizes --

    def partition(self, offsets, sizes):
        """Create a partition view with explicit offsets and static sizes.

        ``offsets`` may be ints or ScalarValues (dynamic).
        ``sizes`` must be ints (static).
        """
        from ._utils import ensure_index_ssa
        from mlir.dialects import pto as _pto

        off_ssas = [ensure_index_ssa(o) for o in offsets]
        sz_ssas = [ensure_index_ssa(s) for s in sizes]
        static_sizes = list(sizes)

        tv = self._make_tensor_view()
        pv_type = _pto.PartitionTensorViewType.get(
            static_sizes, self.dtype.to_mlir()
        )
        pv = _pto.PartitionViewOp(
            pv_type, tv, offsets=off_ssas, sizes=sz_ssas
        ).result
        return _PartitionView(pv, tuple(static_sizes), self.dtype)

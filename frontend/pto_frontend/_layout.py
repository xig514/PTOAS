"""TileLayout abstraction for multi-dimensional tiling and distribution.

Inspired by CuTe's Layout concept: a (Shape, Stride) pair that maps
logical coordinates to linear offsets.

Usage::

    # 2D tiling
    layout = TileLayout((8, 16), (1, 8))  # 8x16 tiles, row-major
    offset = layout(3, 5)  # = 3 + 5*8 = 43

    # Composition
    tile_layout = TileLayout((64, 32), (1, 64))
    dist_layout = TileLayout((4, 16), (1, 4))
    full = tile_layout.compose(dist_layout)

    # Slicing
    my_tiles = layout.slice(dim=1, idx=core_id)
"""

from mlir.dialects import arith
from ._scalar import ScalarValue
from ._utils import ensure_index_ssa


class TileLayout:
    """CuTe-style TileLayout: (Shape, Stride) mapping.

    Maps a multi-dimensional logical coordinate to a linear offset:
        offset = sum(coord[i] * stride[i] for i in range(rank))

    Parameters
    ----------
    shape : tuple[int, ...]
        Logical shape of each dimension.
    stride : tuple[int, ...], optional
        Stride for each dimension. Defaults to row-major compact.
    """

    def __init__(self, shape, stride=None):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)

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

    @property
    def size(self):
        """Total number of elements (product of shape)."""
        result = 1
        for s in self.shape:
            result *= s
        return result

    @staticmethod
    def _compact_stride(shape):
        """Compute row-major compact stride."""
        if not shape:
            return ()
        stride = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            stride[i] = stride[i + 1] * shape[i + 1]
        return tuple(stride)

    def __call__(self, *coords):
        """Map logical coordinates to linear offset.

        Accepts either separate args or a single tuple/list.
        Returns a ScalarValue (index type) if any coord is dynamic,
        otherwise returns an int.
        """
        if len(coords) == 1 and isinstance(coords[0], (tuple, list)):
            coords = coords[0]

        assert len(coords) == self.rank

        # Check if all coords are static ints
        all_static = all(isinstance(c, int) for c in coords)

        if all_static:
            return sum(c * s for c, s in zip(coords, self.stride))

        # Dynamic: emit arith ops
        from ._ir_builder import get_builder
        builder = get_builder()

        offset = builder.constant_index(0)
        for coord, stride_val in zip(coords, self.stride):
            coord_ssa = ensure_index_ssa(coord)
            stride_ssa = builder.constant_index(stride_val)
            term = arith.MulIOp(coord_ssa, stride_ssa).result
            offset = arith.AddIOp(offset, term).result

        return ScalarValue(offset)

    def slice(self, dim, idx):
        """Fix dimension *dim* to *idx*, returning a lower-rank TileLayout.

        Parameters
        ----------
        dim : int
            Dimension to fix.
        idx : int or ScalarValue
            Index value (can be dynamic).

        Returns
        -------
        SlicedTileLayout
            A TileLayout with rank reduced by 1, plus a base offset.
        """
        new_shape = self.shape[:dim] + self.shape[dim + 1:]
        new_stride = self.stride[:dim] + self.stride[dim + 1:]

        if isinstance(idx, int):
            offset = idx * self.stride[dim]
        else:
            from ._ir_builder import get_builder
            idx_ssa = ensure_index_ssa(idx)
            stride_ssa = get_builder().constant_index(self.stride[dim])
            offset = ScalarValue(arith.MulIOp(idx_ssa, stride_ssa).result)

        return SlicedTileLayout(new_shape, new_stride, offset)

    def compose(self, inner):
        """Compose two TileLayouts: self(inner(coord)).

        Returns a ComposedTileLayout that applies *inner* first, then *self*.
        """
        return ComposedTileLayout(self, inner)

    def __repr__(self):
        return f"TileLayout({self.shape}, {self.stride})"


class SlicedTileLayout(TileLayout):
    """A TileLayout with a base offset (from slicing)."""

    def __init__(self, shape, stride, offset):
        super().__init__(shape, stride)
        self.offset = offset

    def __call__(self, *coords):
        base = super().__call__(*coords)
        if isinstance(self.offset, int) and isinstance(base, int):
            return self.offset + base
        # Dynamic
        offset_ssa = ensure_index_ssa(self.offset)
        base_ssa = ensure_index_ssa(base)
        return ScalarValue(arith.AddIOp(offset_ssa, base_ssa).result)

    def __repr__(self):
        return f"SlicedTileLayout({self.shape}, {self.stride}, offset={self.offset})"


class ComposedTileLayout:
    """Composition of two TileLayouts: outer(inner(coord))."""

    def __init__(self, outer, inner):
        self.outer = outer
        self.inner = inner
        self.shape = inner.shape

    @property
    def rank(self):
        return self.inner.rank

    @property
    def size(self):
        return self.inner.size

    def __call__(self, *coords):
        mid = self.inner(*coords)
        return self.outer(mid)

    def __repr__(self):
        return f"Compose({self.outer}, {self.inner})"

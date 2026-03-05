"""Multi-dimensional tiling with Layout-based distribution.

Extends TiledTensor to support N-D tiling and distribution across
multi-dimensional core grids.

Usage::

    # 3D tiling: Query BSND [B, S, N, D]
    q_tiled = query.tile_nd(
        tile_sizes=(1, 128, 1, 64),  # tile each dimension
        tile_dims=[0, 1, 2]          # iterate over B, S, N
    )

    # Distribute 3D tiles to 1D cores
    q_dist = q_tiled.distribute_nd(core_grid=(B_cores, S_cores, N_cores))

    with q_dist.for_each() as (tile_idx, q_view):
        # tile_idx is (b, s, n) tuple
        ...
"""

from contextlib import contextmanager

from mlir.ir import InsertionPoint, IndexType
from mlir.dialects import scf, arith

from ._scalar import ScalarValue
from ._tensor import _PartitionView
from ._utils import ensure_index_ssa


class TiledTensorND:
    """N-dimensional tiling of a tensor.

    Parameters
    ----------
    tensor_proxy : _TensorProxy
        The underlying tensor.
    tile_sizes : tuple[int | None, ...]
        Tile size for each tensor dimension. ``None`` means full extent.
    tile_dims : list[int]
        Dimensions to iterate over (others are fixed at 0).
    """

    def __init__(self, tensor_proxy, tile_sizes, tile_dims):
        self._tensor = tensor_proxy
        self._tile_sizes = tuple(tile_sizes)
        self._tile_dims = list(tile_dims)

        assert len(tile_sizes) == tensor_proxy.ndim
        for d in tile_dims:
            assert tile_sizes[d] is not None, f"tile_dims[{d}] must have a static size"

    @property
    def tile_shape(self):
        """Shape of the tile grid (only for tile_dims).

        Returns a tuple of ScalarValues (ceildiv for each tile_dim).
        """
        result = []
        for d in self._tile_dims:
            dim_size = self._tensor.shape[d]
            ts = self._tile_sizes[d]
            result.append((dim_size + (ts - 1)) // ts)
        return tuple(result)

    def _make_partition(self, tile_idx):
        """Build a partition_view for the tile at *tile_idx*.

        Parameters
        ----------
        tile_idx : tuple
            Indices for each dimension in tile_dims (can be int or ScalarValue).

        Returns
        -------
        _PartitionView
        """
        from mlir.dialects import pto as _pto
        from ._ir_builder import get_builder

        builder = get_builder()

        # tile_idx is a tuple with len(tile_dims) elements
        assert len(tile_idx) == len(self._tile_dims)

        offsets, sizes, static_sizes = [], [], []

        tile_idx_map = dict(zip(self._tile_dims, tile_idx))

        for d in range(self._tensor.ndim):
            if d in self._tile_dims:
                # Tiled dimension
                idx = tile_idx_map[d]
                ts = self._tile_sizes[d]

                idx_ssa = ensure_index_ssa(idx)
                ts_ssa = builder.constant_index(ts)
                offset = arith.MulIOp(idx_ssa, ts_ssa).result

                offsets.append(offset)
                sizes.append(ts_ssa)
                static_sizes.append(ts)
            else:
                # Non-tiled dimension
                offsets.append(builder.constant_index(0))
                ts = self._tile_sizes[d]
                if ts is not None:
                    sizes.append(builder.constant_index(ts))
                    static_sizes.append(ts)
                else:
                    sizes.append(self._tensor._shape_ssas[d])
                    static_sizes.append(-1)

        tv = self._tensor._make_tensor_view()
        pv_type = _pto.PartitionTensorViewType.get(
            static_sizes, self._tensor.dtype.to_mlir()
        )
        pv = _pto.PartitionViewOp(
            pv_type, tv, offsets=offsets, sizes=sizes
        ).result
        return _PartitionView(pv, tuple(static_sizes), self._tensor.dtype)

    def __getitem__(self, idx):
        """Get partition for tile at *idx*.

        *idx* can be a single int (for 1D tiling) or a tuple (for N-D).
        """
        if not isinstance(idx, tuple):
            idx = (idx,)
        return self._make_partition(idx)

    @contextmanager
    def for_each(self, ranges=None):
        """Iterate over all tiles in the grid.

        Parameters
        ----------
        ranges : list[tuple[start, end, step]], optional
            Per-dimension ranges. Defaults to full range for each tile_dim.

        Yields
        ------
        tile_idx : tuple of ScalarValue
            N-D tile index.
        partition_view : _PartitionView
        """
        if ranges is None:
            ranges = [(0, ts, 1) for ts in self.tile_shape]

        assert len(ranges) == len(self._tile_dims)

        # Use recursive helper to build nested loops
        yield from self._build_nested_loops(ranges, 0, [])

    def _build_nested_loops(self, ranges, depth, indices):
        """Recursively build nested scf.for loops."""
        if depth == len(ranges):
            # Base case: at innermost level, yield the partition
            pv = self._make_partition(tuple(indices))
            yield tuple(indices), pv
            return

        # Recursive case: create one loop level
        start, end, step = ranges[depth]
        s = ensure_index_ssa(start)
        e = ensure_index_ssa(end)
        st = ensure_index_ssa(step)

        loop = scf.ForOp(s, e, st, [])
        ip = InsertionPoint(loop.body)
        ip.__enter__()
        try:
            idx = ScalarValue(loop.induction_variable)
            # Recursively build inner loops
            yield from self._build_nested_loops(ranges, depth + 1, indices + [idx])
            # Emit yield for this loop level
            scf.YieldOp([])
        finally:
            ip.__exit__(None, None, None)

    def distribute_nd(self, core_grid):
        """Distribute tiles across a multi-dimensional core grid.

        Parameters
        ----------
        core_grid : tuple[int, ...]
            Number of cores along each tile dimension.

        Returns
        -------
        DistributedTiledTensorND
        """
        return DistributedTiledTensorND(self, core_grid)


class DistributedTiledTensorND:
    """N-D tiles distributed across an N-D core grid.

    Uses TileLayout to map tile indices to core indices.
    """

    def __init__(self, tiled_tensor, core_grid):
        self._tiled = tiled_tensor
        self._core_grid = tuple(core_grid)

        assert len(core_grid) == len(tiled_tensor._tile_dims)

    @contextmanager
    def for_each(self):
        """Iterate over this core's tiles.

        Yields
        ------
        tile_idx : tuple of ScalarValue
            Global N-D tile index.
        partition_view : _PartitionView
        """
        from mlir.dialects import pto as _pto

        # Get linear core_id
        idx_ty = IndexType.get()
        core_id = arith.IndexCastOp(
            idx_ty, _pto.GetBlockIdxOp().result
        ).result
        num_cores = arith.IndexCastOp(
            idx_ty, _pto.GetBlockNumOp().result
        ).result

        # Compute tile_shape (static if possible)
        tile_shape = []
        for ts in self._tiled.tile_shape:
            if isinstance(ts, int):
                tile_shape.append(ts)
            else:
                # Dynamic: evaluate at runtime
                tile_shape.append(ensure_index_ssa(ts))

        # Build Layout: tile_shape → core_grid
        # Use row-major mapping: tile_idx = (t0, t1, ...) → linear
        # Then map linear → core_grid → core_id

        # For simplicity, assume tiles_per_core = ceildiv(tile_shape, core_grid)
        # and use blocked distribution along each dimension.

        ranges = []
        for dim_idx, (num_tiles, num_cores_dim) in enumerate(
            zip(tile_shape, self._core_grid)
        ):
            # Compute this core's range for this dimension
            # core_coord[dim] = (core_id % prod(core_grid[:dim+1])) // prod(core_grid[:dim])

            # For now, use a simple 1D linearization:
            # Assume core_grid is (C0, C1, ...) and we map linearly.
            # This is a simplification; full N-D mapping needs more work.

            # Simplified: treat as 1D distribution for now
            # TODO: proper N-D core coordinate extraction

            from ._ir_builder import get_builder
            builder = get_builder()

            num_tiles_ssa = ensure_index_ssa(num_tiles)
            num_cores_ssa = builder.constant_index(num_cores_dim)

            one = builder.constant_index(1)
            tpc = arith.DivSIOp(
                arith.AddIOp(
                    num_tiles_ssa,
                    arith.SubIOp(num_cores_ssa, one).result
                ).result,
                num_cores_ssa,
            ).result

            # Extract core coordinate for this dimension
            # For 1D: core_coord = core_id
            # For N-D: need to decompose core_id
            if len(self._core_grid) == 1:
                core_coord = core_id
            else:
                # Multi-dimensional: decompose core_id
                # core_coord[i] = (core_id // stride[i]) % core_grid[i]
                stride = 1
                for j in range(dim_idx):
                    stride *= self._core_grid[j]

                stride_ssa = builder.constant_index(stride)
                cg_ssa = builder.constant_index(num_cores_dim)

                core_coord = arith.RemSIOp(
                    arith.DivSIOp(core_id, stride_ssa).result,
                    cg_ssa
                ).result

            raw_start = arith.MulIOp(core_coord, tpc).result
            start = self._index_min(raw_start, num_tiles_ssa)

            raw_end = arith.MulIOp(
                arith.AddIOp(core_coord, one).result, tpc
            ).result
            end = self._index_min(raw_end, num_tiles_ssa)

            ranges.append((ScalarValue(start), ScalarValue(end), 1))

        # Delegate to TiledTensorND.for_each with computed ranges
        with self._tiled.for_each(ranges=ranges) as result:
            yield result

    @staticmethod
    def _index_min(a, b):
        """Emit arith.select(a < b, a, b)."""
        from mlir.dialects.arith import CmpIPredicate
        cmp = arith.CmpIOp(CmpIPredicate.slt, a, b).result
        return arith.SelectOp(cmp, a, b).result

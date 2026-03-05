"""TiledTensor and DistributedTiledTensor for multi-core tile distribution.

TiledTensor logically divides a tensor along one dimension into fixed-size
tiles and provides iteration helpers.  DistributedTiledTensor adds automatic
even-split across cores via ``get_block_idx`` / ``get_block_num``.

Usage::

    q_tiled = query.tile(dim=0, tile_sizes=(128, 128))

    # iterate all tiles
    with q_tiled.for_each() as (i, q_view):
        ...

    # even multi-core split
    with q_tiled.distribute().for_each() as (i, q_view):
        ...
"""

from contextlib import contextmanager

from mlir.ir import InsertionPoint, IndexType
from mlir.dialects import scf, arith
from mlir.dialects.arith import CmpIPredicate

from ._scalar import ScalarValue
from ._tensor import _PartitionView
from ._utils import ensure_index_ssa


def _index_min(a, b):
    """Emit ``arith.select(a < b, a, b)`` on index-typed SSA values."""
    cmp = arith.CmpIOp(CmpIPredicate.slt, a, b).result
    return arith.SelectOp(cmp, a, b).result


class TiledTensor:
    """A tensor with one dimension logically divided into fixed-size tiles.

    Created via :meth:`_TensorProxy.tile`.

    Parameters
    ----------
    tensor_proxy : _TensorProxy
        The underlying tensor proxy.
    dim : int
        The dimension to iterate over.
    tile_sizes : tuple[int | None, ...]
        Per-dimension tile sizes.  An ``int`` gives a static partition size;
        ``None`` means "use the full dynamic extent of that dimension".
    """

    def __init__(self, tensor_proxy, dim, tile_sizes):
        self._tensor = tensor_proxy
        self._dim = dim
        self._tile_sizes = tile_sizes
        assert len(tile_sizes) == tensor_proxy.ndim
        assert tile_sizes[dim] is not None, "tiled dimension must have a static size"

    @property
    def tile_size(self):
        """Tile size along the iterated dimension (int)."""
        return self._tile_sizes[self._dim]

    @property
    def num_tiles(self):
        """Total tile count as a ScalarValue (ceildiv, index type).

        .. note:: Each access emits new IR ops; cache the result if needed.
        """
        dim_size = self._tensor.shape[self._dim]
        ts = self._tile_sizes[self._dim]
        return (dim_size + (ts - 1)) // ts

    # -- internal helpers ------------------------------------------------

    def _make_partition(self, tile_idx):
        """Build a ``_PartitionView`` for the tile at *tile_idx*."""
        from mlir.dialects import pto as _pto
        from ._ir_builder import get_builder

        builder = get_builder()
        idx_ssa = ensure_index_ssa(tile_idx)
        ts_ssa = builder.constant_index(self._tile_sizes[self._dim])
        offset_dim = arith.MulIOp(idx_ssa, ts_ssa).result

        offsets, sizes, static_sizes = [], [], []
        for d in range(self._tensor.ndim):
            if d == self._dim:
                offsets.append(offset_dim)
                sizes.append(ts_ssa)
                static_sizes.append(self._tile_sizes[self._dim])
            else:
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

    # -- public API ------------------------------------------------------

    def __getitem__(self, idx):
        """Return the partition view for the tile at global index *idx*.

        *idx* may be an ``int`` or a :class:`ScalarValue`.
        """
        return self._make_partition(idx)

    @contextmanager
    def for_each(self, start=None, end=None, step=1):
        """Iterate tiles in ``[start, end)`` with *step*.

        Yields ``(tile_idx, partition_view)`` where *tile_idx* is the
        **global** tile index as a :class:`ScalarValue`.

        Defaults: ``start=0``, ``end=num_tiles``, ``step=1``.
        """
        s = ensure_index_ssa(start if start is not None else 0)
        if end is not None:
            e = ensure_index_ssa(end)
        else:
            e = ensure_index_ssa(self.num_tiles)
        st = ensure_index_ssa(step)

        loop = scf.ForOp(s, e, st, [])
        ip = InsertionPoint(loop.body)
        ip.__enter__()
        try:
            tile_idx = ScalarValue(loop.induction_variable)
            pv = self._make_partition(tile_idx)
            yield tile_idx, pv
            scf.YieldOp([])
        finally:
            ip.__exit__(None, None, None)

    def distribute(self):
        """Distribute tiles evenly across cores.

        Returns a :class:`DistributedTiledTensor` whose
        :meth:`~DistributedTiledTensor.for_each` iterates only over this
        core's share of tiles.
        """
        return DistributedTiledTensor(self)


class DistributedTiledTensor:
    """Tiles evenly split across cores.

    Created by :meth:`TiledTensor.distribute`.  Uses
    ``pto.get_block_idx`` / ``pto.get_block_num`` to determine each
    core's range.

    For uneven division the last core(s) may receive fewer (or zero) tiles.
    """

    def __init__(self, tiled_tensor):
        self._tiled = tiled_tensor

    @contextmanager
    def for_each(self):
        """Iterate this core's tiles.

        Yields ``(global_tile_idx, partition_view)``.
        """
        from mlir.dialects import pto as _pto

        idx_ty = IndexType.get()
        core_id = arith.IndexCastOp(
            idx_ty, _pto.GetBlockIdxOp().result
        ).result
        num_cores = arith.IndexCastOp(
            idx_ty, _pto.GetBlockNumOp().result
        ).result

        num_tiles = ensure_index_ssa(self._tiled.num_tiles)
        one = ensure_index_ssa(1)

        # tiles_per_core = ceildiv(num_tiles, num_cores)
        tpc = arith.DivSIOp(
            arith.AddIOp(num_tiles, arith.SubIOp(num_cores, one).result).result,
            num_cores,
        ).result

        # start = min(core_id * tpc, num_tiles)
        raw_start = arith.MulIOp(core_id, tpc).result
        start = _index_min(raw_start, num_tiles)

        # end = min((core_id + 1) * tpc, num_tiles)
        raw_end = arith.MulIOp(
            arith.AddIOp(core_id, one).result, tpc
        ).result
        end = _index_min(raw_end, num_tiles)

        with self._tiled.for_each(
            start=ScalarValue(start), end=ScalarValue(end)
        ) as result:
            yield result

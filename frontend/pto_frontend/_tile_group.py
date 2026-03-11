"""TileGroup: dynamic buffer selection for multi-buffering patterns.

Allows indexing a group of tiles with a dynamic ScalarValue, generating
``arith.select`` chains on the base address to select the correct tile at
runtime.  The result is a regular ``Tile`` whose ``_group_tiles`` attribute
references all underlying tiles for sync tracking.

Example::

    a = pto.TileGroup([a0, a1])
    b = pto.TileGroup([b0, b1])
    c = pto.TileGroup([c0, c1])

    buf_idx = (i * n_tiles + j) % 2
    pto.tload(a[buf_idx], pv_x)
    pto.tload(b[buf_idx], pv_y)
    pto.tadd(c[buf_idx], a[buf_idx], b[buf_idx])
    pto.tstore(pv_z, c[buf_idx])
"""

from mlir.dialects import arith
from mlir.dialects import pto as _pto

from ._scalar import ScalarValue
from ._tile import Tile


class TileGroup:
    """A group of tiles for dynamic buffer selection (double/multi-buffering).

    All tiles must share the same ``shape``, ``dtype``, and ``loc``
    (address space).  Only the base address differs between tiles.

    Indexing with a Python ``int`` returns the tile directly (trace-time).
    Indexing with a :class:`ScalarValue` generates an ``arith.select`` chain
    on the base address and returns a ``Tile`` with ``_group_tiles`` set to
    all underlying tiles (so the sync tracker covers every buffer).
    """

    def __init__(self, tiles):
        self._tiles = list(tiles)
        if len(self._tiles) < 2:
            raise ValueError("TileGroup requires at least 2 tiles")
        t0 = self._tiles[0]
        for t in self._tiles[1:]:
            if t.shape != t0.shape or t.dtype != t0.dtype or t.loc != t0.loc:
                raise ValueError(
                    "All tiles in a TileGroup must have the same "
                    "shape, dtype, and loc")

    def __len__(self):
        return len(self._tiles)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._tiles[idx]
        if isinstance(idx, ScalarValue):
            return self._dynamic_select(idx)
        raise TypeError(
            f"TileGroup index must be int or ScalarValue, "
            f"got {type(idx).__name__}")

    def _dynamic_select(self, idx):
        """Select a tile at runtime via ``arith.select`` on the base address.

        Generates a chain of ``arith.select`` on i64 addresses, then creates
        a new ``AllocTileOp`` with the selected address.  This avoids putting
        ``!pto.tile_buf`` types through ``scf.if`` results (which the PTO
        lowering passes don't handle).
        """
        from ._ir_builder import get_builder

        tiles = self._tiles
        builder = get_builder()

        # Build i64 address constants for each tile
        addrs = [builder.constant_i64(t.byte_offset) for t in tiles]

        # Select address using arith.select chain
        selected_addr = _build_addr_select(idx, addrs)

        # Create new AllocTileOp with the dynamically selected address
        t0 = tiles[0]
        tile_buf_type = t0.ssa.type
        tile_ssa = _pto.AllocTileOp(tile_buf_type, addr=selected_addr).result

        return Tile(tile_ssa, t0.shape, t0.dtype, t0.loc,
                    group_tiles=tiles, group_idx=idx)


def _build_addr_select(idx, addrs, cmp_val=0):
    """Recursively build ``arith.select`` chain for i64 address selection.

    Generates::

        select(idx==0, addr0, select(idx==1, addr1, ... addrN))
    """
    if len(addrs) == 1:
        return addrs[0]

    cond = (idx == cmp_val)
    rest = _build_addr_select(idx, addrs[1:], cmp_val + 1)
    return arith.SelectOp(cond.ssa, addrs[0], rest).result

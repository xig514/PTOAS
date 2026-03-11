"""Tile proxy wrapping a pto.tile_buf SSA value."""


class Tile:
    """Proxy for an allocated tile buffer."""

    __slots__ = ("ssa", "shape", "dtype", "loc", "byte_offset", "byte_size",
                 "_group_tiles", "_group_idx")

    def __init__(self, ssa, shape, dtype, loc, byte_offset=0, byte_size=0,
                 group_tiles=None, group_idx=None):
        self.ssa = ssa          # MLIR SSA Value of !pto.tile_buf<...>
        self.shape = shape      # tuple of ints (physical shape)
        self.dtype = dtype      # DType
        self.loc = loc          # AddressSpace enum value
        self.byte_offset = byte_offset  # starting byte address
        self.byte_size = byte_size      # total size in bytes
        self._group_tiles = group_tiles # list of Tiles when from TileGroup
        self._group_idx = group_idx     # ScalarValue used for selection

"""Tile proxy wrapping a pto.tile_buf SSA value."""


class Tile:
    """Proxy for an allocated tile buffer."""

    __slots__ = ("ssa", "shape", "dtype", "loc", "byte_offset", "byte_size")

    def __init__(self, ssa, shape, dtype, loc, byte_offset=0, byte_size=0):
        self.ssa = ssa          # MLIR SSA Value of !pto.tile_buf<...>
        self.shape = shape      # tuple of ints (physical shape)
        self.dtype = dtype      # DType
        self.loc = loc          # AddressSpace enum value
        self.byte_offset = byte_offset  # starting byte address
        self.byte_size = byte_size      # total size in bytes

"""Tile proxy wrapping a pto.tile_buf SSA value."""


class Tile:
    """Proxy for an allocated tile buffer."""

    __slots__ = ("ssa", "shape", "dtype", "loc")

    def __init__(self, ssa, shape, dtype, loc):
        self.ssa = ssa          # MLIR SSA Value of !pto.tile_buf<...>
        self.shape = shape      # tuple of ints (physical shape)
        self.dtype = dtype      # DType
        self.loc = loc          # AddressSpace enum value

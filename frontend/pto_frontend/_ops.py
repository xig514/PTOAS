"""High-level PTO operation wrappers.

Each function creates the underlying MLIR op at the current insertion point
and works with Tile / _PartitionView / ScalarValue proxies.
"""

from mlir.dialects import pto as _pto

from ._tile import Tile
from ._scalar import ScalarValue
from ._constants import VEC, MAT, LEFT, RIGHT, ACC, BIAS, SCALING


# ---------------------------------------------------------------------------
#  Tile configuration defaults by address space
# ---------------------------------------------------------------------------

_TILE_DEFAULTS = {
    #  addr_space: (BLayout,              SLayout,             fractal)
    VEC:     (_pto.BLayout.RowMajor, _pto.SLayout.NoneBox,  512),
    MAT:     (_pto.BLayout.ColMajor, _pto.SLayout.RowMajor, 512),
    LEFT:    (_pto.BLayout.RowMajor, _pto.SLayout.RowMajor, 512),
    RIGHT:   (_pto.BLayout.RowMajor, _pto.SLayout.ColMajor, 512),
    ACC:     (_pto.BLayout.ColMajor, _pto.SLayout.RowMajor, 1024),
    BIAS:    (_pto.BLayout.RowMajor, _pto.SLayout.NoneBox,  512),
    SCALING: (_pto.BLayout.RowMajor, _pto.SLayout.NoneBox,  512),
}


# ---------------------------------------------------------------------------
#  make_tile
# ---------------------------------------------------------------------------

def make_tile(shape, dtype, loc, addr=0, *,
              valid_shape=None, blayout=None, slayout=None,
              fractal=None, pad=None):
    """Allocate a tile buffer.

    Parameters
    ----------
    shape : tuple[int, ...]
        Physical shape of the tile (e.g. ``(32, 32)``).
    dtype : DType
        Element type.
    loc : AddressSpace enum
        Memory location (``VEC``, ``MAT``, ``LEFT``, ``RIGHT``, ``ACC``, …).
    addr : int or ScalarValue
        Byte address inside the memory space (default 0).
    valid_shape : tuple[int, ...] or None
        Logical valid shape (defaults to *shape*).
    blayout / slayout / fractal / pad
        Override the default tile-buffer configuration.
    """
    from ._ir_builder import get_builder

    defaults = _TILE_DEFAULTS.get(
        loc, (_pto.BLayout.RowMajor, _pto.SLayout.NoneBox, 512)
    )
    bl = blayout if blayout is not None else defaults[0]
    sl = slayout if slayout is not None else defaults[1]
    frac = fractal if fractal is not None else defaults[2]
    pd = pad if pad is not None else _pto.PadValue.Null

    bl_attr = _pto.BLayoutAttr.get(bl)
    sl_attr = _pto.SLayoutAttr.get(sl)
    pd_attr = _pto.PadValueAttr.get(pd)
    cfg = _pto.TileBufConfigAttr.get(bl_attr, sl_attr, frac, pd_attr)

    addr_space_attr = _pto.AddressSpaceAttr.get(loc)
    mlir_elem = dtype.to_mlir()
    shape_list = list(shape)
    vs = list(valid_shape) if valid_shape is not None else shape_list

    tile_buf_type = _pto.TileBufType.get(
        shape_list, mlir_elem, addr_space_attr, vs, cfg
    )

    builder = get_builder()
    if isinstance(addr, ScalarValue):
        addr_ssa = addr.ssa
    elif isinstance(addr, int):
        addr_ssa = builder.constant_i64(addr)
    else:
        addr_ssa = addr

    tile_ssa = _pto.AllocTileOp(tile_buf_type, addr=addr_ssa).result
    return Tile(tile_ssa, tuple(shape), dtype, loc)


# ---------------------------------------------------------------------------
#  DMA ops  (first arg = None for result-type override)
# ---------------------------------------------------------------------------

def tload(pv, tile):
    """Load from a partition view into a tile buffer."""
    _pto.TLoadOp(None, pv.ssa, tile.ssa)


def tstore(tile, pv):
    """Store a tile buffer back to a partition view."""
    _pto.TStoreOp(None, tile.ssa, pv.ssa)


def tload_tile(tensor, tile_coord, tile_layout, tile_buf):
    """Load data from tensor at tile coordinate into tile buffer.

    Parameters
    ----------
    tensor : _TensorProxy
        Source tensor
    tile_coord : TileCoordinate
        Coordinate of the tile to load
    tile_layout : TileLayout
        Layout of the tile
    tile_buf : Tile
        Destination tile buffer
    """
    pv = tensor.partition_at_coord(tile_coord, tile_layout)
    _pto.TLoadOp(None, pv.ssa, tile_buf.ssa)


def tstore_tile(tile_buf, tensor, tile_coord, tile_layout):
    """Store tile buffer data back to tensor at tile coordinate.

    Parameters
    ----------
    tile_buf : Tile
        Source tile buffer
    tensor : _TensorProxy
        Destination tensor
    tile_coord : TileCoordinate
        Coordinate of the tile to store
    tile_layout : TileLayout
        Layout of the tile
    """
    pv = tensor.partition_at_coord(tile_coord, tile_layout)
    _pto.TStoreOp(None, tile_buf.ssa, pv.ssa)


def tmov(src, dst):
    """Move data between tile buffers (possibly across address spaces)."""
    _pto.TMovOp(None, src.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Binary element-wise ops  (src0, src1, dst — no None prefix)
# ---------------------------------------------------------------------------

def _binary(op_cls, a, b, c):
    op_cls(a.ssa, b.ssa, c.ssa)


def tadd(src0, src1, dst):
    _binary(_pto.TAddOp, src0, src1, dst)


def tsub(src0, src1, dst):
    _binary(_pto.TSubOp, src0, src1, dst)


def tmul(src0, src1, dst):
    _binary(_pto.TMulOp, src0, src1, dst)


def tdiv(src0, src1, dst):
    _binary(_pto.TDivOp, src0, src1, dst)


def tand(src0, src1, dst):
    _binary(_pto.TAndOp, src0, src1, dst)


def tor(src0, src1, dst):
    _binary(_pto.TOrOp, src0, src1, dst)


def txor(src0, src1, dst):
    _binary(_pto.TXorOp, src0, src1, dst)


def tmax(src0, src1, dst):
    _binary(_pto.TMaxOp, src0, src1, dst)


def tmin(src0, src1, dst):
    _binary(_pto.TMinOp, src0, src1, dst)


# ---------------------------------------------------------------------------
#  Unary element-wise ops  (src, dst)
# ---------------------------------------------------------------------------

def _unary(op_cls, src, dst):
    op_cls(src.ssa, dst.ssa)


def texp(src, dst):
    _unary(_pto.TExpOp, src, dst)


def tlog(src, dst):
    _unary(_pto.TLogOp, src, dst)


def tsqrt(src, dst):
    _unary(_pto.TSqrtOp, src, dst)


def trsqrt(src, dst):
    _unary(_pto.TRsqrtOp, src, dst)


def trecip(src, dst):
    _unary(_pto.TRecipOp, src, dst)


def tneg(src, dst):
    _unary(_pto.TNegOp, src, dst)


def tnot(src, dst):
    _unary(_pto.TNotOp, src, dst)


def trelu(src, dst):
    _unary(_pto.TReluOp, src, dst)


def tabs(src, dst):
    _unary(_pto.TAbsOp, src, dst)


# ---------------------------------------------------------------------------
#  Tile-scalar ops  (src, scalar, dst)
# ---------------------------------------------------------------------------

def _scalar_op(op_cls, src, scalar, dst):
    from ._utils import make_scalar_constant

    if isinstance(scalar, (int, float)):
        scalar_ssa = make_scalar_constant(scalar, src.dtype)
    elif isinstance(scalar, ScalarValue):
        scalar_ssa = scalar.ssa
    else:
        scalar_ssa = scalar
    op_cls(src.ssa, scalar_ssa, dst.ssa)


def tadds(src, scalar, dst):
    _scalar_op(_pto.TAddSOp, src, scalar, dst)


def tsubs(src, scalar, dst):
    _scalar_op(_pto.TSubSOp, src, scalar, dst)


def tmuls(src, scalar, dst):
    _scalar_op(_pto.TMulSOp, src, scalar, dst)


def tdivs(src, scalar, dst):
    _scalar_op(_pto.TDivSOp, src, scalar, dst)


def tmaxs(src, scalar, dst):
    _scalar_op(_pto.TMaxSOp, src, scalar, dst)


def tmins(src, scalar, dst):
    _scalar_op(_pto.TMinSOp, src, scalar, dst)


# ---------------------------------------------------------------------------
#  Reduction ops  (src, tmp, dst)
# ---------------------------------------------------------------------------

def trowmax(src, tmp, dst):
    _pto.TRowMaxOp(src.ssa, tmp.ssa, dst.ssa)


def trowmin(src, tmp, dst):
    _pto.TRowMinOp(src.ssa, tmp.ssa, dst.ssa)


def trowsum(src, tmp, dst):
    _pto.TRowSumOp(src.ssa, tmp.ssa, dst.ssa)


def tcolmax(src, tmp, dst):
    _pto.TColMaxOp(src.ssa, tmp.ssa, dst.ssa)


def tcolmin(src, tmp, dst):
    _pto.TColMinOp(src.ssa, tmp.ssa, dst.ssa)


def tcolsum(src, tmp, dst):
    _pto.TColSumOp(src.ssa, tmp.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Matrix-multiplication ops  (None prefix)
# ---------------------------------------------------------------------------

def tmatmul(lhs, rhs, dst):
    _pto.TMatmulOp(None, lhs.ssa, rhs.ssa, dst.ssa)


def tmatmul_acc(acc, lhs, rhs, dst):
    _pto.TMatmulAccOp(None, acc.ssa, lhs.ssa, rhs.ssa, dst.ssa)


def tmatmul_bias(lhs, rhs, bias, dst):
    _pto.TMatmulBiasOp(None, lhs.ssa, rhs.ssa, bias.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Type conversion
# ---------------------------------------------------------------------------

def tcvt(src, dst, rmode=None):
    """Convert tile element type.  *rmode* is a ``pto.RoundMode`` enum."""
    kwargs = {}
    if rmode is not None:
        kwargs["rmode"] = _pto.RoundModeAttr.get(rmode)
    _pto.TCvtOp(src.ssa, dst.ssa, **kwargs)


# ---------------------------------------------------------------------------
#  Synchronisation
# ---------------------------------------------------------------------------

def record_event(src_op, dst_op, event_id):
    """Emit ``pto.record_event``.  Accepts SyncOpType enums directly."""
    _pto.record_event(src_op, dst_op, event_id)


def wait_event(src_op, dst_op, event_id):
    """Emit ``pto.wait_event``.  Accepts SyncOpType enums directly."""
    _pto.wait_event(src_op, dst_op, event_id)


def barrier_sync(op_type):
    """Emit ``pto.barrier_sync``.  Accepts SyncOpType enum directly."""
    _pto.barrier(op_type)


# ---------------------------------------------------------------------------
#  System / runtime queries
# ---------------------------------------------------------------------------

def get_block_idx():
    """Return the current block (core) index as a ScalarValue."""
    return ScalarValue(_pto.GetBlockIdxOp().result)


def get_block_num():
    """Return the total number of blocks as a ScalarValue."""
    return ScalarValue(_pto.GetBlockNumOp().result)

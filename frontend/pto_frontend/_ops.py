"""High-level PTO operation wrappers.

Each function creates the underlying MLIR op at the current insertion point
and works with Tile / _PartitionView / ScalarValue proxies.

Convention: destination operand first, source operand(s) after.
"""

from mlir.dialects import pto as _pto

from ._tile import Tile
from ._scalar import ScalarValue
from ._constants import VEC, MAT, LEFT, RIGHT, ACC, BIAS, SCALING

from mlir.dialects import arith


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
#  DMA ops  (dst, src)
# ---------------------------------------------------------------------------

def tload(tile, pv):
    """Load from a partition view into a tile buffer."""
    _pto.TLoadOp(None, pv.ssa, tile.ssa)


def tstore(pv, tile):
    """Store a tile buffer back to a partition view."""
    _pto.TStoreOp(None, tile.ssa, pv.ssa)


def tload_tile(tile_buf, tensor, tile_coord, tile_layout):
    """Load data from tensor at tile coordinate into tile buffer.

    Parameters
    ----------
    tile_buf : Tile
        Destination tile buffer
    tensor : _TensorProxy
        Source tensor
    tile_coord : TileCoordinate
        Coordinate of the tile to load
    tile_layout : TileLayout
        Layout of the tile
    """
    pv = tensor.partition_at_coord(tile_coord, tile_layout)
    _pto.TLoadOp(None, pv.ssa, tile_buf.ssa)


def tstore_tile(pv_tensor, tile_buf, tile_coord, tile_layout):
    """Store tile buffer data back to tensor at tile coordinate.

    Parameters
    ----------
    pv_tensor : _TensorProxy
        Destination tensor
    tile_buf : Tile
        Source tile buffer
    tile_coord : TileCoordinate
        Coordinate of the tile to store
    tile_layout : TileLayout
        Layout of the tile
    """
    pv = pv_tensor.partition_at_coord(tile_coord, tile_layout)
    _pto.TStoreOp(None, tile_buf.ssa, pv.ssa)


def tmov(dst, src):
    """Move data between tile buffers (possibly across address spaces)."""
    _pto.TMovOp(None, src.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Binary element-wise ops  (dst, src0, src1)
# ---------------------------------------------------------------------------

def _binary(op_cls, dst, src0, src1):
    op_cls(src0.ssa, src1.ssa, dst.ssa)


def tadd(dst, src0, src1):
    _binary(_pto.TAddOp, dst, src0, src1)


def tsub(dst, src0, src1):
    _binary(_pto.TSubOp, dst, src0, src1)


def tmul(dst, src0, src1):
    _binary(_pto.TMulOp, dst, src0, src1)


def tdiv(dst, src0, src1):
    _binary(_pto.TDivOp, dst, src0, src1)


def tand(dst, src0, src1):
    _binary(_pto.TAndOp, dst, src0, src1)


def tor(dst, src0, src1):
    _binary(_pto.TOrOp, dst, src0, src1)


def txor(dst, src0, src1):
    _binary(_pto.TXorOp, dst, src0, src1)


def tmax(dst, src0, src1):
    _binary(_pto.TMaxOp, dst, src0, src1)


def tmin(dst, src0, src1):
    _binary(_pto.TMinOp, dst, src0, src1)


# ---------------------------------------------------------------------------
#  Unary element-wise ops  (dst, src)
# ---------------------------------------------------------------------------

def _unary(op_cls, dst, src):
    op_cls(src.ssa, dst.ssa)


def texp(dst, src):
    _unary(_pto.TExpOp, dst, src)


def tlog(dst, src):
    _unary(_pto.TLogOp, dst, src)


def tsqrt(dst, src):
    _unary(_pto.TSqrtOp, dst, src)


def trsqrt(dst, src):
    _unary(_pto.TRsqrtOp, dst, src)


def trecip(dst, src):
    _unary(_pto.TRecipOp, dst, src)


def tneg(dst, src):
    _unary(_pto.TNegOp, dst, src)


def tnot(dst, src):
    _unary(_pto.TNotOp, dst, src)


def trelu(dst, src):
    _unary(_pto.TReluOp, dst, src)


def tabs(dst, src):
    _unary(_pto.TAbsOp, dst, src)


# ---------------------------------------------------------------------------
#  Tile-scalar ops  (dst, src, scalar)
# ---------------------------------------------------------------------------

def _scalar_op(op_cls, dst, src, scalar):
    from ._utils import make_scalar_constant

    if isinstance(scalar, (int, float)):
        scalar_ssa = make_scalar_constant(scalar, src.dtype)
    elif isinstance(scalar, ScalarValue):
        scalar_ssa = scalar.ssa
    else:
        scalar_ssa = scalar
    op_cls(src.ssa, scalar_ssa, dst.ssa)


def tadds(dst, src, scalar):
    _scalar_op(_pto.TAddSOp, dst, src, scalar)


def tsubs(dst, src, scalar):
    _scalar_op(_pto.TSubSOp, dst, src, scalar)


def tmuls(dst, src, scalar):
    _scalar_op(_pto.TMulSOp, dst, src, scalar)


def tdivs(dst, src, scalar):
    _scalar_op(_pto.TDivSOp, dst, src, scalar)


def tmaxs(dst, src, scalar):
    _scalar_op(_pto.TMaxSOp, dst, src, scalar)


def tmins(dst, src, scalar):
    _scalar_op(_pto.TMinSOp, dst, src, scalar)


# ---------------------------------------------------------------------------
#  Reduction ops  (dst, src, tmp)
# ---------------------------------------------------------------------------

def trowmax(dst, src, tmp):
    _pto.TRowMaxOp(src.ssa, tmp.ssa, dst.ssa)


def trowmin(dst, src, tmp):
    _pto.TRowMinOp(src.ssa, tmp.ssa, dst.ssa)


def trowsum(dst, src, tmp):
    _pto.TRowSumOp(src.ssa, tmp.ssa, dst.ssa)


def tcolmax(dst, src, tmp):
    _pto.TColMaxOp(src.ssa, tmp.ssa, dst.ssa)


def tcolmin(dst, src, tmp):
    _pto.TColMinOp(src.ssa, tmp.ssa, dst.ssa)


def tcolsum(dst, src, tmp):
    _pto.TColSumOp(src.ssa, tmp.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Matrix-multiplication ops  (dst, ...)
# ---------------------------------------------------------------------------

def tmatmul(dst, lhs, rhs):
    _pto.TMatmulOp(None, lhs.ssa, rhs.ssa, dst.ssa)


def tmatmul_acc(dst, acc, lhs, rhs):
    _pto.TMatmulAccOp(None, acc.ssa, lhs.ssa, rhs.ssa, dst.ssa)


def tmatmul_bias(dst, lhs, rhs, bias):
    _pto.TMatmulBiasOp(None, lhs.ssa, rhs.ssa, bias.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Data movement / layout
# ---------------------------------------------------------------------------

def ttrans(dst, src):
    """Transpose a tile into *dst* using an implementation-defined temporary."""
    _pto.TTransOp(src.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Row-expand broadcast ops  (dst, src, ...)
# ---------------------------------------------------------------------------

def trowexpand(dst, src):
    """Broadcast first element of each row across the entire row."""
    _pto.TRowExpandOp(src.ssa, dst.ssa)


def trowexpanddiv(dst, src, div_vec):
    """Row-wise broadcast divide: dst[i,j] = src[i,j] / div_vec[i,0]."""
    _pto.TRowExpandDivOp(src.ssa, div_vec.ssa, dst.ssa)


def trowexpandmul(dst, src, mul_vec):
    """Row-wise broadcast multiply: dst[i,j] = src[i,j] * mul_vec[i,0]."""
    _pto.TRowExpandMulOp(src.ssa, mul_vec.ssa, dst.ssa)


def trowexpandsub(dst, src, sub_vec):
    """Row-wise broadcast subtract: dst[i,j] = src[i,j] - sub_vec[i,0]."""
    _pto.TRowExpandSubOp(src.ssa, sub_vec.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Type conversion
# ---------------------------------------------------------------------------

def tcvt(dst, src, rmode=None):
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


def set_flag(src_pipe, dst_pipe, event_id):
    """Emit ``pto.set_flag`` for low-level pipeline synchronization.

    Signals that operations on *src_pipe* have completed, allowing
    *dst_pipe* to proceed.  Accepts ``PIPE`` and ``EVENT`` enums.
    """
    from mlir.dialects.pto import PipeAttr, EventAttr, PIPE, EVENT
    src = PipeAttr.get(src_pipe) if isinstance(src_pipe, PIPE) else src_pipe
    dst = PipeAttr.get(dst_pipe) if isinstance(dst_pipe, PIPE) else dst_pipe
    evt = EventAttr.get(event_id) if isinstance(event_id, EVENT) else event_id
    _pto.SetFlagOp(src_pipe=src, dst_pipe=dst, event_id=evt)


def wait_flag(src_pipe, dst_pipe, event_id):
    """Emit ``pto.wait_flag`` for low-level pipeline synchronization.

    Blocks *dst_pipe* until the matching ``set_flag`` from *src_pipe*
    has been issued.  Accepts ``PIPE`` and ``EVENT`` enums.
    """
    from mlir.dialects.pto import PipeAttr, EventAttr, PIPE, EVENT
    src = PipeAttr.get(src_pipe) if isinstance(src_pipe, PIPE) else src_pipe
    dst = PipeAttr.get(dst_pipe) if isinstance(dst_pipe, PIPE) else dst_pipe
    evt = EventAttr.get(event_id) if isinstance(event_id, EVENT) else event_id
    _pto.WaitFlagOp(src_pipe=src, dst_pipe=dst, event_id=evt)


# ---------------------------------------------------------------------------
#  System / runtime queries
# ---------------------------------------------------------------------------

def get_block_idx():
    """Return the current block (core) index as a ScalarValue."""
    return ScalarValue(_pto.GetBlockIdxOp().result)


def get_block_num():
    """Return the total number of blocks as a ScalarValue."""
    return ScalarValue(_pto.GetBlockNumOp().result)


# ---------------------------------------------------------------------------
#  Scalar memory access
# ---------------------------------------------------------------------------

def get_value(tensor, idx, *, as_index=True):
    """Load a single scalar element from a tensor at the given index.

    Uses ``pto.load_scalar`` to read one element from global memory.
    The result is a ScalarValue usable in arithmetic, loop bounds, etc.

    Parameters
    ----------
    tensor : _TensorProxy
        Source tensor
    idx : int or ScalarValue
        Flat element offset
    as_index : bool
        If True and the element type is integer, automatically cast
        the result to index type for use in arithmetic and loop bounds.

    Returns
    -------
    ScalarValue
        The loaded scalar value
    """
    from ._utils import ensure_index_ssa
    from mlir.ir import IndexType

    idx_ssa = ensure_index_ssa(idx)
    result_type = tensor.dtype.to_mlir()
    val = _pto.load_scalar(result_type, tensor.ptr_ssa, idx_ssa)
    is_float = tensor.dtype.name in ("float16", "float32", "bfloat16")

    if as_index and not is_float:
        idx_ty = IndexType.get()
        val = arith.IndexCastOp(idx_ty, val).result
    return ScalarValue(val, is_float=is_float)

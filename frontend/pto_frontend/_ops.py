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
from mlir.dialects.pto import PIPE


# ---------------------------------------------------------------------------
#  Sync tracker helpers
# ---------------------------------------------------------------------------

def _get_tracker():
    from ._sync_tracker import get_sync_tracker
    return get_sync_tracker()


def _product(seq):
    """Return the product of an iterable of ints."""
    r = 1
    for x in seq:
        r *= x
    return r


def _tmov_pipe(dst, src):
    """Determine the pipeline for a tmov based on src/dst address spaces."""
    if src.loc == MAT and dst.loc in (LEFT, RIGHT):
        return PIPE.PIPE_MTE1
    if src.loc == MAT and dst.loc == VEC:
        return PIPE.PIPE_V
    if src.loc == MAT and dst.loc == BIAS:
        return PIPE.PIPE_MTE1
    # VEC→MAT, MAT→SCALING are FIX
    if src.loc == VEC and dst.loc == MAT:
        return PIPE.PIPE_FIX
    if src.loc == MAT and dst.loc == SCALING:
        return PIPE.PIPE_FIX
    return PIPE.PIPE_MTE3


# ---------------------------------------------------------------------------
#  TileType — reusable tile buffer descriptor
# ---------------------------------------------------------------------------

class TileType:
    """Descriptor that bundles tile buffer configuration.

    Groups ``(shape, dtype, loc)`` with optional layout overrides so that
    multiple tiles sharing the same configuration can be created concisely::

        mat_f16 = TileType((64, 64), float16, MAT)
        a = make_tile(mat_f16, addr=0)
        b = make_tile(mat_f16, addr=8192)

    Parameters
    ----------
    shape : tuple[int, ...]
        Physical shape of the tile (e.g. ``(32, 32)``).
    dtype : DType
        Element type.
    loc : AddressSpace enum
        Memory location (``VEC``, ``MAT``, ``LEFT``, ``RIGHT``, ``ACC``, …).
    valid_shape / blayout / slayout / fractal / pad
        Optional layout overrides (same semantics as ``make_tile``).
    """

    __slots__ = ("shape", "dtype", "loc", "valid_shape",
                 "blayout", "slayout", "fractal", "pad")

    def __init__(self, shape, dtype, loc, *,
                 valid_shape=None, blayout=None, slayout=None,
                 fractal=None, pad=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.loc = loc
        self.valid_shape = tuple(valid_shape) if valid_shape is not None else None
        self.blayout = blayout
        self.slayout = slayout
        self.fractal = fractal
        self.pad = pad

    def __repr__(self):
        parts = [repr(self.shape), repr(self.dtype), f"{self.loc}"]
        if self.valid_shape is not None:
            parts.append(f"valid_shape={self.valid_shape}")
        if self.blayout is not None:
            parts.append(f"blayout={self.blayout}")
        if self.slayout is not None:
            parts.append(f"slayout={self.slayout}")
        return f"TileType({', '.join(parts)})"


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

def make_tile(shape_or_type, dtype=None, loc=None, addr=0, *,
              valid_shape=None, blayout=None, slayout=None,
              fractal=None, pad=None):
    """Allocate a tile buffer.

    Can be called in two forms::

        make_tile((64, 64), pto.float16, pto.MAT, addr=0)
        make_tile(tile_type, addr=0)

    Parameters
    ----------
    shape_or_type : tuple[int, ...] or TileType
        Physical shape, or a ``TileType`` that bundles shape/dtype/loc/layout.
    dtype : DType, optional
        Element type (required when *shape_or_type* is a tuple).
    loc : AddressSpace enum, optional
        Memory location (required when *shape_or_type* is a tuple).
    addr : int or ScalarValue
        Byte address inside the memory space (default 0).
    valid_shape : tuple[int, ...] or None
        Logical valid shape (defaults to *shape*).
    blayout / slayout / fractal / pad
        Override the default tile-buffer configuration.
    """
    from ._ir_builder import get_builder

    if isinstance(shape_or_type, TileType):
        tt = shape_or_type
        shape = tt.shape
        dtype = dtype if dtype is not None else tt.dtype
        loc = loc if loc is not None else tt.loc
        valid_shape = valid_shape if valid_shape is not None else tt.valid_shape
        blayout = blayout if blayout is not None else tt.blayout
        slayout = slayout if slayout is not None else tt.slayout
        fractal = fractal if fractal is not None else tt.fractal
        pad = pad if pad is not None else tt.pad
    else:
        shape = shape_or_type
        if dtype is None or loc is None:
            raise TypeError(
                "make_tile() requires dtype and loc when first argument "
                "is not a TileType"
            )

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
        addr_int = 0  # dynamic — can't do static overlap analysis
    elif isinstance(addr, int):
        addr_ssa = builder.constant_i64(addr)
        addr_int = addr
    else:
        addr_ssa = addr
        addr_int = 0

    byte_size = _product(shape) * dtype.byte_size

    tile_ssa = _pto.AllocTileOp(tile_buf_type, addr=addr_ssa).result
    tile = Tile(tile_ssa, tuple(shape), dtype, loc,
                byte_offset=addr_int, byte_size=byte_size)

    tracker = _get_tracker()
    if tracker:
        tracker.register_tile(tile, loc, addr_int, byte_size)

    return tile


# ---------------------------------------------------------------------------
#  DMA ops  (dst, src)
# ---------------------------------------------------------------------------

def tload(tile, src, offsets=None, layout=None):
    """Load from a partition view (or tensor with offsets) into a tile buffer.

    Parameters
    ----------
    tile : Tile
        Destination tile buffer.
    src : _PartitionView or _TensorProxy
        Source data. If a ``_TensorProxy`` is given, *offsets* must be provided
        and a partition is created internally using ``tile.shape`` as sizes.
    offsets : list[int|ScalarValue], optional
        When *src* is a ``_TensorProxy``, element offsets for each dimension.
        Sizes are inferred from ``tile.shape``.
    layout : str or Layout enum, optional
        Memory layout of the source tensor view.  ``"DN"`` (or
        ``Layout.DN``) creates a column-major (transposed) view of a 2-D
        source — the offsets are given in the **original** coordinate
        system and swapped internally.  ``"ND"`` / ``None`` keeps the
        default row-major behaviour.  ``"NZ"`` is accepted for
        forward-compatibility but currently behaves like ``"ND"``.
    """
    from ._tensor import _TensorProxy

    # Normalise layout to a string (or None).
    if layout is not None:
        layout_str = layout if isinstance(layout, str) else layout.name
        layout_str = layout_str.upper()
    else:
        layout_str = None

    # DN path: build a transposed tensor view, then partition + load.
    if layout_str == "DN":
        from ._utils import ensure_index_ssa
        from ._ir_builder import get_builder

        assert isinstance(src, _TensorProxy) and src.ndim == 2, \
            "tload with layout='DN' requires a 2D tensor source"

        builder = get_builder()

        # Source tensor shape [dim0, dim1], row-major strides [dim1, 1].
        dim0_ssa = src._shape_ssas[0]
        dim1_ssa = src._shape_ssas[1]

        # Transposed view: shape [dim1, dim0], col-major strides [1, dim1].
        tv_type = _pto.TensorViewType.get(2, src.dtype.to_mlir())
        tv_op = _pto.MakeTensorViewOp(
            tv_type, src.ptr_ssa, [dim1_ssa, dim0_ssa],
            [builder.constant_index(1), dim1_ssa]
        )
        # Mark as DN so InferPTOLayout respects it (dynamic strides
        # prevent automatic inference) and EmitC emits Layout::DN.
        tv_op.operation.attributes["layout"] = _pto.LayoutAttr.get(
            _pto.Layout.DN)
        tv = tv_op.result

        # Convert offsets from source coords [row, col] → transposed [col, row].
        if offsets is not None:
            off_ssas = [ensure_index_ssa(offsets[1]),
                        ensure_index_ssa(offsets[0])]
        else:
            off_ssas = [builder.constant_index(0),
                        builder.constant_index(0)]

        sz_ssas = [builder.constant_index(s) for s in tile.shape]
        static_sizes = list(tile.shape)

        pv_type = _pto.PartitionTensorViewType.get(static_sizes,
                                                    src.dtype.to_mlir())
        pv = _pto.PartitionViewOp(
            pv_type, tv, offsets=off_ssas, sizes=sz_ssas
        ).result

        tracker = _get_tracker()
        if tracker:
            tracker.record_op(PIPE.PIPE_MTE2, reads=[], writes=[tile])
        _pto.TLoadOp(None, pv, tile.ssa)
        return

    # Default (ND / NZ / None) path.
    if offsets is not None and isinstance(src, _TensorProxy):
        pv = src.partition(offsets=offsets, sizes=list(tile.shape))
        src = pv
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_MTE2, reads=[], writes=[tile])
    _pto.TLoadOp(None, src.ssa, tile.ssa)


def tstore(dst, tile, offsets=None):
    """Store a tile buffer back to a partition view (or tensor with offsets).

    Parameters
    ----------
    dst : _PartitionView or _TensorProxy
        Destination. If a ``_TensorProxy`` is given, *offsets* must be provided
        and a partition is created internally using ``tile.shape`` as sizes.
    tile : Tile
        Source tile buffer.
    offsets : list[int|ScalarValue], optional
        When *dst* is a ``_TensorProxy``, element offsets for each dimension.
        Sizes are inferred from ``tile.shape``.
    """
    from ._tensor import _TensorProxy
    if offsets is not None and isinstance(dst, _TensorProxy):
        pv = dst.partition(offsets=offsets, sizes=list(tile.shape))
        dst = pv
    tracker = _get_tracker()
    if tracker:
        pipe = PIPE.PIPE_FIX if tile.loc == ACC else PIPE.PIPE_MTE3
        tracker.record_op(pipe, reads=[tile], writes=[])
    _pto.TStoreOp(None, tile.ssa, dst.ssa)


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
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_MTE2, reads=[], writes=[tile_buf])
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
    tracker = _get_tracker()
    if tracker:
        pipe = PIPE.PIPE_FIX if tile_buf.loc == ACC else PIPE.PIPE_MTE3
        tracker.record_op(pipe, reads=[tile_buf], writes=[])
    pv = pv_tensor.partition_at_coord(tile_coord, tile_layout)
    _pto.TStoreOp(None, tile_buf.ssa, pv.ssa)


def tmov(dst, src):
    """Move data between tile buffers (possibly across address spaces)."""
    tracker = _get_tracker()
    if tracker:
        pipe = _tmov_pipe(dst, src)
        tracker.record_op(pipe, reads=[src], writes=[dst])
    _pto.TMovOp(None, src.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Binary element-wise ops  (dst, src0, src1)
# ---------------------------------------------------------------------------

def _binary(op_cls, dst, src0, src1):
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_V, reads=[src0, src1], writes=[dst])
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
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_V, reads=[src], writes=[dst])
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

    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_V, reads=[src], writes=[dst])

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

def _reduction(op_cls, dst, src, tmp):
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_V, reads=[src, tmp], writes=[dst])
    op_cls(src.ssa, tmp.ssa, dst.ssa)


def trowmax(dst, src, tmp):
    _reduction(_pto.TRowMaxOp, dst, src, tmp)


def trowmin(dst, src, tmp):
    _reduction(_pto.TRowMinOp, dst, src, tmp)


def trowsum(dst, src, tmp):
    _reduction(_pto.TRowSumOp, dst, src, tmp)


def tcolmax(dst, src, tmp):
    _reduction(_pto.TColMaxOp, dst, src, tmp)


def tcolmin(dst, src, tmp):
    _reduction(_pto.TColMinOp, dst, src, tmp)


def tcolsum(dst, src, tmp):
    _reduction(_pto.TColSumOp, dst, src, tmp)


# ---------------------------------------------------------------------------
#  Matrix-multiplication ops  (dst, ...)
# ---------------------------------------------------------------------------

def tmatmul(dst, lhs, rhs):
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_M, reads=[lhs, rhs], writes=[dst])
    _pto.TMatmulOp(None, lhs.ssa, rhs.ssa, dst.ssa)


def tmatmul_acc(dst, acc, lhs, rhs):
    tracker = _get_tracker()
    if tracker:
        reads = [lhs, rhs]
        if id(acc) != id(dst):
            reads.append(acc)
        tracker.record_op(PIPE.PIPE_M, reads=reads, writes=[dst])
    _pto.TMatmulAccOp(None, acc.ssa, lhs.ssa, rhs.ssa, dst.ssa)


def tmatmul_bias(dst, lhs, rhs, bias):
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_M, reads=[lhs, rhs, bias], writes=[dst])
    _pto.TMatmulBiasOp(None, lhs.ssa, rhs.ssa, bias.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Data movement / layout
# ---------------------------------------------------------------------------

def ttrans(dst, src):
    """Transpose a tile into *dst* using an implementation-defined temporary."""
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_V, reads=[src], writes=[dst])
    _pto.TTransOp(src.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Row-expand broadcast ops  (dst, src, ...)
# ---------------------------------------------------------------------------

def trowexpand(dst, src):
    """Broadcast first element of each row across the entire row."""
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_V, reads=[src], writes=[dst])
    _pto.TRowExpandOp(src.ssa, dst.ssa)


def trowexpanddiv(dst, src, div_vec):
    """Row-wise broadcast divide: dst[i,j] = src[i,j] / div_vec[i,0]."""
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_V, reads=[src, div_vec], writes=[dst])
    _pto.TRowExpandDivOp(src.ssa, div_vec.ssa, dst.ssa)


def trowexpandmul(dst, src, mul_vec):
    """Row-wise broadcast multiply: dst[i,j] = src[i,j] * mul_vec[i,0]."""
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_V, reads=[src, mul_vec], writes=[dst])
    _pto.TRowExpandMulOp(src.ssa, mul_vec.ssa, dst.ssa)


def trowexpandsub(dst, src, sub_vec):
    """Row-wise broadcast subtract: dst[i,j] = src[i,j] - sub_vec[i,0]."""
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_V, reads=[src, sub_vec], writes=[dst])
    _pto.TRowExpandSubOp(src.ssa, sub_vec.ssa, dst.ssa)


# ---------------------------------------------------------------------------
#  Type conversion
# ---------------------------------------------------------------------------

def tcvt(dst, src, rmode=None):
    """Convert tile element type.  *rmode* is a ``pto.RoundMode`` enum."""
    tracker = _get_tracker()
    if tracker:
        tracker.record_op(PIPE.PIPE_V, reads=[src], writes=[dst])
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


def sync_set(pipe, event_id):
    """Emit ``pto.sync.set`` for cross-core (cube↔vector) synchronization.

    Signals that operations on the specified pipeline have completed,
    allowing the other core type to proceed.
    Lowers to ``ffts_cross_core_sync`` (A3) or ``set_intra_block`` (A5).

    Parameters
    ----------
    pipe : PIPE enum
        Pipeline that has completed (e.g. ``PIPE_FIX``, ``PIPE_M``).
    event_id : int
        Synchronization event identifier.
    """
    from mlir.dialects.pto import PipeAttr, PIPE as _PIPE
    p = PipeAttr.get(pipe) if isinstance(pipe, _PIPE) else pipe
    _pto.SyncSetOp(pipe=p, event_id=event_id)


def sync_wait(pipe, event_id):
    """Emit ``pto.sync.wait`` for cross-core (cube↔vector) synchronization.

    Blocks the current core until the matching ``sync_set`` from the
    other core type has been issued.
    Lowers to ``wait_flag_dev`` (A3) or ``wait_intra_block`` (A5).

    Parameters
    ----------
    pipe : PIPE enum
        Pipeline to wait on (e.g. ``PIPE_V``, ``PIPE_M``).
    event_id : int
        Synchronization event identifier (must match the ``sync_set``).
    """
    from mlir.dialects.pto import PipeAttr, PIPE as _PIPE
    p = PipeAttr.get(pipe) if isinstance(pipe, _PIPE) else pipe
    _pto.SyncWaitOp(pipe=p, event_id=event_id)


def set_flag(src_pipe, dst_pipe, event_id):
    """Emit ``pto.set_flag`` for low-level pipeline synchronization.

    Signals that operations on *src_pipe* have completed, allowing
    *dst_pipe* to proceed.  Accepts ``PIPE`` and ``EVENT`` enums.

    If *event_id* is from EventIdGroup[ScalarValue], automatically
    generates conditional branches for runtime EVENT_ID selection.
    """
    from mlir.dialects.pto import PipeAttr, EventAttr, PIPE, EVENT
    from ._event_group import _DynamicEventSelection

    # Check if this is a dynamic event selection
    if isinstance(event_id, _DynamicEventSelection):
        # Generate conditional code: if idx==0: set_flag(ID0) else: set_flag(ID1) ...
        def emit_set(evt):
            src = PipeAttr.get(src_pipe) if isinstance(src_pipe, PIPE) else src_pipe
            dst = PipeAttr.get(dst_pipe) if isinstance(dst_pipe, PIPE) else dst_pipe
            evt_attr = EventAttr.get(evt) if isinstance(evt, EVENT) else evt
            _pto.SetFlagOp(src_pipe=src, dst_pipe=dst, event_id=evt_attr)
        event_id.emit_conditional(emit_set)
        return

    # Normal path: static EVENT_ID
    src = PipeAttr.get(src_pipe) if isinstance(src_pipe, PIPE) else src_pipe
    dst = PipeAttr.get(dst_pipe) if isinstance(dst_pipe, PIPE) else dst_pipe
    evt = EventAttr.get(event_id) if isinstance(event_id, EVENT) else event_id
    _pto.SetFlagOp(src_pipe=src, dst_pipe=dst, event_id=evt)


def wait_flag(src_pipe, dst_pipe, event_id):
    """Emit ``pto.wait_flag`` for low-level pipeline synchronization.

    Blocks *dst_pipe* until the matching ``set_flag`` from *src_pipe*
    has been issued.  Accepts ``PIPE`` and ``EVENT`` enums.

    If *event_id* is from EventIdGroup[ScalarValue], automatically
    generates conditional branches for runtime EVENT_ID selection.
    """
    from mlir.dialects.pto import PipeAttr, EventAttr, PIPE, EVENT
    from ._event_group import _DynamicEventSelection

    # Check if this is a dynamic event selection
    if isinstance(event_id, _DynamicEventSelection):
        # Generate conditional code: if idx==0: wait_flag(ID0) else: wait_flag(ID1) ...
        def emit_wait(evt):
            src = PipeAttr.get(src_pipe) if isinstance(src_pipe, PIPE) else src_pipe
            dst = PipeAttr.get(dst_pipe) if isinstance(dst_pipe, PIPE) else dst_pipe
            evt_attr = EventAttr.get(evt) if isinstance(evt, EVENT) else evt
            _pto.WaitFlagOp(src_pipe=src, dst_pipe=dst, event_id=evt_attr)
        event_id.emit_conditional(emit_wait)
        return

    # Normal path: static EVENT_ID
    src = PipeAttr.get(src_pipe) if isinstance(src_pipe, PIPE) else src_pipe
    dst = PipeAttr.get(dst_pipe) if isinstance(dst_pipe, PIPE) else dst_pipe
    evt = EventAttr.get(event_id) if isinstance(event_id, EVENT) else event_id
    _pto.WaitFlagOp(src_pipe=src, dst_pipe=dst, event_id=evt)


# ---------------------------------------------------------------------------
#  System / runtime queries
# ---------------------------------------------------------------------------

def get_block_idx():
    """Return the current block (core) index as a ScalarValue (index type)."""
    from mlir.ir import IndexType
    val = _pto.GetBlockIdxOp().result
    val = arith.IndexCastOp(IndexType.get(), val).result
    return ScalarValue(val)


def get_block_num():
    """Return the total number of blocks as a ScalarValue (index type)."""
    from mlir.ir import IndexType
    val = _pto.GetBlockNumOp().result
    val = arith.IndexCastOp(IndexType.get(), val).result
    return ScalarValue(val)


def get_subblock_idx():
    """Return the current sub-block (vector core) index as a ScalarValue (index type).

    On Ascend NPU each Cube core has 2 Vector sub-blocks (index 0 or 1).
    """
    from mlir.ir import IndexType
    val = _pto.GetSubBlockIdxOp().result
    val = arith.IndexCastOp(IndexType.get(), val).result
    return ScalarValue(val)


def get_subblock_num():
    """Return the total number of sub-blocks per block as a ScalarValue (index type)."""
    from mlir.ir import IndexType
    val = _pto.GetSubBlockNumOp().result
    val = arith.IndexCastOp(IndexType.get(), val).result
    return ScalarValue(val)


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

"""Utility helpers for SSA value conversions and stride computation."""

from mlir.ir import FloatAttr, IntegerAttr
from mlir.dialects import arith


def ensure_index_ssa(val):
    """Convert *val* to an SSA index value.

    Handles ints, ScalarValues, DynVars, and raw SSA values.  If the underlying
    SSA value is not already ``index``-typed (e.g. ``i64`` from
    ``get_block_idx``), an ``arith.index_cast`` is inserted automatically.
    """
    from ._ir_builder import get_builder
    from ._scalar import ScalarValue
    from ._dynvar import DynVar
    from mlir.ir import IndexType

    if isinstance(val, int):
        return get_builder().constant_index(val)
    if isinstance(val, DynVar):
        val = val._require_bound().ssa
    elif isinstance(val, ScalarValue):
        val = val.ssa
    # val is now a raw SSA value — cast to index if needed
    idx_ty = IndexType.get()
    if val.type != idx_ty:
        return arith.IndexCastOp(idx_ty, val).result
    return val


def compute_row_major_strides(shape_ssas):
    """Compute row-major strides from a list of SSA index values."""
    from ._ir_builder import get_builder

    n = len(shape_ssas)
    if n == 0:
        return []
    strides = [None] * n
    strides[n - 1] = get_builder().constant_index(1)
    for i in range(n - 2, -1, -1):
        strides[i] = arith.MulIOp(shape_ssas[i + 1], strides[i + 1]).result
    return strides


def make_scalar_constant(val, dtype):
    """Create an arith.constant SSA value matching *dtype*."""
    mlir_type = dtype.to_mlir()
    if dtype.name in ("float16", "float32", "bfloat16"):
        return arith.ConstantOp(
            mlir_type, FloatAttr.get(mlir_type, float(val))
        ).result
    else:
        return arith.ConstantOp(
            mlir_type, IntegerAttr.get(mlir_type, int(val))
        ).result

"""Thread-local IR builder managing MLIR Context, Location, and Module."""

import threading

from mlir.ir import (
    Context,
    Location,
    Module,
    IndexType,
    IntegerType,
    IntegerAttr,
    FloatAttr,
    F16Type,
    F32Type,
    BF16Type,
)
from mlir.dialects import arith
from mlir.dialects import pto as _pto_dialect

_tls = threading.local()


class IRBuilder:
    """Manages an MLIR Context, Location, and Module for a single kernel trace."""

    def __init__(self):
        self.ctx = Context()
        self.ctx.__enter__()
        _pto_dialect.register_dialect(self.ctx, load=True)
        self.loc = Location.unknown(self.ctx)
        self.loc.__enter__()
        self.module = Module.create()

    # -- constant helpers (created at the current insertion point) --

    def constant_index(self, val):
        return arith.ConstantOp(IndexType.get(), val).result

    def constant_i64(self, val):
        i64 = IntegerType.get_signless(64)
        return arith.ConstantOp(i64, IntegerAttr.get(i64, val)).result

    def constant_f32(self, val):
        f32 = F32Type.get()
        return arith.ConstantOp(f32, FloatAttr.get(f32, float(val))).result

    def constant_f16(self, val):
        f16 = F16Type.get()
        return arith.ConstantOp(f16, FloatAttr.get(f16, float(val))).result

    # -- IR emission --

    def emit_ir(self):
        self.module.operation.verify()
        return str(self.module)

    def close(self):
        self.loc.__exit__(None, None, None)
        self.ctx.__exit__(None, None, None)


def get_builder():
    """Return the thread-local IRBuilder (set during @kernel tracing)."""
    return getattr(_tls, "builder", None)


def set_builder(builder):
    _tls.builder = builder


def clear_builder():
    _tls.builder = None

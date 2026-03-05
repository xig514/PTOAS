"""ScalarValue proxy wrapping an MLIR SSA value with operator overloads."""

from mlir.dialects import arith
from mlir.dialects.arith import CmpIPredicate


class ScalarValue:
    """Proxy for an SSA scalar value (index, integer, or float).

    Arithmetic operators emit arith dialect ops and return new ScalarValues.
    """

    def __init__(self, ssa, is_float=False):
        self.ssa = ssa
        self.is_float = is_float

    def _coerce(self, other):
        if isinstance(other, ScalarValue):
            return other.ssa
        from ._ir_builder import get_builder

        if isinstance(other, int):
            # Create a constant matching self.ssa's type so that arith ops
            # don't get a type mismatch (e.g. i64 vs index).
            from mlir.ir import IndexType, IntegerType, IntegerAttr

            ty = self.ssa.type
            if ty == IndexType.get():
                return get_builder().constant_index(other)
            if ty == IntegerType.get_signless(64):
                return get_builder().constant_i64(other)
            # Fallback: treat as index (legacy behaviour)
            return get_builder().constant_index(other)
        if isinstance(other, float):
            return get_builder().constant_f32(other)
        return other

    # -- arithmetic --

    def __add__(self, other):
        rhs = self._coerce(other)
        if self.is_float:
            return ScalarValue(arith.AddFOp(self.ssa, rhs).result, is_float=True)
        return ScalarValue(arith.AddIOp(self.ssa, rhs).result)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        rhs = self._coerce(other)
        if self.is_float:
            return ScalarValue(arith.SubFOp(self.ssa, rhs).result, is_float=True)
        return ScalarValue(arith.SubIOp(self.ssa, rhs).result)

    def __rsub__(self, other):
        lhs = self._coerce(other)
        if self.is_float:
            return ScalarValue(arith.SubFOp(lhs, self.ssa).result, is_float=True)
        return ScalarValue(arith.SubIOp(lhs, self.ssa).result)

    def __mul__(self, other):
        rhs = self._coerce(other)
        if self.is_float:
            return ScalarValue(arith.MulFOp(self.ssa, rhs).result, is_float=True)
        return ScalarValue(arith.MulIOp(self.ssa, rhs).result)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        rhs = self._coerce(other)
        if self.is_float:
            return ScalarValue(arith.DivFOp(self.ssa, rhs).result, is_float=True)
        return ScalarValue(arith.DivSIOp(self.ssa, rhs).result)

    def __mod__(self, other):
        rhs = self._coerce(other)
        if self.is_float:
            return ScalarValue(arith.RemFOp(self.ssa, rhs).result, is_float=True)
        return ScalarValue(arith.RemSIOp(self.ssa, rhs).result)

    # -- comparisons (return i1-typed ScalarValue) --

    def __lt__(self, other):
        rhs = self._coerce(other)
        if self.is_float:
            from mlir.dialects.arith import CmpFPredicate
            return ScalarValue(arith.CmpFOp(CmpFPredicate.OLT, self.ssa, rhs).result)
        return ScalarValue(arith.CmpIOp(CmpIPredicate.slt, self.ssa, rhs).result)

    def __le__(self, other):
        rhs = self._coerce(other)
        if self.is_float:
            from mlir.dialects.arith import CmpFPredicate
            return ScalarValue(arith.CmpFOp(CmpFPredicate.OLE, self.ssa, rhs).result)
        return ScalarValue(arith.CmpIOp(CmpIPredicate.sle, self.ssa, rhs).result)

    def __gt__(self, other):
        rhs = self._coerce(other)
        if self.is_float:
            from mlir.dialects.arith import CmpFPredicate
            return ScalarValue(arith.CmpFOp(CmpFPredicate.OGT, self.ssa, rhs).result)
        return ScalarValue(arith.CmpIOp(CmpIPredicate.sgt, self.ssa, rhs).result)

    def __ge__(self, other):
        rhs = self._coerce(other)
        if self.is_float:
            from mlir.dialects.arith import CmpFPredicate
            return ScalarValue(arith.CmpFOp(CmpFPredicate.OGE, self.ssa, rhs).result)
        return ScalarValue(arith.CmpIOp(CmpIPredicate.sge, self.ssa, rhs).result)

    def __eq__(self, other):
        rhs = self._coerce(other)
        if self.is_float:
            from mlir.dialects.arith import CmpFPredicate
            return ScalarValue(arith.CmpFOp(CmpFPredicate.OEQ, self.ssa, rhs).result)
        return ScalarValue(arith.CmpIOp(CmpIPredicate.eq, self.ssa, rhs).result)

    def __ne__(self, other):
        rhs = self._coerce(other)
        if self.is_float:
            from mlir.dialects.arith import CmpFPredicate
            return ScalarValue(arith.CmpFOp(CmpFPredicate.ONE, self.ssa, rhs).result)
        return ScalarValue(arith.CmpIOp(CmpIPredicate.ne, self.ssa, rhs).result)

    def __hash__(self):
        return id(self)

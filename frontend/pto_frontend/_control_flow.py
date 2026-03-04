"""Control-flow helpers: for_range and if_ context managers."""

from contextlib import contextmanager

from mlir.ir import InsertionPoint
from mlir.dialects import scf

from ._scalar import ScalarValue
from ._utils import ensure_index_ssa


@contextmanager
def for_range(start, end, step=1):
    """Emit an ``scf.for`` loop.

    Usage::

        with pto.for_range(0, M, 32) as i:
            ...  # i is a ScalarValue (index)
    """
    start_ssa = ensure_index_ssa(start)
    end_ssa = ensure_index_ssa(end)
    step_ssa = ensure_index_ssa(step)

    loop = scf.ForOp(start_ssa, end_ssa, step_ssa, [])
    ip = InsertionPoint(loop.body)
    ip.__enter__()
    try:
        yield ScalarValue(loop.induction_variable)
        scf.YieldOp([])
    finally:
        ip.__exit__(None, None, None)


class _Branch:
    """Helper for the ``has_else=True`` variant of :func:`if_`."""

    def __init__(self, block):
        self._block = block
        self._ip = None

    def __enter__(self):
        self._ip = InsertionPoint(self._block)
        self._ip.__enter__()
        return self

    def __exit__(self, *exc):
        scf.YieldOp([])
        self._ip.__exit__(*exc)


@contextmanager
def if_(condition, has_else=False):
    """Emit an ``scf.if`` operation.

    Simple (no else)::

        with pto.if_(cond):
            ...

    With else::

        with pto.if_(cond, has_else=True) as (then_br, else_br):
            with then_br:
                ...
            with else_br:
                ...
    """
    cond_ssa = condition.ssa if isinstance(condition, ScalarValue) else condition

    if_op = scf.IfOp(cond_ssa, [], hasElse=has_else)

    if has_else:
        yield _Branch(if_op.then_block), _Branch(if_op.else_block)
    else:
        ip = InsertionPoint(if_op.then_block)
        ip.__enter__()
        try:
            yield
            scf.YieldOp([])
        finally:
            ip.__exit__(None, None, None)

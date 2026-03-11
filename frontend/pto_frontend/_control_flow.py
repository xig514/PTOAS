"""Control-flow helpers: for_range, range, if_, and else_."""

import threading
from contextlib import contextmanager

from mlir.ir import InsertionPoint
from mlir.dialects import scf

from ._scalar import ScalarValue
from ._utils import ensure_index_ssa


# ---------------------------------------------------------------------------
#  Thread-local state for flat if_/else_ API
# ---------------------------------------------------------------------------

_tls = threading.local()


def _get_pending_if():
    """Return the pending (if_op, else_block) tuple, or None."""
    return getattr(_tls, "pending_if", None)


def _set_pending_if(if_op):
    _tls.pending_if = if_op


def _clear_pending_if():
    _tls.pending_if = None


def _finalize_pending_if():
    """If there is a pending if_ without a matching else_, fill the else block
    with an empty YieldOp so the SCF IfOp is well-formed."""
    pending = _get_pending_if()
    if pending is not None:
        if_op = pending
        with InsertionPoint(if_op.else_block):
            scf.YieldOp([])
        _clear_pending_if()


# ---------------------------------------------------------------------------
#  for_range  (context-manager style, kept for backward compatibility)
# ---------------------------------------------------------------------------

@contextmanager
def for_range(start, end, step=1):
    """Emit an ``scf.for`` loop.

    Usage::

        with pto.for_range(0, M, 32) as i:
            ...  # i is a ScalarValue (index)
    """
    _finalize_pending_if()
    start_ssa = ensure_index_ssa(start)
    end_ssa = ensure_index_ssa(end)
    step_ssa = ensure_index_ssa(step)

    loop = scf.ForOp(start_ssa, end_ssa, step_ssa, [])
    ip = InsertionPoint(loop.body)
    ip.__enter__()

    from ._sync_tracker import get_sync_tracker
    tracker = get_sync_tracker()
    if tracker:
        tracker.enter_loop(loop)

    try:
        yield ScalarValue(loop.induction_variable)

        if tracker:
            tracker.finalize_loop_body(loop)

        _finalize_pending_if()
        scf.YieldOp([])
    finally:
        ip.__exit__(None, None, None)

    if tracker:
        tracker.emit_loop_priming(loop)


# ---------------------------------------------------------------------------
#  range  (iterator style — Pythonic ``for i in pto.range(...)``)
# ---------------------------------------------------------------------------

class _RangeIterator:
    """Iterator that emits ``scf.for`` using Python's ``for`` protocol.

    During tracing the loop body executes exactly **once**.  The iterator
    creates the ``scf.ForOp`` in ``__iter__``, yields the induction variable
    on the first ``__next__``, and on the second call emits ``scf.YieldOp``,
    restores the insertion point, then raises ``StopIteration``.

    Usage::

        for i in pto.range(N):
            ...

        for i in pto.range(0, M, 64):
            ...
    """

    def __init__(self, start, stop, step):
        self._start = start
        self._stop = stop
        self._step = step
        self._loop = None
        self._ip = None
        self._yielded = False

    def __iter__(self):
        _finalize_pending_if()
        start_ssa = ensure_index_ssa(self._start)
        stop_ssa = ensure_index_ssa(self._stop)
        step_ssa = ensure_index_ssa(self._step)

        self._loop = scf.ForOp(start_ssa, stop_ssa, step_ssa, [])
        self._ip = InsertionPoint(self._loop.body)
        self._ip.__enter__()
        self._yielded = False

        from ._sync_tracker import get_sync_tracker
        tracker = get_sync_tracker()
        if tracker:
            tracker.enter_loop(self._loop)

        return self

    def __next__(self):
        if not self._yielded:
            self._yielded = True
            return ScalarValue(self._loop.induction_variable)

        # Loop body tracing is done — emit backward sync, then close region.
        _finalize_pending_if()

        from ._sync_tracker import get_sync_tracker
        tracker = get_sync_tracker()
        if tracker:
            tracker.finalize_loop_body(self._loop)

        scf.YieldOp([])
        self._ip.__exit__(None, None, None)

        if tracker:
            tracker.emit_loop_priming(self._loop)

        raise StopIteration


def range(*args):
    """Create an ``scf.for`` loop via Python's ``for`` statement.

    Signatures (mirror Python's built-in ``range``)::

        pto.range(stop)
        pto.range(start, stop)
        pto.range(start, stop, step)

    Each argument can be an ``int``, a :class:`ScalarValue`, or a
    :class:`DynVar`.

    Examples::

        for i in pto.range(N):
            ...

        for i in pto.range(0, M, 64):
            ...
    """
    if len(args) == 1:
        return _RangeIterator(0, args[0], 1)
    elif len(args) == 2:
        return _RangeIterator(args[0], args[1], 1)
    elif len(args) == 3:
        return _RangeIterator(args[0], args[1], args[2])
    else:
        raise TypeError(
            f"pto.range() takes 1 to 3 positional arguments, got {len(args)}"
        )


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

    **Flat API** (default, ``has_else`` not given)::

        with pto.if_(cond):
            ...        # then branch
        with pto.else_():
            ...        # else branch  (optional)

    **Nested API** (backward-compatible, ``has_else=True``)::

        with pto.if_(cond, has_else=True) as (then_br, else_br):
            with then_br:
                ...
            with else_br:
                ...
    """
    _finalize_pending_if()
    cond_ssa = condition.ssa if isinstance(condition, ScalarValue) else condition

    if has_else:
        # Legacy nested API — unchanged
        if_op = scf.IfOp(cond_ssa, [], hasElse=True)
        yield _Branch(if_op.then_block), _Branch(if_op.else_block)
    else:
        # Flat API: always create with hasElse=True so else_ can fill it later
        if_op = scf.IfOp(cond_ssa, [], hasElse=True)
        _set_pending_if(if_op)
        ip = InsertionPoint(if_op.then_block)
        ip.__enter__()
        try:
            yield
            scf.YieldOp([])
        finally:
            ip.__exit__(None, None, None)
        # Don't finalize yet — else_() may follow


@contextmanager
def else_():
    """Enter the else-branch of the immediately preceding ``pto.if_()``.

    Usage::

        with pto.if_(cond):
            ...        # then branch
        with pto.else_():
            ...        # else branch
    """
    pending = _get_pending_if()
    if pending is None:
        raise RuntimeError("pto.else_() used without a preceding pto.if_()")
    if_op = pending
    _clear_pending_if()
    ip = InsertionPoint(if_op.else_block)
    ip.__enter__()
    try:
        yield
        scf.YieldOp([])
    finally:
        ip.__exit__(None, None, None)

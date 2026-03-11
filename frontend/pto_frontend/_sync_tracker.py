"""Automatic pipeline synchronization for PTO kernels.

Tracks tile allocations and op emissions during kernel tracing.  Before each
op it checks for inter-pipeline data dependencies (including address-range
overlap) and emits ``set_flag`` / ``wait_flag`` ops inline.  At loop
boundaries it handles backward dependencies via MLIR insertion-point
manipulation.

Event allocation: EVENT_ID0 is used for all sync ops (both forward and
backward).  Since hardware set_flag/wait_flag match on (src, dst, event),
different pipe pairs don't conflict even when sharing the same event ID.
Forward syncs (set+wait immediate) net to zero.  Backward syncs use
set at body end, wait at body start, with priming before each loop and
drain waits after each loop to keep event counters bounded.
"""

import threading
from dataclasses import dataclass, field

from mlir.ir import InsertionPoint
from mlir.dialects import pto as _pto
from mlir.dialects.pto import PIPE, EVENT, PipeAttr, EventAttr


# ---------------------------------------------------------------------------
#  Thread-local tracker access
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def get_sync_tracker():
    """Return the current SyncTracker, or *None* if auto-sync is off."""
    return getattr(_thread_local, "sync_tracker", None)


def set_sync_tracker(tracker):
    _thread_local.sync_tracker = tracker


def clear_sync_tracker():
    _thread_local.sync_tracker = None


# ---------------------------------------------------------------------------
#  TileRegion — physical memory footprint of a tile
# ---------------------------------------------------------------------------

@dataclass
class TileRegion:
    addr_space: int          # AddressSpace enum value
    byte_offset: int         # starting byte address
    byte_size: int           # total size in bytes

    def overlaps(self, other):
        """True when two regions share at least one byte."""
        return (self.addr_space == other.addr_space
                and self.byte_offset < other.byte_offset + other.byte_size
                and other.byte_offset < self.byte_offset + self.byte_size)


# ---------------------------------------------------------------------------
#  BufferState — per-tile pipeline tracking
# ---------------------------------------------------------------------------

@dataclass
class BufferState:
    last_write_pipe: object = None       # PIPE enum or None
    last_read_pipes: set = field(default_factory=set)  # set of PIPE


# ---------------------------------------------------------------------------
#  LoopContext — first/last access per tile within one loop body
# ---------------------------------------------------------------------------

@dataclass
class LoopContext:
    loop_op: object                                      # scf.ForOp
    first_access: dict = field(default_factory=dict)     # tile_id -> PIPE
    last_access: dict = field(default_factory=dict)      # tile_id -> PIPE


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Single event for all sync — hardware distinguishes by (src, dst, event) triple
_SYNC_EVENT = EVENT.EVENT_ID0


# ---------------------------------------------------------------------------
#  Helpers — emit set_flag / wait_flag using the PTO dialect bindings
# ---------------------------------------------------------------------------

def _emit_set_flag(src_pipe, dst_pipe, event_id):
    src = PipeAttr.get(src_pipe)
    dst = PipeAttr.get(dst_pipe)
    evt = EventAttr.get(event_id)
    _pto.SetFlagOp(src_pipe=src, dst_pipe=dst, event_id=evt)


def _emit_wait_flag(src_pipe, dst_pipe, event_id):
    src = PipeAttr.get(src_pipe)
    dst = PipeAttr.get(dst_pipe)
    evt = EventAttr.get(event_id)
    _pto.WaitFlagOp(src_pipe=src, dst_pipe=dst, event_id=evt)


# ---------------------------------------------------------------------------
#  SyncTracker — the main auto-sync engine
# ---------------------------------------------------------------------------

class SyncTracker:

    def __init__(self):
        self._tile_regions = {}    # id(tile) -> TileRegion
        self._buffer_states = {}   # id(tile) -> BufferState
        self._loop_stack = []      # stack of LoopContext
        self._last_backward = []   # saved backward pairs for priming/drain

    # -- tile registration --------------------------------------------------

    def register_tile(self, tile, addr_space, byte_offset, byte_size):
        """Register a tile's physical memory region."""
        tid = id(tile)
        self._tile_regions[tid] = TileRegion(addr_space, byte_offset, byte_size)
        self._buffer_states[tid] = BufferState()

    # -- find overlapping tiles ---------------------------------------------

    def _overlapping_tiles(self, tile):
        """Return list of tile ids whose region overlaps *tile*'s region."""
        tid = id(tile)
        region = self._tile_regions.get(tid)
        if region is None:
            return []
        result = []
        for other_id, other_region in self._tile_regions.items():
            if region.overlaps(other_region):
                result.append(other_id)
        return result

    # -- record an op and emit needed syncs ---------------------------------

    def record_op(self, pipe, reads, writes):
        """Check dependencies and emit forward sync before the op, then update state."""
        emitted = set()  # (src_pipe, dst_pipe) already emitted this call

        # --- RAW: this op reads tiles that were written on a different pipe ---
        for tile in reads:
            for oid in self._overlapping_tiles(tile):
                state = self._buffer_states.get(oid)
                if state and state.last_write_pipe is not None and state.last_write_pipe != pipe:
                    pair = (state.last_write_pipe, pipe)
                    if pair not in emitted:
                        _emit_set_flag(*pair, _SYNC_EVENT)
                        _emit_wait_flag(*pair, _SYNC_EVENT)
                        emitted.add(pair)

        # --- WAW: this op writes tiles that were written on a different pipe ---
        for tile in writes:
            for oid in self._overlapping_tiles(tile):
                state = self._buffer_states.get(oid)
                if state and state.last_write_pipe is not None and state.last_write_pipe != pipe:
                    pair = (state.last_write_pipe, pipe)
                    if pair not in emitted:
                        _emit_set_flag(*pair, _SYNC_EVENT)
                        _emit_wait_flag(*pair, _SYNC_EVENT)
                        emitted.add(pair)

        # --- WAR: this op writes tiles that were read on a different pipe ---
        for tile in writes:
            for oid in self._overlapping_tiles(tile):
                state = self._buffer_states.get(oid)
                if state:
                    for read_pipe in state.last_read_pipes:
                        if read_pipe != pipe:
                            pair = (read_pipe, pipe)
                            if pair not in emitted:
                                _emit_set_flag(*pair, _SYNC_EVENT)
                                _emit_wait_flag(*pair, _SYNC_EVENT)
                                emitted.add(pair)

        # --- Update buffer state ---
        for tile in writes:
            for oid in self._overlapping_tiles(tile):
                st = self._buffer_states.get(oid)
                if st:
                    st.last_write_pipe = pipe
                    st.last_read_pipes.clear()

        for tile in reads:
            for oid in self._overlapping_tiles(tile):
                st = self._buffer_states.get(oid)
                if st:
                    st.last_read_pipes.add(pipe)

        # --- Update only the innermost loop context ---
        if self._loop_stack:
            ctx = self._loop_stack[-1]
            all_tiles = list(reads) + list(writes)
            for tile in all_tiles:
                for oid in self._overlapping_tiles(tile):
                    if oid not in ctx.first_access:
                        ctx.first_access[oid] = pipe
                    ctx.last_access[oid] = pipe

    # -- loop boundary hooks ------------------------------------------------

    def enter_loop(self, loop_op):
        """Push a new LoopContext when entering a for loop body."""
        self._loop_stack.append(LoopContext(loop_op=loop_op))

    def finalize_loop_body(self, loop_op):
        """Emit backward sync ops at the end/start of the loop body.

        Called just before ``scf.YieldOp`` is emitted.
        """
        if not self._loop_stack:
            self._last_backward = []
            return
        ctx = self._loop_stack.pop()

        backward_pairs = set()
        for tid in ctx.first_access:
            first = ctx.first_access[tid]
            last = ctx.last_access.get(tid)
            if last is not None and last != first:
                backward_pairs.add((last, first))

        if not backward_pairs:
            self._last_backward = []
            return

        sorted_pairs = sorted(backward_pairs, key=lambda p: (str(p[0]), str(p[1])))

        # Store for emit_loop_priming
        self._last_backward = sorted_pairs

        # --- Emit set_flag at end of body (current insertion point) ---
        for pair in sorted_pairs:
            _emit_set_flag(*pair, _SYNC_EVENT)

        # --- Emit wait_flag at start of body ---
        body_block = loop_op.body
        with InsertionPoint.at_block_begin(body_block):
            for pair in sorted_pairs:
                _emit_wait_flag(*pair, _SYNC_EVENT)

    def emit_loop_priming(self, loop_op):
        """Emit priming set_flag ops before the loop and drain wait_flag ops after.

        Each loop gets its own priming + drain to keep event counters bounded:
        - Priming (before loop): set_flag so the first iteration's backward
          waits don't deadlock.
        - Drain (after loop): wait_flag to consume the last iteration's
          backward sets, preventing counter accumulation in outer loops.
        """
        backward_pairs = self._last_backward
        self._last_backward = []

        if not backward_pairs:
            return

        # Emit priming set_flags BEFORE the loop
        with InsertionPoint(loop_op.operation):
            for pair in backward_pairs:
                _emit_set_flag(*pair, _SYNC_EVENT)

        # Emit drain wait_flags AFTER the loop (current insertion point)
        for pair in backward_pairs:
            _emit_wait_flag(*pair, _SYNC_EVENT)

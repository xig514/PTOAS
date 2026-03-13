"""Automatic pipeline synchronization for PTO kernels.

Tracks tile allocations and op emissions during kernel tracing.  Before each
op it checks for inter-pipeline data dependencies (including address-range
overlap) and emits ``set_flag`` / ``wait_flag`` ops inline.  At loop
boundaries it handles backward dependencies via MLIR insertion-point
manipulation.

Event allocation:
- Forward syncs for TileGroup tiles use a shared ``EventIdGroup`` (one per
  group size) with dynamic ``if/else`` selection based on the slot index.
  Non-TileGroup tiles use static EVENT_ID0.
- Backward syncs follow the same pattern: TileGroup backward deps use
  conditional set_flag/wait_flag, non-TileGroup use static EVENT_IDs.
- Priming/drain: for loop-varying slot indices (depending on the loop IV),
  emit set_flag/wait_flag for **every** EVENT_ID.  For loop-constant slot
  indices (not depending on the loop IV), skip priming/drain — the parent
  loop's priming already provides the needed credits.  For slot indices
  defined outside the loop body, emit conditional set_flag/wait_flag.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field

from mlir.ir import InsertionPoint
from mlir.dialects import scf
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
#  BufferState — per-slot pipeline tracking
# ---------------------------------------------------------------------------

@dataclass
class BufferState:
    last_write_pipe: object = None       # PIPE enum or None
    last_read_pipes: set = field(default_factory=set)  # set of PIPE


# ---------------------------------------------------------------------------
#  LoopContext — first/last access per slot key within one loop body
# ---------------------------------------------------------------------------

@dataclass
class LoopContext:
    loop_op: object                                      # scf.ForOp
    first_access: dict = field(default_factory=dict)     # slot_key -> PIPE
    last_access: dict = field(default_factory=dict)      # slot_key -> PIPE
    # slot_key -> (group_id, slot_idx_ScalarValue, group_size)
    tile_group_info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Pool of event IDs for both forward and backward sync
_EVENT_IDS = [
    EVENT.EVENT_ID0, EVENT.EVENT_ID1, EVENT.EVENT_ID2, EVENT.EVENT_ID3,
    EVENT.EVENT_ID4, EVENT.EVENT_ID5, EVENT.EVENT_ID6, EVENT.EVENT_ID7,
]


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
#  Helpers — dynamic (conditional) sync emission for TileGroup tiles
# ---------------------------------------------------------------------------

def _emit_dynamic_set_flag(src_pipe, dst_pipe, event_group, slot_idx):
    """Emit set_flag with dynamic EVENT_ID selection via if/else chain."""
    from ._event_group import _DynamicEventSelection
    selection = _DynamicEventSelection(event_group._events, slot_idx)
    selection.emit_conditional(
        lambda evt: _emit_set_flag(src_pipe, dst_pipe, evt))


def _emit_dynamic_wait_flag(src_pipe, dst_pipe, event_group, slot_idx):
    """Emit wait_flag with dynamic EVENT_ID selection via if/else chain."""
    from ._event_group import _DynamicEventSelection
    selection = _DynamicEventSelection(event_group._events, slot_idx)
    selection.emit_conditional(
        lambda evt: _emit_wait_flag(src_pipe, dst_pipe, evt))


# ---------------------------------------------------------------------------
#  Helpers — insert conditional wait_flags after slot_idx definition
# ---------------------------------------------------------------------------

def _insert_conditional_waits_after_def(body_block, conditional_groups):
    """Insert conditional backward wait_flags after slot_idx definitions.

    For each conditional group, the wait_flag ``if/else`` chain references
    ``slot_idx.ssa``.  To satisfy SSA dominance, the wait_flag must appear
    *after* the defining op of ``slot_idx.ssa`` in the block.

    Groups sharing the same defining op are batched so that all their
    wait_flags are inserted at the same point.
    """
    from collections import defaultdict as _dd

    # Map defining-op hash → list of (src, dst, event_group, slot_idx)
    by_def_op = _dd(list)
    for item in conditional_groups:
        _s, _d, _eg, slot_idx = item
        def_op = slot_idx.ssa.owner
        by_def_op[hash(def_op)].append((item, def_op))

    # For each defining op, find the next operation in the block and insert
    # the wait_flags before it (i.e. right after the defining op).
    block_ops = list(body_block.operations)
    op_to_idx = {}
    for i, op in enumerate(block_ops):
        op_to_idx[hash(op.operation)] = i

    for def_op_hash, entries in by_def_op.items():
        def_op = entries[0][1]  # the actual defining operation
        idx = op_to_idx.get(hash(def_op))
        if idx is not None and idx + 1 < len(block_ops):
            # Insert before the next op (= after the defining op)
            next_op = block_ops[idx + 1]
            with InsertionPoint(next_op.operation):
                for (s, d, eg, slot_idx), _ in entries:
                    _emit_dynamic_wait_flag(s, d, eg, slot_idx)
        else:
            # Fallback: defining op not found in this block or is last.
            # This can happen if slot_idx is defined in a parent block
            # (e.g., block argument of an outer loop).  In that case
            # block-begin is safe.
            with InsertionPoint.at_block_begin(body_block):
                for (s, d, eg, slot_idx), _ in entries:
                    _emit_dynamic_wait_flag(s, d, eg, slot_idx)


# ---------------------------------------------------------------------------
#  SyncTracker — the main auto-sync engine
# ---------------------------------------------------------------------------

class SyncTracker:

    def __init__(self):
        self._tile_regions = {}    # id(tile) -> TileRegion
        self._buffer_states = {}   # slot_key -> BufferState
        self._loop_stack = []      # stack of LoopContext
        self._last_backward = ([], [])  # (unconditional, conditional_groups)
        self._if_stack = []        # stack of (pre_if_states, then_states_or_None)
        # Lazy EventIdGroup allocation: group_size -> EventIdGroup
        self._auto_event_groups = {}
        self._auto_event_counter = 0

    # -- tile registration --------------------------------------------------

    def register_tile(self, tile, addr_space, byte_offset, byte_size):
        """Register a tile's physical memory region for overlap checking."""
        tid = id(tile)
        self._tile_regions[tid] = TileRegion(addr_space, byte_offset, byte_size)

    # -- slot key helpers ---------------------------------------------------

    def _get_slot_key(self, tile):
        """Get a unique key for the buffer slot accessed by this tile.

        For non-TileGroup tiles: ``('tile', id(tile))``.
        For TileGroup tiles: ``('group_slot', group_id, slot_idx_ssa)``
        so that different accesses to the same slot share the same key.
        """
        group = getattr(tile, '_group_tiles', None)
        if group is None:
            return ('tile', id(tile))
        group_id = id(group[0])  # Stable group identifier
        group_idx = getattr(tile, '_group_idx', None)
        if group_idx is None:
            return ('tile', id(tile))
        return ('group_slot', group_id, group_idx.ssa)

    def _get_auto_event_group(self, group_size):
        """Get or allocate an EventIdGroup for a given group size.

        All TileGroups of the same size share one EventIdGroup.  This is
        safe because slots within a group are non-overlapping, and
        EVENT_IDs are distinguished per pipe pair by hardware.
        """
        if group_size in self._auto_event_groups:
            return self._auto_event_groups[group_size]

        from ._event_group import EventIdGroup
        events = _EVENT_IDS[self._auto_event_counter:
                            self._auto_event_counter + group_size]
        if len(events) < group_size:
            raise RuntimeError(
                f"Insufficient EVENT_IDs: need {group_size}, "
                f"only {len(_EVENT_IDS) - self._auto_event_counter} remaining")
        self._auto_event_counter += group_size
        eg = EventIdGroup(events)
        self._auto_event_groups[group_size] = eg
        return eg

    def _get_tg_info(self, tile):
        """Return (event_group, slot_idx) for a TileGroup tile, or None."""
        group = getattr(tile, '_group_tiles', None)
        if group is None:
            return None
        slot_idx = getattr(tile, '_group_idx', None)
        if slot_idx is None:
            return None
        eg = self._get_auto_event_group(len(group))
        return (eg, slot_idx)

    @staticmethod
    def _is_inside_loop(ssa_value, body_block):
        """Check if *ssa_value* is defined inside *body_block* (or nested).

        Uses hash-based comparison for reliable matching across
        MLIR Python binding wrapper objects.
        """
        try:
            target_hash = hash(ssa_value.owner)
        except Exception:
            return False

        def _search_block(block):
            for op in block.operations:
                try:
                    if hash(op.operation) == target_hash:
                        return True
                except Exception:
                    pass
                for region in op.regions:
                    for sub_block in region:
                        if _search_block(sub_block):
                            return True
            return False

        return _search_block(body_block)

    # -- record an op and emit needed syncs ---------------------------------

    def record_op(self, pipe, reads, writes):
        """Check dependencies and emit forward sync before the op, then
        update buffer state.

        Uses slot keys to avoid expanding TileGroup tiles.  For TileGroup
        tiles, emits dynamic ``if/else`` EVENT_ID selection.
        """
        # Collect dependencies per pipe pair.
        # Value: (event_group, slot_idx) for TileGroup, or None for static.
        emitted = {}  # (src_pipe, dst_pipe) -> tg_info or None

        # --- RAW: this op reads tiles written on a different pipe ---
        for tile in reads:
            slot_key = self._get_slot_key(tile)
            state = self._buffer_states.get(slot_key)
            if (state and state.last_write_pipe is not None
                    and state.last_write_pipe != pipe):
                pair = (state.last_write_pipe, pipe)
                if pair not in emitted:
                    emitted[pair] = self._get_tg_info(tile)

        # --- WAW: this op writes tiles written on a different pipe ---
        for tile in writes:
            slot_key = self._get_slot_key(tile)
            state = self._buffer_states.get(slot_key)
            if (state and state.last_write_pipe is not None
                    and state.last_write_pipe != pipe):
                pair = (state.last_write_pipe, pipe)
                if pair not in emitted:
                    emitted[pair] = self._get_tg_info(tile)

        # --- WAR: this op writes tiles read on a different pipe ---
        for tile in writes:
            slot_key = self._get_slot_key(tile)
            state = self._buffer_states.get(slot_key)
            if state:
                for read_pipe in state.last_read_pipes:
                    if read_pipe != pipe:
                        pair = (read_pipe, pipe)
                        if pair not in emitted:
                            emitted[pair] = self._get_tg_info(tile)

        # --- Emit all collected syncs ---
        for (src, dst), tg_info in emitted.items():
            if tg_info is not None:
                eg, slot_idx = tg_info
                _emit_dynamic_set_flag(src, dst, eg, slot_idx)
                _emit_dynamic_wait_flag(src, dst, eg, slot_idx)
            else:
                _emit_set_flag(src, dst, _EVENT_IDS[0])
                _emit_wait_flag(src, dst, _EVENT_IDS[0])

        # --- Update buffer state using slot keys ---
        for tile in writes:
            slot_key = self._get_slot_key(tile)
            st = self._buffer_states.get(slot_key)
            if not st:
                st = BufferState()
                self._buffer_states[slot_key] = st
            st.last_write_pipe = pipe
            st.last_read_pipes.clear()

        for tile in reads:
            slot_key = self._get_slot_key(tile)
            st = self._buffer_states.get(slot_key)
            if not st:
                st = BufferState()
                self._buffer_states[slot_key] = st
            st.last_read_pipes.add(pipe)

        # --- Update only the innermost loop context ---
        if self._loop_stack:
            ctx = self._loop_stack[-1]
            for tile in list(reads) + list(writes):
                slot_key = self._get_slot_key(tile)
                if slot_key not in ctx.first_access:
                    ctx.first_access[slot_key] = pipe
                ctx.last_access[slot_key] = pipe

                # Track TileGroup info for backward sync
                group = getattr(tile, '_group_tiles', None)
                if group is not None:
                    group_id = id(group[0])
                    slot_idx = getattr(tile, '_group_idx', None)
                    if slot_idx is not None and slot_key not in ctx.tile_group_info:
                        ctx.tile_group_info[slot_key] = (
                            group_id, slot_idx, len(group))

    # -- loop boundary hooks ------------------------------------------------

    def enter_loop(self, loop_op):
        """Push a new LoopContext when entering a for loop body."""
        self._loop_stack.append(LoopContext(loop_op=loop_op))

    def finalize_loop_body(self, loop_op):
        """Emit backward sync ops at the end/start of the loop body.

        For TileGroup slot keys, emits conditional set_flag/wait_flag
        using the auto-allocated EventIdGroup.  Deduplicates by
        (pipe_pair, slot_idx_ssa, group_size) so each pipe pair gets at
        most one conditional sync per distinct slot index.
        """
        if not self._loop_stack:
            self._last_backward = ([], [])
            return
        ctx = self._loop_stack.pop()

        # Collect backward dependencies: slot keys where last_pipe != first_pipe
        backward_deps = []
        for slot_key in ctx.first_access:
            first = ctx.first_access[slot_key]
            last = ctx.last_access.get(slot_key)
            if last is not None and last != first:
                backward_deps.append((last, first, slot_key))

        if not backward_deps:
            self._last_backward = ([], [])
            return

        # Group by pipe pair
        pair_to_keys = defaultdict(list)
        for src, dst, slot_key in backward_deps:
            pair_to_keys[(src, dst)].append(slot_key)

        unconditional_events = []  # list of (src, dst, event_id)
        conditional_groups = []    # list of (src, dst, event_group, slot_idx)

        for (src, dst) in sorted(pair_to_keys,
                                 key=lambda p: (str(p[0]), str(p[1]))):
            slot_keys = pair_to_keys[(src, dst)]

            # Deduplicate TileGroup backward deps:
            # same (pipe_pair, slot_idx_ssa, group_size) → one conditional sync
            seen_dynamic = set()  # (id(slot_idx.ssa), group_size)
            has_static = False

            for sk in slot_keys:
                tg_info = ctx.tile_group_info.get(sk)
                if tg_info is not None:
                    group_id, slot_idx, group_size = tg_info
                    dedup_key = (id(slot_idx.ssa), group_size)
                    if dedup_key not in seen_dynamic:
                        seen_dynamic.add(dedup_key)
                        eg = self._get_auto_event_group(group_size)
                        conditional_groups.append((src, dst, eg, slot_idx))
                else:
                    has_static = True

            if has_static:
                unconditional_events.append((src, dst, _EVENT_IDS[0]))

        # Store for emit_loop_priming
        self._last_backward = (unconditional_events, conditional_groups)

        # --- Emit set_flag at end of body (current insertion point) ---
        for (s, d, e) in unconditional_events:
            _emit_set_flag(s, d, e)
        for (s, d, eg, slot_idx) in conditional_groups:
            _emit_dynamic_set_flag(s, d, eg, slot_idx)

        # --- Emit wait_flag near start of body ---
        body_block = loop_op.body
        # Unconditional wait_flags can go at block begin (no SSA deps).
        if unconditional_events:
            with InsertionPoint.at_block_begin(body_block):
                for (s, d, e) in unconditional_events:
                    _emit_wait_flag(s, d, e)
        # Conditional wait_flags must go AFTER the slot_idx definition
        # to satisfy SSA dominance.  Group by defining op to batch.
        if conditional_groups:
            _insert_conditional_waits_after_def(
                body_block, conditional_groups)

    @staticmethod
    def _depends_on_loop_iv(ssa_value, body_block):
        """Check if *ssa_value* transitively depends on any block argument
        of *body_block* (i.e. the loop induction variable or loop-carried
        values).  If not, the value is constant across iterations.
        """
        block_args = list(body_block.arguments)
        if not block_args:
            return False
        block_arg_hashes = {}
        for ba in block_args:
            try:
                block_arg_hashes[hash(ba)] = ba
            except Exception:
                pass
        visited_hashes = set()
        stack = [ssa_value]
        while stack:
            val = stack.pop()
            try:
                h = hash(val)
            except Exception:
                continue
            if h in visited_hashes:
                continue
            visited_hashes.add(h)
            if h in block_arg_hashes and val == block_arg_hashes[h]:
                return True
            try:
                op = val.owner
                if hasattr(op, 'operands'):
                    for operand in op.operands:
                        stack.append(operand)
            except Exception:
                pass
        return False

    def emit_loop_priming(self, loop_op):
        """Emit priming set_flag ops before the loop and drain wait_flag
        ops after.

        For conditional groups whose slot_idx *varies* across iterations
        (depends on the loop induction variable), emits set_flag/wait_flag
        for **every** EVENT_ID unconditionally.

        For groups whose slot_idx is *constant* across iterations (does
        not depend on the loop IV), skips priming/drain entirely — the
        parent loop's priming already provides the needed initial credits.

        For groups whose slot_idx is defined *outside* the loop body,
        emits conditional set_flag/wait_flag.
        """
        unconditional_events, conditional_groups = self._last_backward
        self._last_backward = ([], [])

        if not unconditional_events and not conditional_groups:
            return

        body_block = loop_op.body
        varying_groups = []    # slot_idx varies across iterations → unconditional
        constant_groups = []   # slot_idx is constant → skip priming
        external_groups = []   # slot_idx defined outside → conditional
        for item in conditional_groups:
            _s, _d, _eg, slot_idx = item
            if not self._is_inside_loop(slot_idx.ssa, body_block):
                external_groups.append(item)
            elif self._depends_on_loop_iv(slot_idx.ssa, body_block):
                varying_groups.append(item)
            else:
                constant_groups.append(item)

        # constant_groups: skip priming/drain — outer loop covers these.

        # Priming: set_flags BEFORE the loop
        with InsertionPoint(loop_op.operation):
            for (s, d, e) in unconditional_events:
                _emit_set_flag(s, d, e)
            for (s, d, eg, _slot_idx) in varying_groups:
                for evt in eg._events:
                    _emit_set_flag(s, d, evt)
            for (s, d, eg, slot_idx) in external_groups:
                _emit_dynamic_set_flag(s, d, eg, slot_idx)

        # Drain: wait_flags AFTER the loop
        for (s, d, e) in unconditional_events:
            _emit_wait_flag(s, d, e)
        for (s, d, eg, _slot_idx) in varying_groups:
            for evt in eg._events:
                _emit_wait_flag(s, d, evt)
        for (s, d, eg, slot_idx) in external_groups:
            _emit_dynamic_wait_flag(s, d, eg, slot_idx)

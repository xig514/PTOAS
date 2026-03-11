"""Automatic pipeline synchronization for PTO kernels.

Tracks tile allocations and op emissions during kernel tracing.  Before each
op it checks for inter-pipeline data dependencies (including address-range
overlap) and emits ``set_flag`` / ``wait_flag`` ops inline.  At loop
boundaries it handles backward dependencies via MLIR insertion-point
manipulation.

Event allocation:
- Forward syncs use per-pipe-pair EVENT_IDs to avoid conflicts between
  different pipeline pairs (e.g., PIPE_MTE2→PIPE_V uses a different EVENT_ID
  than PIPE_V→PIPE_MTE3).
- Backward syncs for TileGroup tiles share one EVENT_ID per group since
  different buffer slots are mutually exclusive. Non-overlapping tiles get
  independent EVENT_IDs.
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
    tile_group_info: dict = field(default_factory=dict)  # tile_id -> (slot, group_id)


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Pool of event IDs for both forward and backward sync
_EVENT_IDS = [
    EVENT.EVENT_ID0, EVENT.EVENT_ID1, EVENT.EVENT_ID2, EVENT.EVENT_ID3,
    EVENT.EVENT_ID4, EVENT.EVENT_ID5, EVENT.EVENT_ID6, EVENT.EVENT_ID7,
]


# ---------------------------------------------------------------------------
#  Helpers — clustering non-overlapping tile groups
# ---------------------------------------------------------------------------

def _cluster_by_overlap(tile_ids, tile_regions):
    """Partition *tile_ids* into clusters where tiles within a cluster overlap.

    Two tiles in different clusters are guaranteed non-overlapping, so they can
    safely use separate EVENT_IDs for backward sync.

    Returns a list of lists of tile IDs.
    """
    clusters = []  # list of (set_of_tile_ids, merged_region)
    for tid in tile_ids:
        region = tile_regions.get(tid)
        if region is None:
            continue
        merged = False
        for cluster_tids, cluster_region in clusters:
            if region.overlaps(cluster_region):
                cluster_tids.add(tid)
                # Expand the cluster bounding box
                new_start = min(cluster_region.byte_offset, region.byte_offset)
                new_end = max(
                    cluster_region.byte_offset + cluster_region.byte_size,
                    region.byte_offset + region.byte_size,
                )
                cluster_region.byte_offset = new_start
                cluster_region.byte_size = new_end - new_start
                merged = True
                break
        if not merged:
            clusters.append(({tid}, TileRegion(
                region.addr_space, region.byte_offset, region.byte_size)))
    return [list(c[0]) for c in clusters]


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
#  Helpers — conditional sync emission for TileGroup backward sync
# ---------------------------------------------------------------------------

def _emit_conditional_sync(group_idx, slot_events, emit_fn):
    """Emit sync ops inside ``scf.if`` chain, one branch per buffer slot.

    *group_idx* is a ScalarValue with the runtime buffer index.
    *slot_events* maps slot number to a list of ``(src, dst, event)`` triples.
    *emit_fn* is either ``_emit_set_flag`` or ``_emit_wait_flag``.
    """
    slots = sorted(slot_events.keys())
    _emit_cond_chain(group_idx, slots, slot_events, emit_fn)


def _emit_cond_chain(group_idx, slots, slot_events, emit_fn):
    """Recursively build ``scf.if`` chain for per-slot sync emission."""
    if len(slots) == 0:
        return
    if len(slots) == 1:
        # Last slot: emit unconditionally (it's the else/fallback case)
        for (s, d, e) in slot_events[slots[0]]:
            emit_fn(s, d, e)
        return

    slot = slots[0]
    # group_idx is a ScalarValue, need to compare with constant
    from ._ir_builder import get_builder
    builder = get_builder()
    slot_const = builder.constant_index(slot)
    from ._scalar import ScalarValue
    cond_scalar = ScalarValue(group_idx.ssa) == slot_const
    cond = cond_scalar.ssa

    if_op = scf.IfOp(cond, [], hasElse=True)
    with InsertionPoint(if_op.then_block):
        for (s, d, e) in slot_events[slot]:
            emit_fn(s, d, e)
        scf.YieldOp([])
    with InsertionPoint(if_op.else_block):
        _emit_cond_chain(group_idx, slots[1:], slot_events, emit_fn)
        scf.YieldOp([])


# ---------------------------------------------------------------------------
#  SyncTracker — the main auto-sync engine
# ---------------------------------------------------------------------------

class SyncTracker:

    def __init__(self):
        self._tile_regions = {}    # id(tile) -> TileRegion
        self._buffer_states = {}   # id(tile) -> BufferState
        self._loop_stack = []      # stack of LoopContext
        self._last_backward = ([], [])  # (unconditional, conditional_groups)
        self._forward_event_map = {}   # (src_pipe, dst_pipe) -> EVENT_ID
        self._forward_event_counter = 0

    # -- tile registration --------------------------------------------------

    def register_tile(self, tile, addr_space, byte_offset, byte_size):
        """Register a tile's physical memory region."""
        tid = id(tile)
        self._tile_regions[tid] = TileRegion(addr_space, byte_offset, byte_size)
        self._buffer_states[tid] = BufferState()

    # -- forward event allocation -------------------------------------------

    def _tiles_overlap(self, tid1, tid2):
        """Check if two tiles overlap in memory."""
        region1 = self._tile_regions.get(tid1)
        region2 = self._tile_regions.get(tid2)
        if region1 is None or region2 is None:
            return False
        return region1.overlaps(region2)

    def _get_forward_event(self, src_pipe, dst_pipe, tile_ids):
        """Allocate or retrieve EVENT_ID for forward sync between two pipes.

        For the same pipe pair, if buffers don't overlap, different EVENT_IDs
        can be used to enable better parallelism.

        Args:
            src_pipe: Source pipeline
            dst_pipe: Destination pipeline
            tile_ids: Set of tile IDs involved in this sync

        Returns:
            EVENT_ID to use for this sync
        """
        pair = (src_pipe, dst_pipe)

        # Get or create list of (event_id, tile_id_set) for this pipe pair
        if pair not in self._forward_event_map:
            self._forward_event_map[pair] = []

        event_list = self._forward_event_map[pair]

        # Try to find an existing event whose tiles don't overlap with current tiles
        for event_id, existing_tiles in event_list:
            has_overlap = False
            for tid1 in tile_ids:
                for tid2 in existing_tiles:
                    if self._tiles_overlap(tid1, tid2):
                        has_overlap = True
                        break
                if has_overlap:
                    break

            if not has_overlap:
                # Can reuse this event_id, add current tiles to its set
                existing_tiles.update(tile_ids)
                return event_id

        # Need a new event_id
        event_id = _EVENT_IDS[self._forward_event_counter % len(_EVENT_IDS)]
        self._forward_event_counter += 1
        event_list.append((event_id, set(tile_ids)))

        return event_id

    # -- find overlapping tiles ---------------------------------------------

    def _overlapping_tiles(self, tile):
        """Return list of tile ids whose region overlaps *tile*'s region.

        If *tile* has ``_group_tiles`` (TileGroup selection), expand to
        all underlying tiles for conservative dependency analysis.
        """
        group = getattr(tile, '_group_tiles', None)
        if group is not None:
            result = []
            seen = set()
            for t in group:
                for oid in self._overlapping_tiles(t):
                    if oid not in seen:
                        result.append(oid)
                        seen.add(oid)
            return result

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
        emitted = {}  # (src_pipe, dst_pipe) -> set of tile_ids already synced

        # --- RAW: this op reads tiles that were written on a different pipe ---
        for tile in reads:
            for oid in self._overlapping_tiles(tile):
                state = self._buffer_states.get(oid)
                if state and state.last_write_pipe is not None and state.last_write_pipe != pipe:
                    pair = (state.last_write_pipe, pipe)
                    if pair not in emitted:
                        emitted[pair] = set()
                    emitted[pair].add(oid)

        # --- WAW: this op writes tiles that were written on a different pipe ---
        for tile in writes:
            for oid in self._overlapping_tiles(tile):
                state = self._buffer_states.get(oid)
                if state and state.last_write_pipe is not None and state.last_write_pipe != pipe:
                    pair = (state.last_write_pipe, pipe)
                    if pair not in emitted:
                        emitted[pair] = set()
                    emitted[pair].add(oid)

        # --- WAR: this op writes tiles that were read on a different pipe ---
        for tile in writes:
            for oid in self._overlapping_tiles(tile):
                state = self._buffer_states.get(oid)
                if state:
                    for read_pipe in state.last_read_pipes:
                        if read_pipe != pipe:
                            pair = (read_pipe, pipe)
                            if pair not in emitted:
                                emitted[pair] = set()
                            emitted[pair].add(oid)

        # Emit all collected syncs
        for (src, dst), tile_ids in emitted.items():
            event_id = self._get_forward_event(src, dst, tile_ids)
            _emit_set_flag(src, dst, event_id)
            _emit_wait_flag(src, dst, event_id)

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

                # Track TileGroup membership - use first tile's id as group identifier
                group = getattr(tile, '_group_tiles', None)
                if group is not None:
                    group_id = id(group[0])  # Use first tile's id as stable group identifier
                    for slot, t in enumerate(group):
                        tid = id(t)
                        if tid not in ctx.tile_group_info:
                            ctx.tile_group_info[tid] = (slot, group_id)

    # -- loop boundary hooks ------------------------------------------------

    def enter_loop(self, loop_op):
        """Push a new LoopContext when entering a for loop body."""
        self._loop_stack.append(LoopContext(loop_op=loop_op))

    def finalize_loop_body(self, loop_op):
        """Emit backward sync ops at the end/start of the loop body.

        Called just before ``scf.YieldOp`` is emitted.

        For TileGroup tiles, merge all tiles from the same group to share one
        EVENT_ID since different buffer slots are mutually exclusive.
        """
        if not self._loop_stack:
            self._last_backward = ([], [])
            return
        ctx = self._loop_stack.pop()

        # Collect per-tile backward info: (last_pipe, first_pipe, tile_id)
        tile_backward = []
        for tid in ctx.first_access:
            first = ctx.first_access[tid]
            last = ctx.last_access.get(tid)
            if last is not None and last != first:
                tile_backward.append((last, first, tid))

        if not tile_backward:
            self._last_backward = ([], [])
            return

        event_counter = 0
        unconditional_events = []

        if tile_backward:
            groups = defaultdict(list)
            for last, first, tid in tile_backward:
                groups[(last, first)].append(tid)

            # Build a map from tile to its TileGroup identifier
            tile_to_group_id = {}
            for tid, (slot, group_id) in ctx.tile_group_info.items():
                tile_to_group_id[tid] = group_id

            for (src, dst) in sorted(groups, key=lambda p: (str(p[0]), str(p[1]))):
                tile_ids = groups[(src, dst)]

                # Merge tiles from the same TileGroup - only keep one representative per group
                group_id_to_representative = {}
                representative_tiles = []
                for tid in tile_ids:
                    gid = tile_to_group_id.get(tid)
                    if gid is not None:
                        # This tile belongs to a TileGroup
                        if gid not in group_id_to_representative:
                            group_id_to_representative[gid] = tid
                            representative_tiles.append(tid)
                        # else: skip, already have a representative for this group
                    else:
                        # Not a TileGroup tile, include it
                        representative_tiles.append(tid)

                # Cluster the representative tiles
                clusters = _cluster_by_overlap(
                    representative_tiles, self._tile_regions)
                for _ in clusters:
                    unconditional_events.append(
                        (src, dst, _EVENT_IDS[event_counter % len(_EVENT_IDS)]))
                    event_counter += 1

        # Store for emit_loop_priming (no conditional groups)
        self._last_backward = (unconditional_events, [])

        # --- Emit set_flag at end of body (current insertion point) ---
        for (s, d, e) in unconditional_events:
            _emit_set_flag(s, d, e)

        # --- Emit wait_flag at start of body ---
        body_block = loop_op.body
        with InsertionPoint.at_block_begin(body_block):
            for (s, d, e) in unconditional_events:
                _emit_wait_flag(s, d, e)

    def emit_loop_priming(self, loop_op):
        """Emit priming set_flag ops before the loop and drain wait_flag ops after.

        Priming and drain are **unconditional** to ensure the first iteration
        doesn't deadlock and the last iteration's events are drained.
        """
        ungrouped_events, _ = self._last_backward
        self._last_backward = ([], [])

        if not ungrouped_events:
            return

        # Priming: set_flags BEFORE the loop
        with InsertionPoint(loop_op.operation):
            for (s, d, e) in ungrouped_events:
                _emit_set_flag(s, d, e)

        # Drain: wait_flags AFTER the loop
        for (s, d, e) in ungrouped_events:
            _emit_wait_flag(s, d, e)

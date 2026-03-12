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
    group_size_map: dict = field(default_factory=dict)   # group_id -> int


# ---------------------------------------------------------------------------
#  TileGroup Access Tracking (for auto-sync)
# ---------------------------------------------------------------------------

@dataclass
class TileGroupAccessInfo:
    """Records information about a TileGroup access."""
    group_id: int                # Stable identifier: id(group[0])
    slot_index_ssa: object       # SSA value of the slot index expression
    access_op: object            # The operation accessing this TileGroup
    pipe: object                 # Pipeline of the operation
    is_write: bool               # True if write access, False if read
    tile: object                 # The actual Tile object accessed


@dataclass
class TileGroupInfo:
    """Tracks all accesses to a specific TileGroup."""
    group_id: int                                      # Stable identifier
    size: int                                          # Number of slots
    accesses: list = field(default_factory=list)       # List[TileGroupAccessInfo]
    pipeline_pairs: set = field(default_factory=set)   # Set[(src_pipe, dst_pipe)]
    event_group: object = None                         # EventIdGroup (to be assigned)


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
        self._buffer_states = {}   # slot_key -> BufferState
        self._loop_stack = []      # stack of LoopContext
        self._last_backward = ([], [])  # (unconditional, conditional_groups)
        self._forward_event_map = {}   # (src_pipe, dst_pipe) -> EVENT_ID
        self._forward_event_counter = 0
        self._if_stack = []        # stack of (pre_if_states, then_states_or_None)
        # TileGroup tracking for auto-sync
        self._tile_group_registry = {}  # group_id -> TileGroupInfo
        self._current_access_op = None  # Track current operation for access recording

    # -- tile registration --------------------------------------------------

    def register_tile(self, tile, addr_space, byte_offset, byte_size):
        """Register a tile's physical memory region.

        SIMPLIFIED VERSION: Only use tile ID as key, no slot key.
        """
        tid = id(tile)
        self._tile_regions[tid] = TileRegion(addr_space, byte_offset, byte_size)
        # Initialize buffer state for this tile
        if tid not in self._buffer_states:
            self._buffer_states[tid] = BufferState()

    # -- TileGroup access tracking (for auto-sync) ----------------------------

    def _get_slot_key(self, tile):
        """Get a unique key for the slot accessed by this tile.

        For non-TileGroup tiles, the key is the tile ID.
        For TileGroup tiles, the key is (group_id, idx_ssa) so that
        different accesses to the same slot share the same key.
        """
        group = getattr(tile, '_group_tiles', None)
        if group is None:
            return ('tile', id(tile))
        group_id = id(group[0])  # Stable group identifier
        group_idx = getattr(tile, '_group_idx', None)
        if group_idx is None:
            return ('tile', id(tile))
        # Use SSA value as slot identifier within the group
        return ('group_slot', group_id, group_idx.ssa)

    def record_tilegroup_access(self, tile, pipe, is_write):
        """Record an access to a TileGroup tile for auto-sync analysis.

        Args:
            tile: The Tile object being accessed
            pipe: PIPE enum for the operation
            is_write: True if this is a write access, False for read
        """
        group = getattr(tile, '_group_tiles', None)
        if group is None:
            return  # Not a TileGroup tile

        group_idx = getattr(tile, '_group_idx', None)
        if group_idx is None:
            return  # Static selection (not dynamic)

        # Get or create TileGroupInfo
        group_id = id(group[0])  # Stable identifier
        if group_id not in self._tile_group_registry:
            self._tile_group_registry[group_id] = TileGroupInfo(
                group_id=group_id,
                size=len(group)
            )

        # Record this access
        access_info = TileGroupAccessInfo(
            group_id=group_id,
            slot_index_ssa=group_idx.ssa,  # SSA value of the index
            access_op=self._current_access_op,
            pipe=pipe,
            is_write=is_write,
            tile=tile
        )

        self._tile_group_registry[group_id].accesses.append(access_info)

    def get_tilegroup_info(self, group_id):
        """Get TileGroupInfo for a given group_id."""
        return self._tile_group_registry.get(group_id)

    def get_all_tilegroups(self):
        """Get all registered TileGroups."""
        return list(self._tile_group_registry.values())

    def analyze_dependencies(self):
        """Analyze dependencies between TileGroup accesses (Phase 2).

        Returns:
            Dict[(src_pipe, dst_pipe)] -> List[dependency_info]
            where dependency_info = {
                'type': 'RAW'/'WAW'/'WAR',
                'src_access': TileGroupAccessInfo,
                'dst_access': TileGroupAccessInfo,
                'is_slot_specific': bool
            }
        """
        dependencies = defaultdict(list)

        for group_info in self._tile_group_registry.values():
            accesses = group_info.accesses

            # Find RAW dependencies (write -> read)
            for i, write_access in enumerate(accesses):
                if not write_access.is_write:
                    continue

                for j in range(i + 1, len(accesses)):
                    read_access = accesses[j]
                    if read_access.is_write:
                        continue  # Skip write-write (handled by WAW)

                    if write_access.pipe != read_access.pipe:
                        # Cross-pipeline RAW dependency
                        is_slot_specific = (
                            write_access.slot_index_ssa == read_access.slot_index_ssa
                        )

                        pipe_pair = (write_access.pipe, read_access.pipe)
                        dependencies[pipe_pair].append({
                            'type': 'RAW',
                            'src_access': write_access,
                            'dst_access': read_access,
                            'is_slot_specific': is_slot_specific
                        })

            # Find WAW dependencies (write -> write)
            for i, write1 in enumerate(accesses):
                if not write1.is_write:
                    continue

                for j in range(i + 1, len(accesses)):
                    write2 = accesses[j]
                    if not write2.is_write:
                        continue

                    if write1.pipe != write2.pipe:
                        is_slot_specific = (
                            write1.slot_index_ssa == write2.slot_index_ssa
                        )

                        pipe_pair = (write1.pipe, write2.pipe)
                        dependencies[pipe_pair].append({
                            'type': 'WAW',
                            'src_access': write1,
                            'dst_access': write2,
                            'is_slot_specific': is_slot_specific
                        })

            # Find WAR dependencies (read -> write)
            for i, read_access in enumerate(accesses):
                if read_access.is_write:
                    continue

                for j in range(i + 1, len(accesses)):
                    write_access = accesses[j]
                    if not write_access.is_write:
                        continue

                    if read_access.pipe != write_access.pipe:
                        is_slot_specific = (
                            read_access.slot_index_ssa == write_access.slot_index_ssa
                        )

                        pipe_pair = (read_access.pipe, write_access.pipe)
                        dependencies[pipe_pair].append({
                            'type': 'WAR',
                            'src_access': read_access,
                            'dst_access': write_access,
                            'is_slot_specific': is_slot_specific
                        })

        return dependencies

    def print_tilegroup_analysis(self):
        """Print analysis of TileGroup accesses for debugging."""
        print("="*80)
        print("TileGroup Access Analysis")
        print("="*80)

        if not self._tile_group_registry:
            print("No TileGroups detected.")
            return

        for group_info in self._tile_group_registry.values():
            print(f"\nTileGroup #{group_info.group_id} (size={group_info.size}):")
            print(f"  Total accesses: {len(group_info.accesses)}")

            # Group by pipeline
            by_pipe = defaultdict(list)
            for access in group_info.accesses:
                by_pipe[access.pipe].append(access)

            for pipe, accesses in by_pipe.items():
                reads = sum(1 for a in accesses if not a.is_write)
                writes = sum(1 for a in accesses if a.is_write)
                print(f"  {pipe}: {reads} reads, {writes} writes")

        # Print dependencies
        print("\n" + "="*80)
        print("Dependencies")
        print("="*80)

        deps = self.analyze_dependencies()
        if not deps:
            print("No cross-pipeline dependencies detected.")
            return

        for pipe_pair, dep_list in deps.items():
            src, dst = pipe_pair
            print(f"\n{src} → {dst}:")
            slot_specific = sum(1 for d in dep_list if d['is_slot_specific'])
            slot_agnostic = len(dep_list) - slot_specific
            print(f"  Slot-specific: {slot_specific}")
            print(f"  Slot-agnostic: {slot_agnostic}")

            # Show a few examples
            for dep in dep_list[:3]:
                print(f"    {dep['type']}: " +
                      f"{'slot-specific' if dep['is_slot_specific'] else 'slot-agnostic'}")

    # -- Phase 3: Event Allocation ------------------------------------------

    def allocate_events(self):
        """Phase 3: Automatically allocate EventIdGroups for TileGroups.

        Returns:
            bool: True if allocation succeeded, False if EVENT_ID resources exhausted
        """
        from ._event_group import EventIdGroup

        # First, analyze which pipeline pairs each TileGroup uses
        dependencies = self.analyze_dependencies()

        # Build mapping: group_id -> set of pipeline pairs
        group_to_pipe_pairs = defaultdict(set)
        for (src, dst), dep_list in dependencies.items():
            for dep in dep_list:
                if dep['is_slot_specific']:
                    # This pipeline pair needs slot-specific sync
                    group_id = dep['src_access'].group_id
                    group_to_pipe_pairs[group_id].add((src, dst))

        # Organize TileGroups by pipeline pairs
        # Key: frozenset of pipeline pairs -> List of TileGroupInfo
        pipe_pairs_to_groups = defaultdict(list)
        for group_info in self._tile_group_registry.values():
            pipe_pairs = frozenset(group_to_pipe_pairs.get(group_info.group_id, set()))
            if pipe_pairs:  # Only allocate if there are actual dependencies
                pipe_pairs_to_groups[pipe_pairs].append(group_info)

        # EVENT_ID pool
        available_events = list(_EVENT_IDS)  # Copy
        event_allocation_map = {}  # pipe_pairs -> list of allocated event_ids

        # Allocate EVENT_IDs for each unique pipeline pair set
        for pipe_pairs, groups in sorted(pipe_pairs_to_groups.items(),
                                         key=lambda x: -len(x[1])):  # Largest first
            # Calculate total EVENT_IDs needed for this pipeline pair set
            total_needed = sum(g.size for g in groups)

            # Check if we have enough EVENT_IDs
            if len(available_events) < total_needed:
                # Try to find EVENT_IDs that can be reused
                reusable = self._find_reusable_events(pipe_pairs, event_allocation_map)
                if len(reusable) + len(available_events) >= total_needed:
                    # Use reusable events first
                    allocated = reusable + available_events[:total_needed - len(reusable)]
                else:
                    print(f"WARNING: Insufficient EVENT_IDs for TileGroups")
                    print(f"  Need {total_needed}, have {len(available_events)} available")
                    print(f"  Pipeline pairs: {pipe_pairs}")
                    return False
            else:
                # Allocate from available pool
                allocated = available_events[:total_needed]
                available_events = available_events[total_needed:]

            # Record allocation for this pipeline pair set
            event_allocation_map[pipe_pairs] = allocated

            # Assign EventIdGroups to each TileGroup in this set
            offset = 0
            for group_info in groups:
                n = group_info.size
                events = allocated[offset:offset + n]
                group_info.event_group = EventIdGroup(events)
                group_info.pipeline_pairs = pipe_pairs
                offset += n

        # Summary
        print(f"\n{'='*80}")
        print(f"Event Allocation Summary")
        print(f"{'='*80}")
        print(f"Total TileGroups: {len(self._tile_group_registry)}")
        print(f"EVENT_IDs allocated: {len(_EVENT_IDS) - len(available_events)}/{len(_EVENT_IDS)}")
        print(f"EVENT_IDs remaining: {len(available_events)}")

        for group_info in self._tile_group_registry.values():
            if group_info.event_group:
                events_str = ', '.join(str(e).split('.')[-1] for e in group_info.event_group._events)
                print(f"  Group #{group_info.group_id}: [{events_str}]")

        return True

    def _find_reusable_events(self, pipe_pairs, event_allocation_map):
        """Find EVENT_IDs that can be reused for a new pipeline pair set.

        EVENT_IDs can be reused if the pipeline pairs don't overlap.
        """
        reusable = []
        for existing_pairs, allocated_events in event_allocation_map.items():
            # Check if pipeline pairs overlap
            if not pipe_pairs.intersection(existing_pairs):
                # No overlap, can reuse these EVENT_IDs
                reusable.extend(allocated_events)

        return reusable

    def get_event_group_for_tile(self, tile):
        """Get the EventIdGroup for a TileGroup tile.

        Args:
            tile: A Tile object that may be from a TileGroup

        Returns:
            EventIdGroup if tile is from a TileGroup with allocated events,
            None otherwise
        """
        group = getattr(tile, '_group_tiles', None)
        if group is None:
            return None

        group_id = id(group[0])
        group_info = self._tile_group_registry.get(group_id)
        if group_info is None:
            return None

        return group_info.event_group

    def get_slot_index_for_tile(self, tile):
        """Get the slot index ScalarValue for a TileGroup tile.

        Args:
            tile: A Tile object from a TileGroup

        Returns:
            ScalarValue representing the slot index, or None
        """
        return getattr(tile, '_group_idx', None)

    def _check_tilegroup_event(self, tiles):
        """Check if TileGroup tiles use the same slot index and have allocated EventIdGroup.

        This is used for semi-automatic sync insertion. If all TileGroup tiles
        use the same slot index (SSA value) and at least one TileGroup has an
        allocated EventIdGroup, we can use dynamic EventId selection.

        Note: Tiles can be from different TileGroups (e.g., a[idx], b[idx], c[idx])
        as long as they use the same slot_index SSA value.

        Args:
            tiles: List of Tile objects

        Returns:
            (event_group, slot_idx) tuple if all TileGroup tiles use same slot index
            and at least one has allocated EventIdGroup, (None, None) otherwise
        """
        if not tiles:
            return (None, None)

        # Find all TileGroup tiles and their slot indices
        tilegroup_tiles = []
        slot_indices = []

        for tile in tiles:
            group = getattr(tile, '_group_tiles', None)
            if group is not None:
                slot_idx = getattr(tile, '_group_idx', None)
                if slot_idx is not None:
                    tilegroup_tiles.append(tile)
                    slot_indices.append(slot_idx.ssa)

        # If no TileGroup tiles, return None
        if not tilegroup_tiles:
            return (None, None)

        # Check if all TileGroup tiles use the same slot index (SSA value)
        first_ssa = slot_indices[0]
        if not all(ssa == first_ssa for ssa in slot_indices):
            return (None, None)  # Different slot indices

        # Find an EventIdGroup from any of the TileGroup tiles
        event_group = None
        slot_idx = None

        for tile in tilegroup_tiles:
            eg = self.get_event_group_for_tile(tile)
            si = self.get_slot_index_for_tile(tile)
            if eg is not None and si is not None:
                event_group = eg
                slot_idx = si
                break

        return (event_group, slot_idx)

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
        """Check dependencies and emit forward sync before the op, then update state.

        SIMPLIFIED VERSION: Always use EVENT_ID0 for all syncs.
        """
        # Record TileGroup accesses for auto-sync analysis
        for tile in reads:
            self.record_tilegroup_access(tile, pipe, is_write=False)
        for tile in writes:
            self.record_tilegroup_access(tile, pipe, is_write=True)

        emitted = {}  # (src_pipe, dst_pipe) -> set of tile_ids already synced

        # Helper: for TileGroup, expand to all underlying tiles for conservative sync
        def get_all_tiles(tile):
            """Expand TileGroup to all underlying tiles, or return single tile ID."""
            group = getattr(tile, '_group_tiles', None)
            if group is not None:
                # Expand to all tiles in group for conservative sync
                return [id(t) for t in group]
            else:
                return [id(tile)]

        # --- RAW: this op reads tiles that were written on a different pipe ---
        for tile in reads:
            for tid in get_all_tiles(tile):
                state = self._buffer_states.get(tid)
                if state and state.last_write_pipe is not None and state.last_write_pipe != pipe:
                    pair = (state.last_write_pipe, pipe)
                    if pair not in emitted:
                        emitted[pair] = set()
                    emitted[pair].add(tid)

        # --- WAW: this op writes tiles that were written on a different pipe ---
        for tile in writes:
            for tid in get_all_tiles(tile):
                state = self._buffer_states.get(tid)
                if state and state.last_write_pipe is not None and state.last_write_pipe != pipe:
                    pair = (state.last_write_pipe, pipe)
                    if pair not in emitted:
                        emitted[pair] = set()
                    emitted[pair].add(tid)

        # --- WAR: this op writes tiles that were read on a different pipe ---
        for tile in writes:
            for tid in get_all_tiles(tile):
                state = self._buffer_states.get(tid)
                if state:
                    for read_pipe in state.last_read_pipes:
                        if read_pipe != pipe:
                            pair = (read_pipe, pipe)
                            if pair not in emitted:
                                emitted[pair] = set()
                            emitted[pair].add(tid)

        # Emit all collected syncs - ALWAYS USE EVENT_ID0
        for (src, dst), tile_ids in emitted.items():
            _emit_set_flag(src, dst, _EVENT_IDS[0])  # Always EVENT_ID0
            _emit_wait_flag(src, dst, _EVENT_IDS[0])  # Always EVENT_ID0

        # --- Update buffer state ---
        for tile in writes:
            for tid in get_all_tiles(tile):
                st = self._buffer_states.get(tid)
                if not st:
                    st = BufferState()
                    self._buffer_states[tid] = st
                st.last_write_pipe = pipe
                st.last_read_pipes.clear()

        for tile in reads:
            for tid in get_all_tiles(tile):
                st = self._buffer_states.get(tid)
                if not st:
                    st = BufferState()
                    self._buffer_states[tid] = st
                st.last_read_pipes.add(pipe)

        # --- Update only the innermost loop context ---
        if self._loop_stack:
            ctx = self._loop_stack[-1]
            all_tiles = list(reads) + list(writes)
            for tile in all_tiles:
                # For loop context, expand all tiles for conservative tracking
                for tid in get_all_tiles(tile):
                    if tid not in ctx.first_access:
                        ctx.first_access[tid] = pipe
                    ctx.last_access[tid] = pipe

                # Track TileGroup membership for merging later
                group = getattr(tile, '_group_tiles', None)
                if group is not None:
                    group_id = id(group[0])
                    if group_id not in ctx.group_size_map:
                        ctx.group_size_map[group_id] = len(group)
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

        SIMPLIFIED VERSION: Always use EVENT_ID0 for all backward syncs.
        Merge tiles from the same TileGroup to avoid redundant syncs.
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

                # SIMPLIFIED: Always use EVENT_ID0 for all clusters
                for _ in clusters:
                    unconditional_events.append((src, dst, _EVENT_IDS[0]))

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

        SIMPLIFIED VERSION: Only handle unconditional events (no conditional groups).
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

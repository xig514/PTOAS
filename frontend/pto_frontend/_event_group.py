"""EventIdGroup: dynamic EVENT_ID selection for multi-buffering sync patterns.

Allows indexing a group of EVENT_IDs with a dynamic ScalarValue, generating
``scf.if`` chains to select the correct event at runtime. This enables cleaner
multi-buffering code without manually writing if/else branches for each sync.

Example::

    event_ids = pto.EventIdGroup([pto.EVENT_ID0, pto.EVENT_ID1])

    for i in pto.range(N):
        buf_idx = i % 2

        # Automatically generates conditional sync
        pto.wait_flag(pto.PIPE_MTE2, pto.PIPE_V, event_ids[buf_idx])
        # ... operations ...
        pto.set_flag(pto.PIPE_MTE2, pto.PIPE_V, event_ids[buf_idx])

This is equivalent to manually writing:
    with pto.if_(buf_idx == 0):
        pto.wait_flag(pto.PIPE_MTE2, pto.PIPE_V, pto.EVENT_ID0)
    with pto.else_():
        pto.wait_flag(pto.PIPE_MTE2, pto.PIPE_V, pto.EVENT_ID1)
"""

from mlir.dialects.pto import EVENT

from ._scalar import ScalarValue


class EventIdGroup:
    """A group of EVENT_IDs for dynamic selection (double/multi-buffering sync).

    Indexing with a Python ``int`` returns the EVENT_ID directly (trace-time).
    Indexing with a :class:`ScalarValue` returns a special marker object that
    triggers conditional sync generation in set_flag/wait_flag.
    """

    def __init__(self, event_ids):
        """Create an EVENT_ID group.

        Args:
            event_ids: List of EVENT enum values (e.g., [EVENT.EVENT_ID0, EVENT.EVENT_ID1])
        """
        self._events = list(event_ids)
        if len(self._events) < 2:
            raise ValueError("EventIdGroup requires at least 2 EVENT_IDs")

        # Validate all are EVENT enum values
        for e in self._events:
            if not isinstance(e, EVENT):
                raise TypeError(
                    f"EventIdGroup requires EVENT enum values, got {type(e).__name__}"
                )

    def __len__(self):
        return len(self._events)

    def __getitem__(self, idx):
        """Select an EVENT_ID by index.

        Args:
            idx: int (trace-time selection) or ScalarValue (runtime selection)

        Returns:
            For int: The EVENT_ID directly
            For ScalarValue: A _DynamicEventSelection marker
        """
        if isinstance(idx, int):
            return self._events[idx]
        if isinstance(idx, ScalarValue):
            return _DynamicEventSelection(self._events, idx)
        raise TypeError(
            f"EventIdGroup index must be int or ScalarValue, "
            f"got {type(idx).__name__}"
        )


class _DynamicEventSelection:
    """Marker for a runtime-selected EVENT_ID.

    This object is returned by EventIdGroup[ScalarValue] and is recognized
    by set_flag/wait_flag to trigger conditional code generation.
    """

    def __init__(self, events, idx):
        """
        Args:
            events: List of EVENT enum values
            idx: ScalarValue used for selection
        """
        self.events = events
        self.idx = idx

    def emit_conditional(self, callback):
        """Generate scf.if chain to select the correct EVENT_ID.

        Args:
            callback: Function that takes (event_id) and emits the sync op
        """
        from ._control_flow import if_, else_

        # Generate nested if/else chain:
        # if idx == 0:
        #     callback(events[0])
        # else:
        #     if idx == 1:
        #         callback(events[1])
        #     else:
        #         ...
        self._emit_chain(0, callback)

    def _emit_chain(self, cmp_val, callback):
        """Recursively emit if/else chain for event selection."""
        from ._control_flow import if_, else_

        if cmp_val == len(self.events) - 1:
            # Last event: no more conditions needed
            callback(self.events[cmp_val])
        else:
            # if idx == cmp_val: callback(events[cmp_val])
            cond = (self.idx == cmp_val)
            with if_(cond):
                callback(self.events[cmp_val])
            with else_():
                self._emit_chain(cmp_val + 1, callback)

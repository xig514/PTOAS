"""MultiBuffer: auto-cycling multi-buffer abstraction over TileGroup.

Wraps N tile buffers so the user does not need to manually compute
``j % depth`` indices.  Pass pre-allocated tiles directly::

    mb = MultiBuffer(tile0, tile1)               # double buffer
    mb = MultiBuffer(tile0, tile1, tile2)         # triple buffer

Inside a ``pto.range`` / ``pto.for_range`` loop, ``get()`` automatically
detects the loop induction variable and cycles buffers::

    mb = MultiBuffer(tile0, tile1)
    for j in pto.range(N):
        cur = mb.get()      # buffer at j%2, advances counter
        pre = mb.get_pre()  # buffer at (j-1)%2

All MultiBuffers with the same depth inside the same loop share a single
``iv % depth`` SSA value, ensuring the sync tracker can deduplicate
backward sync ops correctly.

Sync insertion is handled by the existing auto_sync / TileGroup machinery.
"""

from ._tile_group import TileGroup
from ._scalar import ScalarValue


class MultiBuffer:
    """Auto-cycling multi-buffer with internal counter.

    Parameters
    ----------
    *tiles : Tile
        Two or more pre-allocated tiles.  Also accepts a single list/tuple
        of tiles for backward compatibility.

    Examples
    --------
    ::

        t0 = pto.make_tile((64,64), pto.float16, pto.MAT, addr=0)
        t1 = pto.make_tile((64,64), pto.float16, pto.MAT, addr=8192)
        mb = pto.MultiBuffer(t0, t1)

        for j in pto.range(N):
            cur = mb.get()       # automatically uses j % 2
    """

    def __init__(self, *tiles):
        # Accept either variadic tiles or a single list/tuple
        if len(tiles) == 1 and isinstance(tiles[0], (list, tuple)):
            tiles = list(tiles[0])
        else:
            tiles = list(tiles)
        if len(tiles) < 2:
            raise ValueError("MultiBuffer requires at least 2 tiles")
        self._tiles = tiles

        self._depth = len(self._tiles)
        self._group = TileGroup(self._tiles)
        self._base_idx = None   # ScalarValue: loop_var % depth (cached)
        self._counter = 0       # Python int, tracks get() calls within trace
        self._last_scope = None # LoopScope from last auto-detection

    @property
    def depth(self):
        """Number of buffers."""
        return self._depth

    @property
    def tiles(self):
        """Underlying tile list (for manual access if needed)."""
        return list(self._tiles)

    @property
    def group(self):
        """Underlying TileGroup."""
        return self._group

    def bind(self, loop_var):
        """Bind to a loop induction variable for per-iteration cycling.

        Normally not needed — ``get()`` auto-detects the enclosing loop.
        Use this only when you need explicit control over the index SSA.

        Parameters
        ----------
        loop_var : ScalarValue or int
            The loop induction variable, or a pre-computed buffer index.

        Returns
        -------
        self
            For chaining.
        """
        if isinstance(loop_var, ScalarValue):
            self._base_idx = loop_var
        else:
            self._base_idx = loop_var % self._depth
        self._counter = 0
        return self

    def _auto_bind_from_loop(self):
        """Auto-detect current loop scope and bind if scope changed."""
        from ._control_flow import get_current_loop_scope
        scope = get_current_loop_scope()
        if scope is None:
            return  # Not inside a loop — use static counter
        if scope is self._last_scope:
            return  # Same loop iteration trace — already bound
        # New or different loop scope: reset and bind via shared cache
        self._last_scope = scope
        self._base_idx = scope.iteration_index(self._depth)
        self._counter = 0

    def get(self):
        """Return the current buffer tile and advance the counter.

        If inside a ``pto.range`` / ``pto.for_range`` loop, automatically
        binds to the loop induction variable (shared ``iv % depth`` SSA).
        Otherwise uses the Python-level counter directly.

        Returns
        -------
        Tile
            The selected tile buffer.
        """
        self._auto_bind_from_loop()
        idx = self._current_index()
        self._counter += 1
        return self._group[idx]

    def get_pre(self):
        """Return the previous buffer tile (one step behind current counter).

        Uses ``(base_idx + counter - 1) % depth``.  Does NOT advance the counter.

        Returns
        -------
        Tile
            The previous tile buffer.
        """
        self._auto_bind_from_loop()
        idx = self._prev_index()
        return self._group[idx]

    def _current_index(self):
        """Compute the current buffer index."""
        if self._base_idx is not None:
            if self._counter == 0:
                return self._base_idx
            return (self._base_idx + self._counter) % self._depth
        # Static (trace-time) index
        return self._counter % self._depth

    def _prev_index(self):
        """Compute the previous buffer index (counter - 1)."""
        if self._base_idx is not None:
            if self._counter == 0:
                return (self._base_idx - 1) % self._depth
            if self._counter == 1:
                return self._base_idx
            return (self._base_idx + self._counter - 1) % self._depth
        return (self._counter - 1) % self._depth

    def reset(self):
        """Reset the internal counter to 0 (without unbinding)."""
        self._counter = 0

    def __len__(self):
        return self._depth

    def __getitem__(self, idx):
        """Direct index access (delegates to TileGroup)."""
        return self._group[idx]

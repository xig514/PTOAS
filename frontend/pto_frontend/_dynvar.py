"""DynVar: dynamic shape variable for Tensor annotations."""


class DynVar:
    """Dynamic shape variable that can appear in ``Tensor[[M, N], dtype]``.

    During ``@kernel`` tracing the variable is *bound* to the SSA index
    value that corresponds to the tensor dimension it first appears in.
    Once bound, it can participate in arithmetic (loop bounds, offsets)
    by delegating to the underlying :class:`ScalarValue`.

    Usage::

        M = pto.DynVar("M")
        N = pto.DynVar("N")

        @pto.kernel
        def my_kernel(x: pto.Tensor[[M, N], pto.float16], ...):
            with pto.for_range(0, M, 64) as i:
                ...
    """

    def __init__(self, name: str):
        self.name = name
        self._scalar = None  # bound during tracing

    # -- binding API (used by _kernel.py) ---------------------------------

    def _bind(self, scalar_value):
        """Bind this DynVar to a :class:`ScalarValue`."""
        self._scalar = scalar_value

    def _unbind(self):
        """Release the binding after tracing completes."""
        self._scalar = None

    @property
    def is_bound(self):
        return self._scalar is not None

    # -- delegate to ScalarValue ------------------------------------------

    def _require_bound(self):
        if self._scalar is None:
            raise RuntimeError(
                f"DynVar '{self.name}' is not bound. "
                "It can only be used inside a @kernel-traced function."
            )
        return self._scalar

    # Arithmetic operators

    def __add__(self, other):
        return self._require_bound().__add__(other)

    def __radd__(self, other):
        return self._require_bound().__radd__(other)

    def __sub__(self, other):
        return self._require_bound().__sub__(other)

    def __rsub__(self, other):
        return self._require_bound().__rsub__(other)

    def __mul__(self, other):
        return self._require_bound().__mul__(other)

    def __rmul__(self, other):
        return self._require_bound().__rmul__(other)

    def __floordiv__(self, other):
        return self._require_bound().__floordiv__(other)

    def __mod__(self, other):
        return self._require_bound().__mod__(other)

    # Comparison operators

    def __lt__(self, other):
        return self._require_bound().__lt__(other)

    def __le__(self, other):
        return self._require_bound().__le__(other)

    def __gt__(self, other):
        return self._require_bound().__gt__(other)

    def __ge__(self, other):
        return self._require_bound().__ge__(other)

    def __eq__(self, other):
        return self._require_bound().__eq__(other)

    def __ne__(self, other):
        return self._require_bound().__ne__(other)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"DynVar({self.name!r})"

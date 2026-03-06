"""@pto_meta_data decorator for kernel metadata configuration.

Supports both static (compile-time) and dynamic (runtime) metadata:

Static metadata example:
    @pto_meta_data
    def meta():
        return {
            "phys_row": 128,
            "phys_col": 64,
            "dtype": pto.float32,
        }

Dynamic metadata example:
    @pto_meta_data
    def meta():
        return {
            "phys_row": 128,
            "phys_col": 64,
            "dtype": pto.float32,
            "valid_row": "dynamic",  # extracted from tensor.shape[0]
            "valid_col": "dynamic",  # extracted from tensor.shape[1]
        }
"""


class MetaDataFunction:
    """Wrapper for metadata function decorated with @pto_meta_data."""

    def __init__(self, fn):
        self._fn = fn
        self._config = None

    def get_config(self):
        """Get metadata configuration (cached)."""
        if self._config is None:
            self._config = self._fn()
        return self._config

    def get_static_metadata(self):
        """Extract only static (non-dynamic) metadata."""
        config = self.get_config()
        return {k: v for k, v in config.items() if v != "dynamic"}

    def get_dynamic_keys(self):
        """Get list of keys marked as 'dynamic'."""
        config = self.get_config()
        return [k for k, v in config.items() if v == "dynamic"]

    def __call__(self):
        """Allow calling the metadata function directly."""
        return self.get_config()


def pto_meta_data(fn):
    """Decorator for kernel metadata configuration.

    Usage::

        @pto_meta_data
        def meta():
            return {
                "phys_row": 128,
                "phys_col": 64,
                "dtype": pto.float32,
                "valid_row": "dynamic",  # runtime from tensor.shape
            }

        @pto.kernel(metadata=meta)
        def my_kernel(src: pto.Tensor(...), out: pto.Tensor(...)):
            # Access static metadata directly
            phys_row = meta.get_static_metadata()["phys_row"]

            # Dynamic metadata extracted from tensor
            valid_row = src.shape[0]
    """
    return MetaDataFunction(fn)


__all__ = ["pto_meta_data", "MetaDataFunction"]

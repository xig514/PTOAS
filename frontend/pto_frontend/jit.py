"""@jit decorator for host-side code that orchestrates @kernel calls.

Usage::

    @pto.jit
    def run():
        x = torch.randn(1024, 128, dtype=torch.float16)
        y = torch.randn(1024, 128, dtype=torch.float16)
        out = torch.empty_like(x)
        vector_add[32](x, y, out)  # auto-compiles if needed
        return out

    result = run()
"""

import functools


def jit(fn):
    """Decorator for host-side code that references @kernel functions.

    Executes the host function normally. When ``kernel[grid](args)`` is
    called inside, the kernel is auto-compiled (IR → C++ → .so).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


__all__ = ["jit"]

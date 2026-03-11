"""@jit decorator, compile(), and launch() for host-side kernel execution.

Usage::

    import pto_frontend as pto

    M = pto.DynVar("M")
    N = pto.DynVar("N")

    @pto.kernel
    def add_kernel(
        x: pto.Tensor[[M, N], pto.float16],
        y: pto.Tensor[[M, N], pto.float16],
        z: pto.Tensor[[M, N], pto.float16],
    ):
        ...

    @pto.jit
    def test_add():
        compiled = pto.compile(add_kernel)

        x = torch.randn(128, 256, dtype=torch.float16, device="npu:0")
        y = torch.randn(128, 256, dtype=torch.float16, device="npu:0")
        z = torch.empty_like(x)

        pto.launch(compiled, x, y, z)
        torch.npu.synchronize()

    test_add()
"""

import ctypes
import dataclasses
import functools
import builtins as _builtins

# Keep a reference to Python's builtin range (we shadow it in the package)
builtins_range = _builtins.range


# ---------------------------------------------------------------------------
#  pto DType name → torch.dtype  (lazy — torch imported on first use)
# ---------------------------------------------------------------------------

_PTO_TO_TORCH_DTYPE = None

def _get_torch_dtype_map():
    global _PTO_TO_TORCH_DTYPE
    if _PTO_TO_TORCH_DTYPE is None:
        import torch
        _PTO_TO_TORCH_DTYPE = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
            "int8":     torch.int8,
            "int16":    torch.int16,
            "int32":    torch.int32,
            "int64":    torch.int64,
        }
    return _PTO_TO_TORCH_DTYPE


# ---------------------------------------------------------------------------
#  CompiledKernel
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CompiledKernel:
    """Result of :func:`compile` — the ``.so`` path and parameter metadata."""
    lib_path: str
    param_specs: list  # list of (param_name, spec_or_tag)


# ---------------------------------------------------------------------------
#  compile()
# ---------------------------------------------------------------------------

def compile(kernel_fn, *, pto_level="level3", arch="a3", npu_arch="dav-c220-vec",
            auto_sync=False):
    """Compile a ``@pto.kernel`` function to a shared library.

    Parameters
    ----------
    kernel_fn : KernelFunction
        A function decorated with ``@pto.kernel``.
    pto_level, arch, npu_arch
        Forwarded to ``kernel_fn.compile()``.
    auto_sync : bool
        When *True*, automatically insert pipeline synchronization ops.

    Returns
    -------
    CompiledKernel
        Contains ``lib_path`` and ``param_specs`` for use with :func:`launch`.
    """
    kernel_fn.compile(pto_level=pto_level, arch=arch, npu_arch=npu_arch,
                      auto_sync=auto_sync)
    return CompiledKernel(
        lib_path=kernel_fn.library_path,
        param_specs=list(kernel_fn._param_specs),
    )


# ---------------------------------------------------------------------------
#  launch()
# ---------------------------------------------------------------------------

def launch(compiled, *args, block_dim=1, stream=None):
    """Launch a compiled kernel with the given arguments.

    Parameters
    ----------
    compiled : CompiledKernel or str
        Result of :func:`compile`, or a path to a ``.so`` file.
    *args
        Runtime arguments matching the kernel signature.
        Tensor parameters → ``torch.Tensor``.
        Scalar parameters → ``int``, ``float``, or ``bool``.
    block_dim : int
        Number of AICore blocks (default 1).
    stream
        NPU stream. Defaults to ``torch.npu.current_stream()``.
    """
    import torch

    if isinstance(compiled, str):
        lib_path = compiled
        param_specs = []
    else:
        lib_path = compiled.lib_path
        param_specs = compiled.param_specs

    if stream is None:
        stream = torch.npu.current_stream()

    lib = ctypes.CDLL(lib_path)

    # Build ctypes args for call_kernel(blockDim, stream, ...)
    ctypes_args = [ctypes.c_uint32(block_dim), stream._as_parameter_]

    if param_specs:
        _validate_args(args, param_specs)
        ctypes_args.extend(_args_to_ctypes(args, param_specs))
    else:
        # Fallback: treat all args as void* (raw tensors)
        for arg in args:
            if isinstance(arg, torch.Tensor):
                ctypes_args.append(ctypes.c_void_p(arg.data_ptr()))
            else:
                ctypes_args.append(arg)

    lib.call_kernel(*ctypes_args)


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------

def _validate_args(args, param_specs):
    """Validate runtime args against the kernel's parameter descriptions."""
    import torch
    from ._tensor import _TensorSpec, _TensorShapeSpec
    from ._dynvar import DynVar

    dtype_map = _get_torch_dtype_map()

    # Count expected args (each param_spec entry = 1 user arg)
    if len(args) != len(param_specs):
        raise TypeError(
            f"Expected {len(param_specs)} arguments, got {len(args)}"
        )

    dyn_values = {}  # DynVar name → resolved runtime value

    for i, (arg, (pname, spec)) in enumerate(zip(args, param_specs)):
        if isinstance(spec, (_TensorSpec, _TensorShapeSpec)):
            if not isinstance(arg, torch.Tensor):
                raise TypeError(
                    f"arg[{i}] '{pname}': expected torch.Tensor, "
                    f"got {type(arg).__name__}"
                )
            # Check dtype
            expected_torch_dtype = dtype_map.get(spec.dtype.name)
            if expected_torch_dtype is not None and arg.dtype != expected_torch_dtype:
                raise TypeError(
                    f"arg[{i}] '{pname}': dtype mismatch — "
                    f"expected {expected_torch_dtype}, got {arg.dtype}"
                )
            # Check rank
            if arg.ndim != spec.ndim:
                raise TypeError(
                    f"arg[{i}] '{pname}': rank mismatch — "
                    f"expected {spec.ndim}D, got {arg.ndim}D"
                )
            # Check shape (for _TensorShapeSpec with static dims / DynVar consistency)
            if isinstance(spec, _TensorShapeSpec):
                for d, dim_spec in enumerate(spec.shape):
                    actual = arg.shape[d]
                    if isinstance(dim_spec, int):
                        if dim_spec != actual:
                            raise TypeError(
                                f"arg[{i}] '{pname}': dim[{d}] mismatch — "
                                f"expected {dim_spec}, got {actual}"
                            )
                    elif isinstance(dim_spec, DynVar):
                        name = dim_spec.name
                        if name in dyn_values and dyn_values[name] != actual:
                            raise TypeError(
                                f"arg[{i}] '{pname}': DynVar '{name}' "
                                f"mismatch — previously {dyn_values[name]}, "
                                f"got {actual} at dim[{d}]"
                            )
                        dyn_values.setdefault(name, actual)

        elif spec == "index":
            if not isinstance(arg, int):
                raise TypeError(
                    f"arg[{i}] '{pname}': expected int, got {type(arg).__name__}"
                )
        elif spec == "f32":
            if not isinstance(arg, (int, float)):
                raise TypeError(
                    f"arg[{i}] '{pname}': expected float, got {type(arg).__name__}"
                )
        elif spec == "i1":
            if not isinstance(arg, (bool, int)):
                raise TypeError(
                    f"arg[{i}] '{pname}': expected bool, got {type(arg).__name__}"
                )


# ---------------------------------------------------------------------------
#  Args → ctypes conversion
# ---------------------------------------------------------------------------

def _args_to_ctypes(args, param_specs):
    """Convert runtime args to ctypes values for the call_kernel ABI.

    The call_kernel ABI for each parameter:
      - Tensor: uint8_t* ptr, int32_t dim0, int32_t dim1, ...
      - index:  int32_t
      - f32:    float
      - i1:     int32_t (0 or 1)
    """
    from ._tensor import _TensorSpec, _TensorShapeSpec

    result = []
    for arg, (pname, spec) in zip(args, param_specs):
        if isinstance(spec, (_TensorSpec, _TensorShapeSpec)):
            # Tensor → pointer + per-dimension sizes
            result.append(ctypes.c_void_p(arg.data_ptr()))
            for d in builtins_range(spec.ndim):
                result.append(ctypes.c_int32(arg.shape[d]))
        elif spec == "index":
            result.append(ctypes.c_int32(int(arg)))
        elif spec == "f32":
            result.append(ctypes.c_float(float(arg)))
        elif spec == "i1":
            result.append(ctypes.c_int32(int(bool(arg))))

    return result


# ---------------------------------------------------------------------------
#  @jit decorator
# ---------------------------------------------------------------------------

def jit(fn=None):
    """Decorator for host-side code that orchestrates kernel compilation and launch.

    The decorated function runs normally on the host.  Inside it you
    can call :func:`compile` and :func:`launch`.

    Usage::

        @pto.jit
        def test():
            compiled = pto.compile(my_kernel)
            x = torch.randn(128, 256, dtype=torch.float16, device="npu:0")
            out = torch.empty_like(x)
            pto.launch(compiled, x, out)
            torch.npu.synchronize()

        test()
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper

    if callable(fn):
        # Called as @pto.jit (no parentheses)
        return decorator(fn)
    # Called as @pto.jit() (with parentheses)
    return decorator


__all__ = ["jit", "compile", "launch", "CompiledKernel"]

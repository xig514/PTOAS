"""@kernel decorator: signature parsing, Tensor flattening, and tracing."""

import inspect

from mlir.ir import InsertionPoint, IndexType, F32Type, IntegerType
from mlir.dialects import func

from ._ir_builder import IRBuilder, set_builder, clear_builder
from ._tensor import _TensorSpec, _TensorProxy
from ._scalar import ScalarValue


class KernelFunction:
    """Wrapper returned by ``@pto.kernel``.

    Calling the object traces the kernel and prints the generated MLIR.
    """

    def __init__(self, fn, name):
        self._fn = fn
        self._name = name

    def emit_ir(self):
        """Trace the kernel and return the MLIR module as a string."""
        builder = IRBuilder()
        set_builder(builder)
        try:
            return self._trace(builder)
        finally:
            builder.close()
            clear_builder()

    def _trace(self, builder):
        hints = {
            k: v
            for k, v in self._fn.__annotations__.items()
            if k != "return"
        }
        params = list(inspect.signature(self._fn).parameters.keys())

        # -- build flattened arg types --
        flat_types = []
        param_specs = []  # (param_name, spec_or_tag)
        for pname in params:
            spec = hints.get(pname)
            if isinstance(spec, _TensorSpec):
                from mlir.dialects import pto as _pto

                flat_types.append(_pto.PtrType.get(spec.dtype.to_mlir()))
                for _ in range(spec.ndim):
                    flat_types.append(IndexType.get())
                param_specs.append((pname, spec))
            elif spec is int:
                flat_types.append(IndexType.get())
                param_specs.append((pname, "index"))
            elif spec is float:
                flat_types.append(F32Type.get())
                param_specs.append((pname, "f32"))
            elif spec is bool:
                flat_types.append(IntegerType.get_signless(1))
                param_specs.append((pname, "i1"))
            else:
                raise TypeError(
                    f"Unsupported annotation for parameter '{pname}': {spec!r}"
                )

        fn_type = func.FunctionType.get(flat_types, [])

        # -- create func.func --
        with InsertionPoint(builder.module.body):
            fn_op = func.FuncOp(self._name, fn_type)
            entry = fn_op.add_entry_block()

        # -- reconstruct proxy objects from block args --
        block_args = list(entry.arguments)
        proxy_args = []
        idx = 0
        for _pname, spec in param_specs:
            if isinstance(spec, _TensorSpec):
                ptr_ssa = block_args[idx]
                idx += 1
                shape_ssas = []
                for _ in range(spec.ndim):
                    shape_ssas.append(block_args[idx])
                    idx += 1
                proxy_args.append(
                    _TensorProxy(ptr_ssa, shape_ssas, spec.dtype, spec.ndim)
                )
            elif spec == "index":
                proxy_args.append(ScalarValue(block_args[idx]))
                idx += 1
            elif spec == "f32":
                proxy_args.append(ScalarValue(block_args[idx], is_float=True))
                idx += 1
            elif spec == "i1":
                proxy_args.append(ScalarValue(block_args[idx]))
                idx += 1

        # -- trace user function --
        with InsertionPoint(entry):
            self._fn(*proxy_args)
            func.ReturnOp([])

        return builder.emit_ir()

    def __call__(self):
        """Trace and print the IR to stdout."""
        print(self.emit_ir())


def kernel(fn):
    """Decorator that turns a Python function into a PTO kernel.

    Usage::

        @pto.kernel
        def vector_add(x: pto.Tensor(pto.float16, 2), ...):
            ...

        vector_add()  # prints MLIR to stdout
    """
    return KernelFunction(fn, fn.__name__)

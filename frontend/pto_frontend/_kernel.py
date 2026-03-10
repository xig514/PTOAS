"""@kernel decorator: signature parsing, Tensor flattening, tracing, and compilation."""
import inspect
import os
import pathlib
import subprocess

from mlir.ir import InsertionPoint, IndexType, F32Type, IntegerType
from mlir.dialects import func

from ._ir_builder import IRBuilder, set_builder, clear_builder
from ._tensor import _TensorSpec, _TensorShapeSpec, _TensorProxy
from ._dynvar import DynVar
from ._scalar import ScalarValue


# -- dtype name → C++ element type --
_DTYPE_TO_CPP = {
    "float16": "half",
    "bfloat16": "__bf16",
    "float32": "float",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
}

class KernelFunction:
    """Wrapper returned by ``@pto.kernel``.

    Supports the full compilation pipeline:
      emit_ir()  → PTO IR string
      emit_cpp() → C++ source via ptoas
      compile()  → shared library (.so) via bisheng
    """

    def __init__(self, fn, name):
        self._fn = fn
        self._name = name
        self._ir_cache = None
        self._cpp_cache = None
        self._lib_path = None
        self._output_dir = None
        self._param_specs = None  # saved from tracing


    # -- IR Generation (existing, now cached) --
    def emit_ir(self):
        """Trace the kernel and return the MLIR module as a string. Cached."""
        if self._ir_cache is None:
            builder = IRBuilder()
            set_builder(builder)
            try:
                self._ir_cache = self._trace(builder)
            finally:
                builder.close()
                clear_builder()
        return self._ir_cache


    # -- C++ Emission --
    def emit_cpp(self, *, pto_level="level3", arch="a3"):
        """IR → C++ via ptoas subprocess. Cached."""
        if self._cpp_cache is not None:
            return self._cpp_cache
        ir = self.emit_ir()
        out_dir = self._ensure_output_dir()
        pto_path = out_dir / "kernel.pto"
        cpp_path = out_dir / "kernel.cpp"
        pto_path.write_text(ir, encoding="utf-8")
        subprocess.run(
            ["ptoas", str(pto_path),
             f"--pto-level={pto_level}",
             f"--pto-arch={arch}",
             "-o", str(cpp_path)],
            check=True,
            cwd=str(out_dir),
        )
        self._cpp_cache = cpp_path.read_text(encoding="utf-8")
        return self._cpp_cache


    # -- Full Compilation --
    def compile(self, *, pto_level="level3", arch="a3", npu_arch="dav-2201"):
        if self._lib_path is not None and self._lib_path.exists():
            return self
        """IR → C++ → .so via ptoas + bisheng. Returns self."""
        if self._lib_path and self._lib_path.exists():
            return self
        self.emit_cpp(pto_level=pto_level, arch=arch)
        out_dir = self._ensure_output_dir()
        caller_path = out_dir / "caller.cpp"
        lib_path = out_dir / "kernel.so"

        caller_path.write_text(
            self._generate_caller_cpp("kernel.cpp"), encoding="utf-8"
        )
        self._compile_with_bisheng(caller_path, lib_path, npu_arch)
        self._lib_path = lib_path
        return self


    # -- Print IR (original __call__ behavior) --
    def __call__(self):
        """Trace and print the IR to stdout."""
        print(self.emit_ir())


    # -- Properties --
    @property
    def library_path(self):
        return str(self._lib_path) if self._lib_path else None


    # -- Internal helpers --
    def _ensure_output_dir(self):
        if self._output_dir is None:
            self._output_dir = pathlib.Path.cwd() / ".ptodsl_jit" / self._name
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir


    def _generate_caller_cpp(self, kernel_cpp_name):
        """Generate extern "C" wrapper that calls the __global__ kernel."""
        if self._param_specs is None:
            raise RuntimeError("emit_ir() must be called before generating caller.")

        cpp_params = []
        kernel_args = []
	
        for pname, spec in self._param_specs:
            if isinstance(spec, (_TensorSpec, _TensorShapeSpec)):
                elem_cpp = _DTYPE_TO_CPP[spec.dtype.name]
                cpp_params.append(f"uint8_t* {pname}")
                kernel_args.append(f"({elem_cpp}*){pname}")
                for d in range(spec.ndim):
                    dim_name = f"{pname}_dim{d}"
                    cpp_params.append(f"int32_t {dim_name}")
                    kernel_args.append(dim_name)
            elif spec == "index":
                cpp_params.append(f"int32_t {pname}")
                kernel_args.append(pname)
            elif spec == "f32":
                cpp_params.append(f"float {pname}")
                kernel_args.append(pname)
            elif spec == "i1":
                cpp_params.append(f"int32_t {pname}")
                kernel_args.append(pname)

        sig = ", ".join(["uint32_t blockDim", "void* stream"] + cpp_params)
        call_args = ", ".join(kernel_args)

        return (
            f'#include "{kernel_cpp_name}"\n'
            f'extern "C" void call_kernel({sig})\n'
            "{\n"
            f"    {self._name}<<<blockDim, nullptr, stream>>>({call_args});\n"
            "}\n"
        )

    def _compile_with_bisheng(self, caller_path, lib_path, npu_arch):
        """Invoke bisheng compiler: C++ → .so"""
        toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME")
        if not toolkit_home:
            raise RuntimeError(
                "ASCEND_TOOLKIT_HOME is required to compile generated caller.cpp."
            )

        # Detect section guards in the generated C++ and add -D flags so
        # that the guarded code is actually compiled (section_cube emits
        # #if defined(__DAV_CUBE__), section_vector emits __DAV_VEC__).
        dav_defines = []
        if self._cpp_cache:
            if "__DAV_CUBE__" in self._cpp_cache:
                dav_defines.append("-D__DAV_CUBE__")
            if "__DAV_VEC__" in self._cpp_cache:
                dav_defines.append("-D__DAV_VEC__")

        cmd = [
            "bisheng",
            f"-I{toolkit_home}/include",
            "-fPIC",
            "-shared",
            "-D_FORTIFY_SOURCE=2",
            "-O2",
            "-std=c++17",
            "-Wno-macro-redefined",
            "-Wno-ignored-attributes",
            "-fstack-protector-strong",
            "-xcce",
            "-Xhost-start",
            "-Xhost-end",
            "-mllvm", "-cce-aicore-stack-size=0x8000",
            "-mllvm", "-cce-aicore-function-stack-size=0x8000",
            "-mllvm", "-cce-aicore-record-overflow=true",
            "-mllvm", "-cce-aicore-addr-transform",
            "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
            f"--cce-aicore-arch={npu_arch}",
            "-DMEMORY_BASE",
            *dav_defines,
            str(caller_path),
            "-o", str(lib_path),
        ]
        print("CMD:", " ".join(cmd) if isinstance(cmd, list) else cmd)
        subprocess.run(cmd, check=True, cwd=str(self._ensure_output_dir()))

    def _trace(self, builder):
        hints = {
            k: v
            for k, v in self._fn.__annotations__.items()
            if k != "return"
        }
        sig = inspect.signature(self._fn)
        # Only include positional parameters
        params = [
            name for name, p in sig.parameters.items()
            if p.kind not in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]

        # -- build flattened arg types --
        flat_types = []
        param_specs = []  # (param_name, spec_or_tag)
        for pname in params:
            spec = hints.get(pname)
            if isinstance(spec, (_TensorSpec, _TensorShapeSpec)):
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

        # Save param_specs for caller.cpp generation
        self._param_specs = param_specs

        fn_type = func.FunctionType.get(flat_types, [])

        # -- create func.func --
        with InsertionPoint(builder.module.body):
            fn_op = func.FuncOp(self._name, fn_type)
            entry = fn_op.add_entry_block()

        # -- reconstruct proxy objects from block args --
        block_args = list(entry.arguments)
        proxy_args = []
        dynvars_to_unbind = []  # track DynVars for cleanup
        idx = 0
        for _pname, spec in param_specs:
            if isinstance(spec, (_TensorSpec, _TensorShapeSpec)):
                ptr_ssa = block_args[idx]
                idx += 1
                shape_ssas = []
                for d in range(spec.ndim):
                    dim_ssa = block_args[idx]
                    shape_ssas.append(dim_ssa)
                    # Bind DynVar (first occurrence wins)
                    if isinstance(spec, _TensorShapeSpec):
                        dim_spec = spec.shape[d]
                        if isinstance(dim_spec, DynVar) and not dim_spec.is_bound:
                            dim_spec._bind(ScalarValue(dim_ssa))
                            dynvars_to_unbind.append(dim_spec)
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
        try:
            with InsertionPoint(entry):
                self._fn(*proxy_args)
                func.ReturnOp([])
        finally:
            # Always unbind DynVars to prevent stale bindings
            for dv in dynvars_to_unbind:
                dv._unbind()

        return builder.emit_ir()


def kernel(fn=None):
    """Decorator that turns a Python function into a PTO kernel.

    Usage::

        # Simple usage:
        @pto.kernel
        def vector_add(x: pto.Tensor(pto.float16, 2), ...):
            ...

        vector_add()          # prints MLIR to stdout
        vector_add.emit_ir()  # returns IR string
        vector_add.emit_cpp() # returns C++ string (via ptoas)
        vector_add.compile()  # compiles to .so (via bisheng)
    """
    if fn is not None:
        # Called as @pto.kernel (no parentheses)
        return KernelFunction(fn, fn.__name__)
    # Called as @pto.kernel(...) (with parentheses)
    def decorator(fn):
        return KernelFunction(fn, fn.__name__)
    return decorator

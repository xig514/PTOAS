"""Data type definitions mapping user-facing names to MLIR type factories."""

from mlir.ir import F16Type, F32Type, BF16Type, IntegerType


class DType:
    """Wraps an MLIR element type with metadata."""

    def __init__(self, name, mlir_factory, byte_size):
        self.name = name
        self._mlir_factory = mlir_factory
        self.byte_size = byte_size

    def to_mlir(self):
        """Create the MLIR type (must be called within an active Context)."""
        return self._mlir_factory()

    def __repr__(self):
        return f"pto.{self.name}"


float16 = DType("float16", F16Type.get, 2)
float32 = DType("float32", F32Type.get, 4)
bfloat16 = DType("bfloat16", BF16Type.get, 2)
int8 = DType("int8", lambda: IntegerType.get_signless(8), 1)
int16 = DType("int16", lambda: IntegerType.get_signless(16), 2)
int32 = DType("int32", lambda: IntegerType.get_signless(32), 4)
int64 = DType("int64", lambda: IntegerType.get_signless(64), 8)

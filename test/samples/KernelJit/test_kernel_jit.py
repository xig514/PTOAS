"""Test @kernel + @jit decorator architecture.

Demonstrates:
1. @pto.kernel defines device-side NPU code
2. kernel.emit_ir() → PTO IR string
3. kernel.emit_cpp() → C++ via ptoas (needs ptoas in PATH)
4. kernel.compile() → .so via bisheng (needs ASCEND_TOOLKIT_HOME)
5. @pto.jit + kernel[grid](args) → auto-compile on first call

Usage:
  # Phase 1 — IR only (always works):
  python3 test_kernel_jit.py --ir-only

  # Phase 2 — IR + C++ (needs ptoas):
  python3 test_kernel_jit.py --cpp-only

  # Phase 3 — Full compile (needs ptoas + bisheng + ASCEND_TOOLKIT_HOME):
  python3 test_kernel_jit.py

  # Phase 4 — @jit test (needs ptoas + bisheng + ASCEND_TOOLKIT_HOME):
  python3 test_kernel_jit.py --jit
"""

import argparse
import pathlib
import shutil
import sys

import pto_frontend as pto


# ============================================================================
# Define a simple vector_add kernel
# ============================================================================

@pto.kernel
def vector_add(
    x:   pto.Tensor(pto.float16, 2),
    y:   pto.Tensor(pto.float16, 2),
    out: pto.Tensor(pto.float16, 2),
):
    """Simple vector addition: out = x + y, tile-based."""
    TILE_M = 32
    TILE_N = 32

    core_id = pto.get_block_idx()

    x_part = x.partition(offsets=[core_id * TILE_M, 0], sizes=[TILE_M, TILE_N])
    y_part = y.partition(offsets=[core_id * TILE_M, 0], sizes=[TILE_M, TILE_N])
    o_part = out.partition(offsets=[core_id * TILE_M, 0], sizes=[TILE_M, TILE_N])

    tx = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC)
    ty = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC)
    to = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC)

    pto.tload(x_part, tx)
    pto.tload(y_part, ty)
    pto.tadd(tx, ty, to)
    pto.tstore(o_part, to)


# ============================================================================
# Tests
# ============================================================================

def test_emit_ir():
    """Test Phase 1: IR generation."""
    print("=" * 60)
    print("Phase 1: emit_ir()")
    print("=" * 60)
    ir = vector_add.emit_ir()
    assert "func.func" in ir, "IR should contain func.func"
    assert "vector_add" in ir, "IR should contain kernel name"
    print(ir)
    print("PASS: emit_ir()\n")


def test_emit_cpp():
    """Test Phase 2: C++ emission via ptoas."""
    print("=" * 60)
    print("Phase 2: emit_cpp()")
    print("=" * 60)
    if not shutil.which("ptoas"):
        print("SKIP: ptoas not in PATH")
        return False
    cpp = vector_add.emit_cpp()
    assert "__global__" in cpp or "void" in cpp, "C++ should contain function definition"
    print(f"Generated C++ ({len(cpp)} chars):")
    # Print first 20 lines
    for line in cpp.splitlines()[:20]:
        print(f"  {line}")
    print("  ...")
    print("PASS: emit_cpp()\n")
    return True


def test_compile():
    """Test Phase 3: Full compilation to .so."""
    print("=" * 60)
    print("Phase 3: compile()")
    print("=" * 60)
    if not shutil.which("bisheng"):
        print("SKIP: bisheng not in PATH")
        return False
    import os
    if not os.environ.get("ASCEND_TOOLKIT_HOME"):
        print("SKIP: ASCEND_TOOLKIT_HOME not set")
        return False
    vector_add.compile()
    lib = vector_add.library_path
    assert lib is not None, "library_path should be set after compile()"
    assert pathlib.Path(lib).exists(), f".so should exist at {lib}"
    print(f"Compiled to: {lib}")
    print(f"File size: {pathlib.Path(lib).stat().st_size} bytes")
    print("PASS: compile()\n")
    return True


def test_jit():
    """Test Phase 4: @jit + kernel[grid] syntax."""
    print("=" * 60)
    print("Phase 4: @jit + kernel[grid]")
    print("=" * 60)
    if not shutil.which("bisheng"):
        print("SKIP: bisheng not in PATH")
        return False
    import os
    if not os.environ.get("ASCEND_TOOLKIT_HOME"):
        print("SKIP: ASCEND_TOOLKIT_HOME not set")
        return False

    @pto.jit
    def run():
        vector_add()  # triggers compile
        return "ok"

    result = run()
    assert result == "ok", "jit function should return its result"
    assert vector_add.library_path is not None, "kernel should be compiled"
    assert pathlib.Path(vector_add.library_path).exists(), ".so should exist"
    print(f"Library: {vector_add.library_path}")
    print("PASS: @jit + kernel[grid]\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test @kernel + @jit decorators")
    parser.add_argument("--ir-only", action="store_true", help="Only test IR generation")
    parser.add_argument("--cpp-only", action="store_true", help="Test IR + C++ generation")
    parser.add_argument("--jit", action="store_true", help="Also test @jit decorator")
    args = parser.parse_args()

    test_emit_ir()

    if args.ir_only:
        return

    if not test_emit_cpp():
        return

    if args.cpp_only:
        return

    if not test_compile():
        return

    if args.jit:
        test_jit()

    print("All tests passed!")


if __name__ == "__main__":
    main()

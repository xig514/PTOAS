"""TRowMax test cases — frontend equivalents of pto-isa trowmax_kernel.cpp.

Implements all 18 cases from pto-isa tests/npu/a2a3/src/st/testcase/trowmax/.
Each case: TLOAD → sync → TROWMAX → sync → TSTORE.

Usage:
  # Generate IR for all cases:
  python3 test_trowmax.py --ir-only

  # Generate IR + C++ (needs ptoas):
  python3 test_trowmax.py --cpp-only

  # Full compile to .so (needs ptoas + bisheng + ASCEND_TOOLKIT_HOME):
  python3 test_trowmax.py
"""

import argparse
import pathlib
import shutil
import os
import sys

import pto_frontend as pto


# ---------------------------------------------------------------------------
# Case definitions: (case_id, dtype, row, validRow, srcCol, srcValidCol, dstCol)
# Cases 1-14: runTRowMax pattern
# Cases 15-18: runTRowMaxDNDst pattern (same frontend logic, different tile layout)
# ---------------------------------------------------------------------------

CASES_TROWMAX = [
    # (case_id, dtype,       row, validRow, srcCol, srcValidCol, dstCol)
    (1,  pto.float32,  127, 127,  64,  63, 1),
    (2,  pto.float32,   63,  63,  64,  64, 1),
    (3,  pto.float32,   31,  31, 128, 127, 1),
    (4,  pto.float32,   15,  15, 192, 192, 1),
    (5,  pto.float32,    7,   7, 448, 447, 1),
    (6,  pto.float16,  256, 256,  16,  15, 1),
    (7,  pto.float32,   30,  30, 216, 216, 1),
    (8,  pto.float32,   30,  30, 216,  24, 1),
    (9,  pto.float32,   30,  11, 216, 216, 1),
    (10, pto.float32,   30,  11, 216,  24, 1),
    (11, pto.float32,  238, 238,  40,  40, 1),
    (12, pto.float32,  238, 238,  40,  16, 1),
    (13, pto.float32,  238, 121,  40,  40, 1),
    (14, pto.float32,  238, 121,  40,  16, 1),
]

CASES_TROWMAX_DN = [
    # (case_id, dtype,       row, validRow, srcCol, srcValidCol, dstCol)
    (15, pto.float32,  64,  64, 128, 128, 1),
    (16, pto.float32,  32,  32, 256, 256, 1),
    (17, pto.float32,  16,  16, 512, 512, 1),
    (18, pto.float32,   8,   8, 1024, 1024, 1),
]

ALL_CASES = CASES_TROWMAX + CASES_TROWMAX_DN


def make_trowmax_kernel(case_id, dtype, row, valid_row, src_col, src_valid_col, dst_col):
    """Create a @pto.kernel for a single TRowMax test case.

    Equivalent to runTRowMax<T, row, validRow, srcCol, srcValidCol, dstCol>.

    Pattern:
      TLOAD(srcTile, srcGlobal)
      sync(MTE2 -> V)
      TROWMAX(dstTile, srcTile, tmpTile)
      sync(V -> MTE3)
      TSTORE(dstGlobal, dstTile)
    """

    @pto.kernel
    def trowmax_case(
        out: pto.Tensor(dtype, 2),
        src: pto.Tensor(dtype, 2),
    ):
        # --- Partition src as [validRow x srcValidCol] from [row x srcCol] ---
        src_part = src.partition(offsets=[0, 0], sizes=[valid_row, src_valid_col])
        # --- Partition out as [validRow x dstCol] ---
        out_part = out.partition(offsets=[0, 0], sizes=[valid_row, dst_col])

        # --- Allocate tiles ---
        byte_size = dtype.byte_size
        src_tile = pto.make_tile(
            (row, src_col), dtype, pto.VEC, addr=0,
            valid_shape=(valid_row, src_valid_col),
        )
        tmp_tile = pto.make_tile(
            (row, src_col), dtype, pto.VEC,
            addr=row * src_col * byte_size,
            valid_shape=(valid_row, src_valid_col),
        )
        # dst tile: physical (row x 16) for alignment, valid (validRow x dstCol)
        dst_tile = pto.make_tile(
            (row, 16), dtype, pto.VEC,
            addr=2 * row * src_col * byte_size,
            valid_shape=(valid_row, dst_col),
        )

        # --- Pipeline ---
        pto.tload(src_part, src_tile)
        pto.record_event(pto.TLOAD, pto.TVEC, pto.EVENT_ID0)
        pto.wait_event(pto.TLOAD, pto.TVEC, pto.EVENT_ID0)
        pto.trowmax(src_tile, tmp_tile, dst_tile)
        pto.record_event(pto.TVEC, pto.TSTORE_VEC, pto.EVENT_ID0)
        pto.wait_event(pto.TVEC, pto.TSTORE_VEC, pto.EVENT_ID0)
        pto.tstore(out_part, dst_tile)

    # Rename for clarity
    name = f"launchTROWMAXCase{case_id}"
    trowmax_case._name = name
    return trowmax_case


# ---------------------------------------------------------------------------
# Build all kernel objects
# ---------------------------------------------------------------------------

kernels = {}
for case_id, dtype, row, valid_row, src_col, src_valid_col, dst_col in ALL_CASES:
    kernels[case_id] = make_trowmax_kernel(
        case_id, dtype, row, valid_row, src_col, src_valid_col, dst_col
    )


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_emit_ir(case_ids=None):
    """Test Phase 1: IR generation for all cases."""
    ids = case_ids or sorted(kernels.keys())
    for cid in ids:
        k = kernels[cid]
        print(f"{'='*60}")
        print(f"Case {cid}: {k._name} — emit_ir()")
        print(f"{'='*60}")
        ir = k.emit_ir()
        assert "func.func" in ir, f"Case {cid}: IR should contain func.func"
        assert k._name in ir, f"Case {cid}: IR should contain kernel name"
        print(ir)
        print(f"PASS: Case {cid} emit_ir()\n")


def test_emit_cpp(case_ids=None):
    """Test Phase 2: C++ emission via ptoas."""
    if not shutil.which("ptoas"):
        print("SKIP: ptoas not in PATH")
        return False
    ids = case_ids or sorted(kernels.keys())
    for cid in ids:
        k = kernels[cid]
        print(f"{'='*60}")
        print(f"Case {cid}: {k._name} — emit_cpp()")
        print(f"{'='*60}")
        cpp = k.emit_cpp()
        assert "TROWMAX" in cpp or "TLOAD" in cpp, f"Case {cid}: C++ should contain tile ops"
        print(f"Generated C++ ({len(cpp)} chars):")
        for line in cpp.splitlines()[:25]:
            print(f"  {line}")
        if len(cpp.splitlines()) > 25:
            print("  ...")
        print(f"PASS: Case {cid} emit_cpp()\n")
    return True


def test_compile(case_ids=None):
    """Test Phase 3: Full compilation to .so."""
    if not shutil.which("bisheng"):
        print("SKIP: bisheng not in PATH")
        return False
    if not os.environ.get("ASCEND_TOOLKIT_HOME"):
        print("SKIP: ASCEND_TOOLKIT_HOME not set")
        return False
    ids = case_ids or sorted(kernels.keys())
    for cid in ids:
        k = kernels[cid]
        print(f"{'='*60}")
        print(f"Case {cid}: {k._name} — compile()")
        print(f"{'='*60}")
        k.compile()
        lib = k.library_path
        assert lib is not None, f"Case {cid}: library_path should be set"
        assert pathlib.Path(lib).exists(), f"Case {cid}: .so should exist at {lib}"
        size = pathlib.Path(lib).stat().st_size
        print(f"Compiled to: {lib} ({size} bytes)")
        print(f"PASS: Case {cid} compile()\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test TRowMax kernels")
    parser.add_argument("--ir-only", action="store_true", help="Only test IR generation")
    parser.add_argument("--cpp-only", action="store_true", help="Test IR + C++ generation")
    parser.add_argument("--case", type=int, nargs="*", help="Run specific case(s)")
    args = parser.parse_args()

    case_ids = args.case

    test_emit_ir(case_ids)

    if args.ir_only:
        return

    if not test_emit_cpp(case_ids):
        return

    if args.cpp_only:
        return

    if not test_compile(case_ids):
        return

    print("="*60)
    print("All TRowMax tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()

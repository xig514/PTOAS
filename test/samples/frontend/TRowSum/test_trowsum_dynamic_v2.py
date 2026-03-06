"""TRowSum with @pto_meta_data — clean kernel signature + static/dynamic metadata.

Key improvements:
1. Kernel signature only has tensor parameters (src, out)
2. All metadata (phys_row, phys_col, dtype, etc.) in @pto_meta_data
3. Supports both static (compile-time) and dynamic (runtime) metadata
4. Dynamic metadata extracted from tensor.shape at runtime

Usage:
  python3 test_trowsum_dynamic_v2.py --ir-only
  python3 test_trowsum_dynamic_v2.py --cpp-only
  python3 test_trowsum_dynamic_v2.py
"""

import argparse
import pathlib
import shutil
import os

import pto_frontend as pto


# ---------------------------------------------------------------------------
# Case definitions: (case_id, dtype, phys_row, phys_col, dst_col)
# ---------------------------------------------------------------------------

CASES = [
    (1,  pto.float32, 128,  64, 1),
    (2,  pto.float32,  64,  64, 1),
    (3,  pto.float32,  32, 128, 1),
    (4,  pto.float32,  16, 192, 1),
    (5,  pto.float32,   8, 448, 1),
    (6,  pto.float16, 256,  16, 1),
    (7,  pto.float32,  32, 256, 1),
    (8,  pto.float32,  64, 128, 1),
]


# ---------------------------------------------------------------------------
# Kernel factory using @pto_meta_data
# ---------------------------------------------------------------------------

def _define_kernel(case_id, dtype, phys_row, phys_col, dst_col):
    """Define a kernel with metadata decorator."""

    # Step 1: Define metadata function
    @pto.pto_meta_data
    def meta_data():
        return {
            "phys_row": phys_row,
            "phys_col": phys_col,
            "dst_col": dst_col,
            "dtype": dtype,
            # Mark dynamic metadata (extracted from tensor.shape at runtime)
            "valid_row": "dynamic",
            "valid_col": "dynamic",
        }

    # Step 2: Define kernel with clean signature
    @pto.kernel(metadata=meta_data)
    def trowsum_case(
        src: pto.Tensor(dtype, 2),
        out: pto.Tensor(dtype, 2),
    ):
        """TROWSUM kernel with metadata from @pto_meta_data."""

        # --- Extract static metadata ---
        static_meta = meta_data.get_static_metadata()
        phys_row = static_meta["phys_row"]
        phys_col = static_meta["phys_col"]
        dst_col = static_meta["dst_col"]
        dtype = static_meta["dtype"]

        # --- Extract dynamic metadata from tensor.shape ---
        valid_row = src.shape[0]
        valid_col = src.shape[1]

        # --- Static partition: use physical sizes (ptoas requires static partition sizes) ---
        src_part = src.partition(offsets=[0, 0], sizes=[phys_row, phys_col])
        out_part = out.partition(offsets=[0, 0], sizes=[phys_row, dst_col])

        # --- Allocate tiles with dynamic valid shapes ---
        from mlir.dialects import pto as _pto
        from pto_frontend._ir_builder import get_builder
        from pto_frontend._ops import Tile

        builder = get_builder()
        byte_size = dtype.byte_size

        bl_attr = _pto.BLayoutAttr.get(_pto.BLayout.RowMajor)
        sl_attr = _pto.SLayoutAttr.get(_pto.SLayout.NoneBox)
        pd_attr = _pto.PadValueAttr.get(_pto.PadValue.Null)
        cfg = _pto.TileBufConfigAttr.get(bl_attr, sl_attr, 512, pd_attr)
        addr_space_attr = _pto.AddressSpaceAttr.get(pto.VEC)
        mlir_elem = dtype.to_mlir()

        # srcTile: physical (phys_row x phys_col), valid dynamic
        tb_type_src = _pto.TileBufType.get(
            [phys_row, phys_col], mlir_elem, addr_space_attr, [-1, -1], cfg
        )
        src_tile = Tile(
            _pto.AllocTileOp(
                tb_type_src, addr=builder.constant_i64(0),
                valid_row=valid_row.ssa, valid_col=valid_col.ssa,
            ).result,
            (phys_row, phys_col), dtype, pto.VEC,
        )

        # tmpTile: same type as srcTile
        tmp_tile = Tile(
            _pto.AllocTileOp(
                tb_type_src,
                addr=builder.constant_i64(phys_row * phys_col * byte_size),
                valid_row=valid_row.ssa, valid_col=valid_col.ssa,
            ).result,
            (phys_row, phys_col), dtype, pto.VEC,
        )

        # dstTile: physical (phys_row x 16), valid_row dynamic, dst_col static
        tb_type_dst = _pto.TileBufType.get(
            [phys_row, 16], mlir_elem, addr_space_attr, [-1, dst_col], cfg
        )
        dst_tile = Tile(
            _pto.AllocTileOp(
                tb_type_dst,
                addr=builder.constant_i64(2 * phys_row * phys_col * byte_size),
                valid_row=valid_row.ssa,
            ).result,
            (phys_row, 16), dtype, pto.VEC,
        )

        # --- Pipeline ---
        pto.tload(src_part, src_tile)
        pto.record_event(pto.TLOAD, pto.TVEC, pto.EVENT_ID0)
        pto.wait_event(pto.TLOAD, pto.TVEC, pto.EVENT_ID0)

        pto.trowsum(src_tile, tmp_tile, dst_tile)

        pto.record_event(pto.TVEC, pto.TSTORE_VEC, pto.EVENT_ID0)
        pto.wait_event(pto.TVEC, pto.TSTORE_VEC, pto.EVENT_ID0)
        pto.tstore(out_part, dst_tile)

    trowsum_case._name = f"launchTROWSUMCase{case_id}"
    return trowsum_case


# ---------------------------------------------------------------------------
# Build all kernel objects
# ---------------------------------------------------------------------------

kernels = {}
for case_id, dtype, phys_row, phys_col, dst_col in CASES:
    kernels[case_id] = _define_kernel(case_id, dtype, phys_row, phys_col, dst_col)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_emit_ir(case_ids=None):
    ids = case_ids or sorted(kernels.keys())
    for cid in ids:
        k = kernels[cid]
        print(f"{'='*60}")
        print(f"Case {cid}: {k._name} — emit_ir()")
        print(f"{'='*60}")
        ir = k.emit_ir()
        assert "func.func" in ir
        assert k._name in ir
        print(ir)
        print(f"PASS: Case {cid} emit_ir()\n")
    return True


def test_emit_cpp(case_ids=None):
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
        assert "__global__" in cpp or "void" in cpp
        print(f"Generated C++ ({len(cpp)} chars):")
        for line in cpp.splitlines()[:30]:
            print(f"  {line}")
        if len(cpp.splitlines()) > 30:
            print("  ...")
        print(f"PASS: Case {cid} emit_cpp()\n")
    return True


def test_compile(case_ids=None):
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
        assert lib is not None
        assert pathlib.Path(lib).exists()
        size = pathlib.Path(lib).stat().st_size
        print(f"Compiled to: {lib} ({size} bytes)")
        print(f"PASS: Case {cid} compile()\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test TRowSum with @pto_meta_data")
    parser.add_argument("--ir-only", action="store_true")
    parser.add_argument("--cpp-only", action="store_true")
    parser.add_argument("--case", type=int, nargs="*")
    args = parser.parse_args()

    test_emit_ir(args.case)
    if args.ir_only:
        return
    if not test_emit_cpp(args.case):
        return
    if args.cpp_only:
        return
    if not test_compile(args.case):
        return

    print("="*60)
    print("All TRowSum @pto_meta_data tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()

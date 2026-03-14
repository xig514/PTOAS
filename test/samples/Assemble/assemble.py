from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F16Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            idx = IndexType.get(ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)

            src_ty = pto.TileBufType.get([16, 16], f16, vec, [16, 16], cfg, ctx)
            dst_ty = pto.TileBufType.get([32, 32], f16, mat, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("assemble_demo", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c16 = arith.ConstantOp(idx, 16).result

                src = pto.AllocTileOp(src_ty).result
                dst = pto.AllocTileOp(dst_ty).result
                pto.TAssembleOp(src, c16, c16, dst)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())

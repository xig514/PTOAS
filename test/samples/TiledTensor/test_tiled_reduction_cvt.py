"""Test: Tiled reduction ops, row-expand ops, and type conversion.

Covers: trowmax, trowsum, trowexpand, trowexpandsub, trowexpanddiv,
        trowexpandmul, tcvt, tmov (VEC→VEC).
"""
import pto_frontend as pto


@pto.kernel
def tiled_softmax(
    x: pto.Tensor(pto.float32, 2),
    z: pto.Tensor(pto.float16, 2),
):
    """Row-wise softmax: z = softmax(x, dim=-1).

    Steps: m = rowmax(x)
           x = x - broadcast(m)
           x = exp(x)
           l = rowsum(x)
           x = x / broadcast(l)
           z = convert(x, float16)
    """
    TILE = 32

    tile_in = pto.make_tile((TILE, TILE), pto.float32, pto.VEC, addr=0)
    tile_tmp = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                             addr=TILE * TILE * 4)
    tile_out_f16 = pto.make_tile((TILE, TILE), pto.float16, pto.VEC,
                                 addr=TILE * TILE * 8)

    x_tiled = x.tile(dim=0, tile_sizes=(TILE, TILE))
    z_tiled = z.tile(dim=0, tile_sizes=(TILE, TILE))

    with x_tiled.for_each() as (i, x_view):
        z_view = z_tiled[i]

        pto.tload(tile_in, x_view)

        # Row-wise softmax using reduction + broadcast ops
        # 1) row max for numerical stability
        pto.trowmax(tile_tmp, tile_in, tile_tmp)
        # 2) subtract broadcast(max) from each element
        pto.trowexpandsub(tile_in, tile_in, tile_tmp)
        # 3) exponentiate
        pto.texp(tile_in, tile_in)
        # 4) row sum
        pto.trowsum(tile_tmp, tile_in, tile_tmp)
        # 5) divide by broadcast(sum)
        pto.trowexpanddiv(tile_in, tile_in, tile_tmp)

        # 6) convert f32 → f16
        pto.tcvt(tile_out_f16, tile_in)

        pto.tstore(z_view, tile_out_f16)


@pto.kernel
def tiled_row_broadcast_ops(
    x: pto.Tensor(pto.float32, 2),
    scale: pto.Tensor(pto.float32, 2),
    z: pto.Tensor(pto.float32, 2),
):
    """Demonstrate row-expand multiply: z[i,j] = x[i,j] * scale[i,0]."""
    TILE = 32

    tile_x = pto.make_tile((TILE, TILE), pto.float32, pto.VEC, addr=0)
    tile_s = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                           addr=TILE * TILE * 4)
    tile_z = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                           addr=TILE * TILE * 8)

    x_tiled = x.tile(dim=0, tile_sizes=(TILE, TILE))
    s_tiled = scale.tile(dim=0, tile_sizes=(TILE, TILE))
    z_tiled = z.tile(dim=0, tile_sizes=(TILE, TILE))

    with x_tiled.for_each() as (i, x_view):
        s_view = s_tiled[i]
        z_view = z_tiled[i]

        pto.tload(tile_x, x_view)
        pto.tload(tile_s, s_view)

        # Row-wise broadcast multiply
        pto.trowexpandmul(tile_z, tile_x, tile_s)

        pto.tstore(z_view, tile_z)


if __name__ == "__main__":
    tiled_softmax()

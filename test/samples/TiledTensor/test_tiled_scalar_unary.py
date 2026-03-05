"""Test: Tiled scalar and unary operations.

Covers: tadds, tsubs, tmuls, tdivs, tmaxs, tmins,
        texp, tlog, tsqrt, trsqrt, trecip, tneg, trelu, tabs.
"""
import pto_frontend as pto


@pto.kernel
def tiled_scalar_ops(
    x: pto.Tensor(pto.float32, 2),
    z: pto.Tensor(pto.float32, 2),
):
    """Apply various scalar operations on tiled data.

    NOTE: Uses f32 because the PTO dialect has a constraint mismatch for
    tile-scalar ops with non-f32 types (ODS declares F32:$scalar but the
    C++ verifier requires scalar type == element type).  Both constraints
    are satisfied only when the element type is f32.
    """
    TILE = 32

    tile_x = pto.make_tile((TILE, TILE), pto.float32, pto.VEC, addr=0)
    tile_y = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                           addr=TILE * TILE * 4)
    tile_z = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                           addr=TILE * TILE * 8)

    x_tiled = x.tile(dim=0, tile_sizes=(TILE, TILE))
    z_tiled = z.tile(dim=0, tile_sizes=(TILE, TILE))

    with x_tiled.for_each() as (i, x_view):
        z_view = z_tiled[i]

        pto.tload(x_view, tile_x)

        # Scalar operations chain: scale, shift, clamp
        pto.tmuls(tile_x, 0.125, tile_y)    # scale by 1/sqrt(64)
        pto.tadds(tile_y, 1.0, tile_y)      # add bias
        pto.tmaxs(tile_y, 0.0, tile_z)      # clamp lower bound (ReLU-like)
        pto.tmins(tile_z, 6.0, tile_z)      # clamp upper bound (ReLU6)

        pto.tstore(tile_z, z_view)


@pto.kernel
def tiled_unary_ops(
    x: pto.Tensor(pto.float32, 2),
    z: pto.Tensor(pto.float32, 2),
):
    """Apply various unary operations on tiled data."""
    TILE = 32

    tile_a = pto.make_tile((TILE, TILE), pto.float32, pto.VEC, addr=0)
    tile_b = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                           addr=TILE * TILE * 4)
    tile_c = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                           addr=TILE * TILE * 8)

    x_tiled = x.tile(dim=0, tile_sizes=(TILE, TILE))
    z_tiled = z.tile(dim=0, tile_sizes=(TILE, TILE))

    with x_tiled.for_each() as (i, x_view):
        z_view = z_tiled[i]

        pto.tload(x_view, tile_a)

        # Unary ops chain
        pto.tabs(tile_a, tile_b)      # abs
        pto.tsqrt(tile_b, tile_c)     # sqrt(abs(x))
        pto.trecip(tile_c, tile_b)    # 1/sqrt(abs(x))
        pto.texp(tile_a, tile_c)      # exp(x)
        pto.tlog(tile_c, tile_c)      # log(exp(x)) = x
        pto.tneg(tile_c, tile_b)      # -x
        pto.trelu(tile_b, tile_c)     # relu(-x) = 0 for positive x

        pto.tstore(tile_c, z_view)


if __name__ == "__main__":
    tiled_scalar_ops()

"""Test: Conditional control flow within tiled loops.

Covers: if_ (simple), if_ (with else), for_range step>1,
        ScalarValue comparisons inside tiled iteration.
"""
import pto_frontend as pto


@pto.kernel
def tiled_conditional_add(
    x: pto.Tensor(pto.float16, 2),
    y: pto.Tensor(pto.float16, 2),
    z: pto.Tensor(pto.float16, 2),
):
    """Add x+y for even-indexed tiles, copy x for odd-indexed tiles."""
    TILE = 32

    tile_x = pto.make_tile((TILE, TILE), pto.float16, pto.VEC, addr=0)
    tile_y = pto.make_tile((TILE, TILE), pto.float16, pto.VEC,
                           addr=TILE * TILE * 2)
    tile_z = pto.make_tile((TILE, TILE), pto.float16, pto.VEC,
                           addr=TILE * TILE * 4)

    x_tiled = x.tile(dim=0, tile_sizes=(TILE, TILE))
    y_tiled = y.tile(dim=0, tile_sizes=(TILE, TILE))
    z_tiled = z.tile(dim=0, tile_sizes=(TILE, TILE))

    with x_tiled.for_each() as (i, x_view):
        y_view = y_tiled[i]
        z_view = z_tiled[i]

        pto.tload(tile_x, x_view)

        # Conditional: is_even = (i % 2 == 0)
        is_even = (i % 2) == 0

        with pto.if_(is_even, has_else=True) as (then_br, else_br):
            with then_br:
                # Even tiles: z = x + y
                pto.tload(tile_y, y_view)
                pto.tadd(tile_z, tile_x, tile_y)
            with else_br:
                # Odd tiles: z = x (copy via VEC→VEC move)
                pto.tmov(tile_z, tile_x)

        pto.tstore(z_view, tile_z)


@pto.kernel
def tiled_step_iteration(
    x: pto.Tensor(pto.float16, 2),
    z: pto.Tensor(pto.float16, 2),
):
    """Iterate with step=2, processing every other tile."""
    TILE = 32

    tile_x = pto.make_tile((TILE, TILE), pto.float16, pto.VEC, addr=0)
    tile_z = pto.make_tile((TILE, TILE), pto.float16, pto.VEC,
                           addr=TILE * TILE * 2)

    x_tiled = x.tile(dim=0, tile_sizes=(TILE, TILE))
    z_tiled = z.tile(dim=0, tile_sizes=(TILE, TILE))

    # Step=2: process tiles 0, 2, 4, ...
    with x_tiled.for_each(step=2) as (i, x_view):
        z_view = z_tiled[i]

        pto.tload(tile_x, x_view)
        pto.tadd(tile_z, tile_x, tile_x)  # z = 2*x
        pto.tstore(z_view, tile_z)


@pto.kernel
def tiled_boundary_check(
    x: pto.Tensor(pto.float32, 2),
    z: pto.Tensor(pto.float32, 2),
):
    """Use if_ to guard tile processing based on tile index vs limit."""
    TILE = 32

    tile_x = pto.make_tile((TILE, TILE), pto.float32, pto.VEC, addr=0)
    tile_z = pto.make_tile((TILE, TILE), pto.float32, pto.VEC,
                           addr=TILE * TILE * 4)

    num_tiles = (x.shape[0] + (TILE - 1)) // TILE

    # Process tiles with explicit boundary check
    with pto.for_range(0, num_tiles) as i:
        # Guard: only process tiles that are fully within bounds
        remaining = x.shape[0] - i * TILE
        cond = remaining >= TILE

        with pto.if_(cond):
            pv = x.partition([i * TILE, 0], [TILE, TILE])
            pto.tload(tile_x, pv)
            pto.texp(tile_z, tile_x)
            out_pv = z.partition([i * TILE, 0], [TILE, TILE])
            pto.tstore(out_pv, tile_z)


if __name__ == "__main__":
    tiled_conditional_add()

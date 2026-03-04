"""Test: control flow with for_range, if_, and ScalarValue comparisons.

Covers: for_range, if_ (simple), ScalarValue arithmetic and comparisons.
"""
import pto_frontend as pto


@pto.kernel
def control_flow_demo(
    x: pto.Tensor(pto.float32, 2),
    y: pto.Tensor(pto.float32, 2),
):
    M = x.shape[0]

    tile_a = pto.make_tile((32, 32), pto.float32, pto.VEC, addr=0)
    tile_b = pto.make_tile((32, 32), pto.float32, pto.VEC, addr=32 * 32 * 4)

    with pto.for_range(0, M, 32) as i:
        pto.tload(x.partition([i, 0], [32, 32]), tile_a)

        cond = i < M - 32
        with pto.if_(cond):
            pto.tadd(tile_a, tile_a, tile_b)

        pto.tstore(tile_b, y.partition([i, 0], [32, 32]))


if __name__ == "__main__":
    control_flow_demo()

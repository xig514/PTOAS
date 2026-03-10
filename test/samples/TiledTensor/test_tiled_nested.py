"""Test: Nested TiledTensor loops — query/key pattern like FlashAttention.

Covers: Two TiledTensors with nested for_each, distribute on outer,
        full iteration on inner.  Demonstrates the typical multi-core
        matmul / attention tiling pattern.
"""
import pto_frontend as pto


@pto.kernel
def nested_tile_matmul(
    query: pto.Tensor(pto.float16, 2),
    key: pto.Tensor(pto.float16, 2),
    out: pto.Tensor(pto.float16, 2),
):
    TILE_M = 32
    TILE_N = 32

    tile_q = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=0)
    tile_k = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=TILE_M * TILE_N * 2)
    tile_o = pto.make_tile((TILE_M, TILE_N), pto.float16, pto.VEC, addr=TILE_M * TILE_N * 4)

    # Outer: distribute query tiles across cores
    q_dist = query.tile(dim=0, tile_sizes=(TILE_M, TILE_N)).distribute()
    # Inner: each core iterates all key tiles
    k_tiled = key.tile(dim=0, tile_sizes=(TILE_M, TILE_N))
    out_tiled = out.tile(dim=0, tile_sizes=(TILE_M, TILE_N))

    with q_dist.for_each() as (qi, q_view):
        pto.tload(tile_q, q_view)

        with k_tiled.for_each() as (ki, k_view):
            pto.tload(tile_k, k_view)
            pto.tadd(tile_o, tile_q, tile_k)

        o_view = out_tiled[qi]
        pto.tstore(o_view, tile_o)


if __name__ == "__main__":
    nested_tile_matmul()

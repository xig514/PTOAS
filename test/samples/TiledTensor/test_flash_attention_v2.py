"""FlashAttention using new Layout-based API.

Demonstrates coordinate-based tile access for attention computation.
"""

import pto_frontend as pto


@pto.kernel
def flash_attention_layout_v2(
    q: pto.Tensor(pto.float16, 2),      # [S_q, D]
    k: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    v: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    out: pto.Tensor(pto.float16, 2),    # [S_q, D]
):
    """FlashAttention with Layout-based coordinate access."""

    TILE_S = 128
    TILE_D = 64

    # Get layouts
    q_layout = q.get_layout()
    k_layout = k.get_layout()
    v_layout = v.get_layout()

    # Define tile pattern
    tile_layout = pto.TileLayout(shape=(TILE_S, TILE_D))

    # Get core information
    num_cores = pto.get_block_num()
    core_id = pto.get_block_idx()

    # Split Q tiles across cores
    q_tiled = pto.split_even(q_layout, tile_layout, num_cores, core_id)

    # K/V tiles: sequential (all tiles)
    k_tiled = pto.split_sequential(k_layout, tile_layout)
    v_tiled = pto.split_sequential(v_layout, tile_layout)

    # Allocate tile buffers
    tile_q = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0)
    tile_k = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x10000)
    tile_v = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x20000)
    tile_out = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x30000)

    # Outer loop: distributed Q tiles
    with q_tiled.for_each() as q_coord:
        # Load Q tile
        pto.tload_tile(q, q_coord, tile_layout, tile_q)

        # Inner loop: all K/V tiles
        with k_tiled.for_each() as kv_coord:
            # Load K and V tiles
            pto.tload_tile(k, kv_coord, tile_layout, tile_k)
            pto.tload_tile(v, kv_coord, tile_layout, tile_v)

            # Attention computation (simplified)
            pto.tadd(tile_q, tile_k, tile_out)
            pto.tadd(tile_out, tile_v, tile_out)

        # Store output tile
        pto.tstore_tile(tile_out, out, q_coord, tile_layout)


@pto.kernel
def flash_attention_causal(
    q: pto.Tensor(pto.float16, 2),      # [S_q, D]
    k: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    v: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    out: pto.Tensor(pto.float16, 2),    # [S_q, D]
):
    """FlashAttention with causal masking using Layout API."""

    TILE_S = 128
    TILE_D = 64

    q_layout = q.get_layout()
    k_layout = k.get_layout()
    v_layout = v.get_layout()

    tile_layout = pto.TileLayout(shape=(TILE_S, TILE_D))

    num_cores = pto.get_block_num()
    core_id = pto.get_block_idx()

    # Split Q with causal pattern
    q_tiled = pto.split_causal(q_layout, tile_layout, num_cores, core_id)

    # K/V: sequential
    k_tiled = pto.split_sequential(k_layout, tile_layout)
    v_tiled = pto.split_sequential(v_layout, tile_layout)

    tile_q = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0)
    tile_k = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x10000)
    tile_v = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x20000)
    tile_out = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x30000)

    with q_tiled.for_each() as q_coord:
        pto.tload_tile(q, q_coord, tile_layout, tile_q)

        # Causal masking: only attend to K/V up to current Q position
        # For simplicity, iterate all K/V (masking would be done in computation)
        with k_tiled.for_each() as kv_coord:
            # In real implementation, check: if kv_coord[0] <= q_coord[0]
            pto.tload_tile(k, kv_coord, tile_layout, tile_k)
            pto.tload_tile(v, kv_coord, tile_layout, tile_v)

            pto.tadd(tile_q, tile_k, tile_out)
            pto.tadd(tile_out, tile_v, tile_out)

        pto.tstore_tile(tile_out, out, q_coord, tile_layout)


if __name__ == "__main__":
    flash_attention_layout_v2()

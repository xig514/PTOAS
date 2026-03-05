"""FlashAttention using TileLayout for explicit tile indexing.

Demonstrates manual tile iteration using TileLayout instead of distribute_nd.
This gives more explicit control over tile access patterns.
"""

import pto_frontend as pto


@pto.kernel
def flash_attention_with_layout(
    q: pto.Tensor(pto.float16, 2),      # [S_q, D]
    k: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    v: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    out: pto.Tensor(pto.float16, 2),    # [S_q, D]
):
    """FlashAttention using TileLayout for manual tile indexing.

    Uses TileLayout to explicitly compute tile coordinates and access patterns.
    Each core computes its tile range based on core_id and TileLayout mapping.
    """

    TILE_S = 128
    TILE_D = 64

    # Create TileLayout for Q tiles
    # Assume S_q = 512, so we have 512/128 = 4 tiles
    q_tile_layout = pto.TileLayout(shape=(4,))  # 4 Q tiles along S dimension

    # Get core information
    core_id = pto.get_block_idx()
    num_cores = pto.get_block_num()

    # Compute this core's tile range using layout
    # For 4 tiles and 2 cores: core 0 gets tiles [0,2), core 1 gets tiles [2,4)
    num_q_tiles = 4
    tiles_per_core = (num_q_tiles + num_cores - 1) // num_cores

    q_tile_start = core_id * tiles_per_core
    q_tile_end = (core_id + 1) * tiles_per_core

    # Clamp to valid range
    if q_tile_start > num_q_tiles:
        q_tile_start = num_q_tiles
    if q_tile_end > num_q_tiles:
        q_tile_end = num_q_tiles

    # Create TileLayout for K/V tiles
    # Assume S_kv = 512, so we have 512/128 = 4 tiles
    kv_tile_layout = pto.TileLayout(shape=(4,))  # 4 K/V tiles along S dimension

    # Allocate tile buffers
    tile_q = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0)
    tile_k = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x10000)
    tile_v = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x20000)
    tile_out = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x30000)

    # Outer loop: iterate over this core's Q tiles
    with pto.for_range(q_tile_start, q_tile_end) as q_tile_idx:
        # Compute Q tile offset using layout
        q_offset = q_tile_idx * TILE_S

        # Load Q tile - use static partition
        q_partition = q.partition(offsets=[q_offset, 0], sizes=[TILE_S, TILE_D])
        pto.tload(q_partition, tile_q)

        # Inner loop: iterate over all K/V tiles
        num_kv_tiles = 4
        with pto.for_range(0, num_kv_tiles) as kv_tile_idx:
            # Compute K/V tile offset using layout
            kv_offset = kv_tile_idx * TILE_S

            # Load K and V tiles - use static partition
            k_partition = k.partition(offsets=[kv_offset, 0], sizes=[TILE_S, TILE_D])
            v_partition = v.partition(offsets=[kv_offset, 0], sizes=[TILE_S, TILE_D])

            pto.tload(k_partition, tile_k)
            pto.tload(v_partition, tile_v)

            # Compute attention (simplified: just add)
            pto.tadd(tile_q, tile_k, tile_out)
            pto.tadd(tile_out, tile_v, tile_out)

        # Store output tile - use static partition
        out_partition = out.partition(offsets=[q_offset, 0], sizes=[TILE_S, TILE_D])
        pto.tstore(tile_out, out_partition)


@pto.kernel
def flash_attention_2d_layout(
    q: pto.Tensor(pto.float16, 2),      # [S_q, D]
    k: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    v: pto.Tensor(pto.float16, 2),      # [S_kv, D]
    out: pto.Tensor(pto.float16, 2),    # [S_q, D]
):
    """FlashAttention with 2D TileLayout for Q tiles.

    Demonstrates using 2D layout to map (batch, seq) tile indices,
    even though we're working with 2D tensors (simplified from BSND).
    """

    TILE_S = 128
    TILE_D = 64

    # Create 2D TileLayout for conceptual (batch, seq) tile grid
    # In this simplified version: 1 batch x 4 seq tiles
    q_tile_layout = pto.TileLayout(shape=(1, 4), stride=(4, 1))  # row-major

    # Get core information
    core_id = pto.get_block_idx()

    # Map linear core_id to 2D tile coordinates using layout
    # For 1x4 grid with 2 cores:
    # core 0 -> tiles (0,0), (0,1)
    # core 1 -> tiles (0,2), (0,3)

    total_tiles = 4
    tiles_per_core = (total_tiles + 1) // 2  # 2 tiles per core

    tile_start = core_id * tiles_per_core
    tile_end = (core_id + 1) * tiles_per_core
    if tile_end > total_tiles:
        tile_end = total_tiles

    # Allocate tile buffers
    tile_q = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0)
    tile_k = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x10000)
    tile_v = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x20000)
    tile_out = pto.make_tile((TILE_S, TILE_D), pto.float16, pto.VEC, addr=0x30000)

    # Iterate over this core's tiles
    with pto.for_range(tile_start, tile_end) as linear_tile_idx:
        # Compute tile offset (in this case, just linear)
        q_offset = linear_tile_idx * TILE_S

        # Load Q tile - use static partition
        q_partition = q.partition(offsets=[q_offset, 0], sizes=[TILE_S, TILE_D])
        pto.tload(q_partition, tile_q)

        # Inner loop: all K/V tiles
        with pto.for_range(0, 4) as kv_tile_idx:
            kv_offset = kv_tile_idx * TILE_S

            k_partition = k.partition(offsets=[kv_offset, 0], sizes=[TILE_S, TILE_D])
            v_partition = v.partition(offsets=[kv_offset, 0], sizes=[TILE_S, TILE_D])

            pto.tload(k_partition, tile_k)
            pto.tload(v_partition, tile_v)

            # Attention computation (simplified)
            pto.tadd(tile_q, tile_k, tile_out)
            pto.tadd(tile_out, tile_v, tile_out)

        # Store output - use static partition
        out_partition = out.partition(offsets=[q_offset, 0], sizes=[TILE_S, TILE_D])
        pto.tstore(tile_out, out_partition)


if __name__ == "__main__":
    flash_attention_with_layout()

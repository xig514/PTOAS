"""Test: Tiled matmul with correct address space pipeline and sync.

Covers: make_tile (MAT, LEFT, RIGHT, ACC), tload, tmov, tmatmul,
        tmatmul_acc, tstore, record_event, wait_event.

Pipeline:  GM ──TLOAD──▸ MAT ──TMOV──▸ LEFT/RIGHT ──TMATMUL──▸ ACC ──TSTORE──▸ GM
"""
import pto_frontend as pto


@pto.kernel
def tiled_matmul_pipeline(
    a: pto.Tensor(pto.float16, 2),   # [M, K]
    b: pto.Tensor(pto.float16, 2),   # [K, N]
    c: pto.Tensor(pto.float16, 2),   # [M, N]
):
    """Distributed matrix multiply with full pipeline and sync.

    C[i, :] = A[i, :] @ B[:, :]  per-core tile block.
    """
    TILE = 32

    # L1 (MAT) buffers for DMA
    a_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
    b_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                          addr=TILE * TILE * 2)

    # L0 buffers for Cube
    a_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
    b_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)
    c_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)

    # Distribute C rows across cores
    a_tiled = a.tile(dim=0, tile_sizes=(TILE, TILE))
    c_tiled = c.tile(dim=0, tile_sizes=(TILE, TILE))
    b_tiled = b.tile(dim=0, tile_sizes=(TILE, TILE))

    a_dist = a_tiled.distribute()

    with a_dist.for_each() as (i, a_view):
        c_view = c_tiled[i]

        # Load A tile: GM → MAT → LEFT
        pto.tload(a_view, a_mat)
        pto.record_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.wait_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.tmov(a_mat, a_left)

        # Inner loop over K dimension
        with b_tiled.for_each() as (j, b_view):
            # Load B tile: GM → MAT → RIGHT
            pto.tload(b_view, b_mat)
            pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
            pto.tmov(b_mat, b_right)

            # Matmul via Cube unit
            pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
            pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
            pto.tmatmul(a_left, b_right, c_acc)

        # Store result: ACC → GM
        pto.record_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
        pto.wait_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
        pto.tstore(c_view, c_acc)


@pto.kernel
def tiled_matmul_acc(
    a: pto.Tensor(pto.float16, 2),   # [M, K]
    b: pto.Tensor(pto.float16, 2),   # [K, N]
    c: pto.Tensor(pto.float16, 2),   # [M, N]
):
    """Matmul with accumulation (tmatmul_acc) over K tiles."""
    TILE = 32

    a_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT, addr=0)
    b_mat = pto.make_tile((TILE, TILE), pto.float16, pto.MAT,
                          addr=TILE * TILE * 2)
    a_left = pto.make_tile((TILE, TILE), pto.float16, pto.LEFT, addr=0)
    b_right = pto.make_tile((TILE, TILE), pto.float16, pto.RIGHT, addr=0)
    c_acc = pto.make_tile((TILE, TILE), pto.float32, pto.ACC, addr=0)

    a_tiled = a.tile(dim=0, tile_sizes=(TILE, TILE))
    b_tiled = b.tile(dim=0, tile_sizes=(TILE, TILE))

    # Single output tile (first block of C)
    c_part = c[0:TILE, 0:TILE]

    # First K tile: regular matmul
    a_view_0 = a_tiled[0]
    b_view_0 = b_tiled[0]

    pto.tload(a_view_0, a_mat)
    pto.record_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
    pto.wait_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
    pto.tmov(a_mat, a_left)

    pto.tload(b_view_0, b_mat)
    pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
    pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
    pto.tmov(b_mat, b_right)

    pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
    pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
    pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
    pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
    pto.tmatmul(a_left, b_right, c_acc)

    # Remaining K tiles: matmul with accumulate
    with a_tiled.for_each(start=1, end=a_tiled.num_tiles) as (k, a_view):
        b_view = b_tiled[k]

        pto.tload(a_view, a_mat)
        pto.record_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.wait_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
        pto.tmov(a_mat, a_left)

        pto.tload(b_view, b_mat)
        pto.record_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
        pto.wait_event(pto.TLOAD, pto.TMOV_M2S, pto.EVENT_ID1)
        pto.tmov(b_mat, b_right)

        pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
        pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
        pto.record_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
        pto.wait_event(pto.TMOV_M2S, pto.TMATMUL, pto.EVENT_ID1)
        pto.tmatmul_acc(c_acc, a_left, b_right, c_acc)

    # Store
    pto.record_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
    pto.wait_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
    pto.tstore(c_part, c_acc)


if __name__ == "__main__":
    tiled_matmul_pipeline()

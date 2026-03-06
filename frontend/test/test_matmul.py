"""Test: matmul kernel with MAT/LEFT/RIGHT/ACC tiles and sync.

Covers: make_tile (multiple address spaces), tload, tmov, tmatmul,
        tstore, record_event, wait_event.
"""
import pto_frontend as pto


@pto.kernel
def matmul_kernel(
    a: pto.Tensor(pto.float32, 2),
    b: pto.Tensor(pto.float32, 2),
    c: pto.Tensor(pto.float32, 2),
):
    M, K, N = 32, 32, 32

    # L1 (MAT) tiles for DMA from GM
    a_mat = pto.make_tile((M, K), pto.float32, pto.MAT, addr=0)
    b_mat = pto.make_tile((K, N), pto.float32, pto.MAT, addr=M * K * 4)

    # L0 tiles for matmul
    a_left = pto.make_tile((M, K), pto.float32, pto.LEFT, addr=0)
    b_right = pto.make_tile((K, N), pto.float32, pto.RIGHT, addr=0)
    c_acc = pto.make_tile((M, N), pto.float32, pto.ACC, addr=0)

    # Load from GM -> MAT
    pto.tload(a[0:M, 0:K], a_mat)
    pto.tload(b[0:K, 0:N], b_mat)

    # Sync: MTE2 -> MTE1
    pto.record_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)
    pto.wait_event(pto.TLOAD, pto.TMOV_M2L, pto.EVENT_ID0)

    # Move MAT -> L0
    pto.tmov(a_mat, a_left)
    pto.tmov(b_mat, b_right)

    # Sync: MTE1 -> M
    pto.record_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)
    pto.wait_event(pto.TMOV_M2L, pto.TMATMUL, pto.EVENT_ID0)

    # Matmul
    pto.tmatmul(a_left, b_right, c_acc)

    # Sync: M -> MTE3
    pto.record_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)
    pto.wait_event(pto.TMATMUL, pto.TSTORE_ACC, pto.EVENT_ID0)

    # Store ACC -> GM
    pto.tstore(c[0:M, 0:N], c_acc)


if __name__ == "__main__":
    matmul_kernel()

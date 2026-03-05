"""Test: TiledTensor with dynamic (non-tiled) dimension using size= API.

Covers: tile(dim, size=N) where non-tiled dimensions keep their full
        dynamic extent (partition_tensor_view with dynamic dims).

NOTE: This test only verifies IR generation.  The generated IR contains
      dynamic partition_tensor_view dimensions which the ptoas parser
      does not currently support (pre-existing limitation).
"""
import pto_frontend as pto


@pto.kernel
def tiled_dynamic_dim(
    x: pto.Tensor(pto.float32, 2),
    z: pto.Tensor(pto.float32, 2),
):
    TILE_M = 32

    x_tiled = x.tile(dim=0, size=TILE_M)
    z_tiled = z.tile(dim=0, size=TILE_M)

    with x_tiled.for_each() as (i, x_view):
        z_view = z_tiled[i]
        # x_view and z_view have static_sizes = [32, -1] (dynamic dim 1)


if __name__ == "__main__":
    # Only verify IR generation succeeds; do NOT run ptoas on the output.
    ir = tiled_dynamic_dim.emit_ir()
    assert "scf.for" in ir
    assert "pto.partition_view" in ir
    print("IR generation OK")

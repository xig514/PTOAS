"""pto_frontend — high-level imperative Python frontend for PTO IR.

Usage::

    import pto_frontend as pto

    @pto.kernel
    def vector_add(x: pto.Tensor(pto.float16, 2), ...):
        tile = pto.make_tile((32, 32), pto.float16, pto.VEC, addr=0)
        pto.tload(x[0:32, 0:32], tile)
        ...

    vector_add()  # prints MLIR to stdout
"""

# -- data types --
from ._dtypes import DType, float16, float32, bfloat16, int8, int16, int32, int64

# -- tensor annotation --
from ._tensor import Tensor

# -- kernel decorator --
from ._kernel import kernel

# -- tile operations --
from ._ops import (
    make_tile,
    # DMA
    tload, tstore, tmov,
    tload_tile, tstore_tile,  # Coordinate-based load/store
    # binary
    tadd, tsub, tmul, tdiv, tand, tor, txor, tmax, tmin,
    # unary
    texp, tlog, tsqrt, trsqrt, trecip, tneg, tnot, trelu, tabs,
    # scalar
    tadds, tsubs, tmuls, tdivs, tmaxs, tmins,
    # reduction
    trowmax, trowmin, trowsum, tcolmax, tcolmin, tcolsum,
    # row-expand broadcast
    trowexpand, trowexpanddiv, trowexpandmul, trowexpandsub,
    # layout / transpose
    ttrans,
    # matmul
    tmatmul, tmatmul_acc, tmatmul_bias,
    # convert
    tcvt,
    # sync
    record_event, wait_event, barrier_sync,
    # system
    get_block_idx, get_block_num,
)

# -- control flow --
from ._control_flow import for_range, if_

# -- tiling utilities --
from ._layout import TileLayout as TileLayout_v1  # Old version
from ._layout_v2 import TensorLayout, TileLayout, TileCoordinate, TiledView
from ._tiled_tensor_nd import TiledTensorND, DistributedTiledTensorND

# -- split utilities --
from ._split_utils import split_even, split_causal, split_sequential

# -- address-space & sync constants --
from ._constants import (
    VEC, MAT, LEFT, RIGHT, ACC, GM, BIAS, SCALING,
    TLOAD, TSTORE_ACC, TSTORE_VEC,
    TMOV_M2L, TMOV_M2S, TMOV_M2B, TMOV_M2V, TMOV_V2M,
    TMATMUL, TVEC, TVECWAIT_EVENT,
    EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3,
    EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7,
)

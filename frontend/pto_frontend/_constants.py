"""Re-exported PTO dialect constants for address spaces, sync types, events."""

from mlir.dialects.pto import (
    AddressSpace,
    SyncOpType,
    EVENT,
    PIPE,
    BLayout,
    SLayout,
    Layout,
    TLOAD,
    TSTORE_ACC,
    TSTORE_VEC,
    TMOV_M2L,
    TMOV_M2S,
    TMOV_M2B,
    TMOV_M2V,
    TMOV_V2M,
    TMATMUL,
    TVEC,
    TVECWAIT_EVENT,
    EVENT_ID0,
    EVENT_ID1,
    EVENT_ID2,
    EVENT_ID3,
    EVENT_ID4,
    EVENT_ID5,
    EVENT_ID6,
    EVENT_ID7,
)

# Address-space aliases
VEC = AddressSpace.VEC
MAT = AddressSpace.MAT
LEFT = AddressSpace.LEFT
RIGHT = AddressSpace.RIGHT
ACC = AddressSpace.ACC
GM = AddressSpace.GM
BIAS = AddressSpace.BIAS
SCALING = AddressSpace.SCALING

# Pipeline aliases
PIPE_MTE1 = PIPE.PIPE_MTE1
PIPE_MTE2 = PIPE.PIPE_MTE2
PIPE_MTE3 = PIPE.PIPE_MTE3
PIPE_V = PIPE.PIPE_V
PIPE_M = PIPE.PIPE_M
PIPE_FIX = PIPE.PIPE_FIX
PIPE_ALL = PIPE.PIPE_ALL

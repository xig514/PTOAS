#include "runtime/rt_ffts.h"
#include "kernel.cpp"
extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* q, int32_t q_dim0, int32_t q_dim1, uint8_t* k, int32_t k_dim0, int32_t k_dim1, uint8_t* v, int32_t v_dim0, int32_t v_dim1, uint8_t* o, int32_t o_dim0, int32_t o_dim1, uint8_t* qk_buf, int32_t qk_buf_dim0, int32_t qk_buf_dim1, uint8_t* p_buf, int32_t p_buf_dim0, int32_t p_buf_dim1, uint8_t* pv_buf, int32_t pv_buf_dim0, int32_t pv_buf_dim1)
{
    uint64_t ffts = 0;
    uint32_t fftsLen = 0;
    rtGetC2cCtrlAddr(&ffts, &fftsLen);
    multicore_fa_mb_kernel<<<blockDim, nullptr, stream>>>((half*)q, q_dim0, q_dim1, (half*)k, k_dim0, k_dim1, (half*)v, v_dim0, v_dim1, (half*)o, o_dim0, o_dim1, (float*)qk_buf, qk_buf_dim0, qk_buf_dim1, (half*)p_buf, p_buf_dim0, p_buf_dim1, (float*)pv_buf, pv_buf_dim0, pv_buf_dim1, (int64_t*)ffts);
}

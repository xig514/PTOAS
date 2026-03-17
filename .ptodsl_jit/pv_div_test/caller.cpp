#include "kernel.cpp"
extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* v, int32_t v_dim0, int32_t v_dim1, uint8_t* o, int32_t o_dim0, int32_t o_dim1, uint8_t* pv_buf, int32_t pv_buf_dim0, int32_t pv_buf_dim1)
{
    pv_div_test<<<blockDim, nullptr, stream>>>((half*)v, v_dim0, v_dim1, (half*)o, o_dim0, o_dim1, (float*)pv_buf, pv_buf_dim0, pv_buf_dim1);
}

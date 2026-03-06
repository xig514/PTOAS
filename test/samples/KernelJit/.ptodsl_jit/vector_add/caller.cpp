#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* x, int32_t x_dim0, int32_t x_dim1, uint8_t* y, int32_t y_dim0, int32_t y_dim1, uint8_t* out, int32_t out_dim0, int32_t out_dim1)
{
    vector_add<<<blockDim, nullptr, stream>>>((half*)x, x_dim0, x_dim1, (half*)y, y_dim0, y_dim1, (half*)out, out_dim0, out_dim1);
}
#endif

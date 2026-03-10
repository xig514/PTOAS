#include "kernel.cpp"
extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* a, int32_t a_dim0, int32_t a_dim1, uint8_t* b, int32_t b_dim0, int32_t b_dim1, uint8_t* c, int32_t c_dim0, int32_t c_dim1)
{
    matmul_kernel<<<blockDim, nullptr, stream>>>((half*)a, a_dim0, a_dim1, (half*)b, b_dim0, b_dim1, (half*)c, c_dim0, c_dim1);
}

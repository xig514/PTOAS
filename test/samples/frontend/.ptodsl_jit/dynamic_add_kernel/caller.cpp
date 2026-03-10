#include "kernel.cpp"
extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* x, int32_t x_dim0, int32_t x_dim1, uint8_t* y, int32_t y_dim0, int32_t y_dim1, uint8_t* z, int32_t z_dim0, int32_t z_dim1)
{
    dynamic_add_kernel<<<blockDim, nullptr, stream>>>((half*)x, x_dim0, x_dim1, (half*)y, y_dim0, y_dim1, (half*)z, z_dim0, z_dim1);
}

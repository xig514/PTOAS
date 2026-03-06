#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* src, int32_t src_dim0, int32_t src_dim1, uint8_t* out, int32_t out_dim0, int32_t out_dim1)
{
    launchTROWSUMCase2<<<blockDim, nullptr, stream>>>((float*)src, src_dim0, src_dim1, (float*)out, out_dim0, out_dim1);
}
#endif

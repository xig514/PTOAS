#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void pv_div_test(__gm__ half* v1, int32_t v2, int32_t v3, __gm__ half* v4, int32_t v5, int32_t v6, __gm__ float* v7, int32_t v8, int32_t v9) {
  RoundMode v10 = RoundMode::CAST_RINT;
  unsigned v11 = 64;
  unsigned v12 = 1;
  unsigned v13 = 0;
  int64_t v14 = 0;
  int64_t v15 = 16384;
  int32_t v16 = 1;
  int32_t v17 = 64;
  using T = float;

  #if defined(__DAV_CUBE__)
  #endif // __DAV_CUBE__


  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, 64, 64, SLayout::NoneBox, 512, PadValue::Null> v18;
  TASSIGN(v18, v14);
  Tile<TileType::Vec, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::NoneBox, 512, PadValue::Null> v19;
  TASSIGN(v19, v15);
  unsigned v20 = (unsigned) v9;
  unsigned v21 = v11 * v20;
  pto::Shape<1, 1, 1, 64, 64> v22 = pto::Shape<1, 1, 1, 64, 64>();
  pto::Stride<-1, -1, -1, -1, 1> v23 = pto::Stride<-1, -1, -1, -1, 1>(v21, v21, v21, v20);
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v24 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v7 + (v13 + v13 * (unsigned) v9 + v13 * (unsigned) v16), v22, v23);
  TLOAD(v18, v24);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(v19, v18, v10);
  unsigned v25 = (unsigned) v6;
  unsigned v26 = v11 * v25;
  pto::Shape<1, 1, 1, 64, 64> v27 = pto::Shape<1, 1, 1, 64, 64>();
  pto::Stride<-1, -1, -1, -1, 1> v28 = pto::Stride<-1, -1, -1, -1, 1>(v26, v26, v26, v25);
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v29 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v13 + v13 * (unsigned) v6 + v13 * (unsigned) v16), v27, v28);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v29, v19);
  #endif // __DAV_VEC__

  return;
}


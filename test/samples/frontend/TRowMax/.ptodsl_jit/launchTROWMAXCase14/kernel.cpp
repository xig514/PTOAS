#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void launchTROWMAXCase14(__gm__ float* v1, int32_t v2, int32_t v3, __gm__ float* v4, int32_t v5, int32_t v6) {
  unsigned v7 = 16;
  unsigned v8 = 121;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 1;
  int64_t v12 = 0;
  int64_t v13 = 38080;
  int64_t v14 = 76160;
  int32_t v15 = 121;
  int32_t v16 = 16;
  using T = float;
  unsigned v17 = (unsigned) v6;
  unsigned v18 = v8 * v17;
  pto::Shape<1, 1, 1, 121, 16> v19 = pto::Shape<1, 1, 1, 121, 16>();
  pto::Stride<-1, -1, -1, -1, 1> v20 = pto::Stride<-1, -1, -1, -1, 1>(v18, v18, v18, v17);
  GlobalTensor<float, pto::Shape<1, 1, 1, 121, 16>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v21 = GlobalTensor<float, pto::Shape<1, 1, 1, 121, 16>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v10 + v10 * (unsigned) v6 + v10 * (unsigned) v11), v19, v20);
  unsigned v22 = (unsigned) v3;
  unsigned v23 = v8 * v22;
  pto::Shape<1, 1, 1, 121, 1> v24 = pto::Shape<1, 1, 1, 121, 1>();
  pto::Stride<-1, -1, -1, -1, 1> v25 = pto::Stride<-1, -1, -1, -1, 1>(v23, v23, v23, v22);
  GlobalTensor<float, pto::Shape<1, 1, 1, 121, 1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v26 = GlobalTensor<float, pto::Shape<1, 1, 1, 121, 1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v10 + v10 * (unsigned) v3 + v10 * (unsigned) v11), v24, v25);
  Tile<TileType::Vec, float, 238, 40, BLayout::RowMajor, 121, 16, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v12);
  Tile<TileType::Vec, float, 238, 40, BLayout::RowMajor, 121, 16, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v13);
  Tile<TileType::Vec, float, 238, 16, BLayout::RowMajor, 121, 1, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v14);
  TLOAD(v27, v21);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TROWMAX(v29, v27, v28);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v26, v29);
  return;
}


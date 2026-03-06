#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void launchTROWSUMCase7(__gm__ float* v1, int32_t v2, int32_t v3, __gm__ float* v4, int32_t v5, int32_t v6) {
  unsigned v7 = 256;
  unsigned v8 = 32;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 1;
  int64_t v12 = 0;
  int64_t v13 = 32768;
  int64_t v14 = 65536;
  using T = float;
  unsigned v15 = (unsigned) v3;
  unsigned v16 = v8 * v15;
  pto::Shape<1, 1, 1, 32, 256> v17 = pto::Shape<1, 1, 1, 32, 256>();
  pto::Stride<-1, -1, -1, -1, 1> v18 = pto::Stride<-1, -1, -1, -1, 1>(v16, v16, v16, v15);
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v19 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v10 + v10 * (unsigned) v3 + v10 * (unsigned) v11), v17, v18);
  unsigned v20 = (unsigned) v6;
  unsigned v21 = v8 * v20;
  pto::Shape<1, 1, 1, 32, 1> v22 = pto::Shape<1, 1, 1, 32, 1>();
  pto::Stride<-1, -1, -1, -1, 1> v23 = pto::Stride<-1, -1, -1, -1, 1>(v21, v21, v21, v20);
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v24 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v10 + v10 * (unsigned) v6 + v10 * (unsigned) v11), v22, v23);
  Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v25 = Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v2, v3);
  TASSIGN(v25, v12);
  Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v26 = Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v2, v3);
  TASSIGN(v26, v13);
  Tile<TileType::Vec, float, 32, 16, BLayout::RowMajor, -1, 1, SLayout::NoneBox, 512, PadValue::Null> v27 = Tile<TileType::Vec, float, 32, 16, BLayout::RowMajor, -1, 1, SLayout::NoneBox, 512, PadValue::Null>(v2);
  TASSIGN(v27, v14);
  TLOAD(v25, v19);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TROWSUM(v27, v25, v26);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v24, v27);
  return;
}


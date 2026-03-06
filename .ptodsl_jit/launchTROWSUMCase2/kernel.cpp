#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void launchTROWSUMCase2(__gm__ float* v1, int32_t v2, int32_t v3, __gm__ float* v4, int32_t v5, int32_t v6) {
  unsigned v7 = 64;
  unsigned v8 = 1;
  unsigned v9 = 0;
  int32_t v10 = 1;
  int64_t v11 = 0;
  int64_t v12 = 16384;
  int64_t v13 = 32768;
  using T = float;
  unsigned v14 = (unsigned) v3;
  unsigned v15 = v7 * v14;
  pto::Shape<1, 1, 1, 64, 64> v16 = pto::Shape<1, 1, 1, 64, 64>();
  pto::Stride<-1, -1, -1, -1, 1> v17 = pto::Stride<-1, -1, -1, -1, 1>(v15, v15, v15, v14);
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v18 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v9 + v9 * (unsigned) v3 + v9 * (unsigned) v10), v16, v17);
  unsigned v19 = (unsigned) v6;
  unsigned v20 = v7 * v19;
  pto::Shape<1, 1, 1, 64, 1> v21 = pto::Shape<1, 1, 1, 64, 1>();
  pto::Stride<-1, -1, -1, -1, 1> v22 = pto::Stride<-1, -1, -1, -1, 1>(v20, v20, v20, v19);
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v23 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v9 + v9 * (unsigned) v6 + v9 * (unsigned) v10), v21, v22);
  Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v24 = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v2, v3);
  TASSIGN(v24, v11);
  Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v25 = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v2, v3);
  TASSIGN(v25, v12);
  Tile<TileType::Vec, float, 64, 16, BLayout::RowMajor, -1, 1, SLayout::NoneBox, 512, PadValue::Null> v26 = Tile<TileType::Vec, float, 64, 16, BLayout::RowMajor, -1, 1, SLayout::NoneBox, 512, PadValue::Null>(v2);
  TASSIGN(v26, v13);
  TLOAD(v24, v18);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TROWSUM(v26, v24, v25);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v23, v26);
  return;
}


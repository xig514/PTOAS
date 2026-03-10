#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dynamic_add_kernel(__gm__ half* v1, int32_t v2, int32_t v3, __gm__ half* v4, int32_t v5, int32_t v6, __gm__ half* v7, int32_t v8, int32_t v9) {
  unsigned v10 = 128;
  unsigned v11 = 64;
  unsigned v12 = 1;
  unsigned v13 = 0;
  int64_t v14 = 0;
  int64_t v15 = 16384;
  int64_t v16 = 32768;
  int32_t v17 = 63;
  int32_t v18 = 64;
  int32_t v19 = 127;
  int32_t v20 = 128;
  int32_t v21 = 0;
  int32_t v22 = 1;
  using T = float;
  size_t v23 = (size_t) v22;
  size_t v24 = (size_t) v21;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v14);
  Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v15);
  Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v16);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  for (size_t v28 = v24; v28 < ((size_t) ((int32_t) ((uint32_t) v2 + (uint32_t) v17) / v18)); v28 += v23) {
    for (size_t v29 = v24; v29 < ((size_t) ((int32_t) ((uint32_t) v3 + (uint32_t) v19) / v20)); v29 += v23) {
      int32_t v30 = (int32_t) ((uint32_t) ((int32_t) v28) * (uint32_t) v18);
      int32_t v31 = (int32_t) ((uint32_t) ((int32_t) v29) * (uint32_t) v20);
      unsigned v32 = (unsigned) v3;
      unsigned v33 = v11 * v32;
      pto::Shape<1, 1, 1, 64, 128> v34 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v35 = pto::Stride<-1, -1, -1, -1, 1>(v33, v33, v33, v32);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v36 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v30 * (unsigned) v3 + (unsigned) v31 * (unsigned) v22), v34, v35);
      unsigned v37 = (unsigned) v6;
      unsigned v38 = v11 * v37;
      pto::Shape<1, 1, 1, 64, 128> v39 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v40 = pto::Stride<-1, -1, -1, -1, 1>(v38, v38, v38, v37);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v41 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v13 + (unsigned) v30 * (unsigned) v6 + (unsigned) v31 * (unsigned) v22), v39, v40);
      unsigned v42 = (unsigned) v9;
      unsigned v43 = v11 * v42;
      pto::Shape<1, 1, 1, 64, 128> v44 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v45 = pto::Stride<-1, -1, -1, -1, 1>(v43, v43, v43, v42);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v46 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v7 + (v13 + (unsigned) v30 * (unsigned) v9 + (unsigned) v31 * (unsigned) v22), v44, v45);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      TLOAD(v25, v36);
      TLOAD(v26, v41);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TADD(v27, v25, v26);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TSTORE(v46, v27);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    };
  }
  #endif // __DAV_VEC__

  return;
}


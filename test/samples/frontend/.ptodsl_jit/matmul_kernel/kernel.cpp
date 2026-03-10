#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void matmul_kernel(__gm__ half* v1, int32_t v2, int32_t v3, __gm__ half* v4, int32_t v5, int32_t v6, __gm__ half* v7, int32_t v8, int32_t v9) {
  unsigned v10 = 64;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int64_t v13 = 0;
  int64_t v14 = 8192;
  int32_t v15 = 63;
  int32_t v16 = 64;
  int32_t v17 = 0;
  int32_t v18 = 1;
  using T = float;
  size_t v19 = (size_t) v18;
  size_t v20 = (size_t) v17;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v21;
  TASSIGN(v21, v13);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v14);
  Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v13);
  Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v24;
  TASSIGN(v24, v13);
  Tile<TileType::Acc, float, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 1024, PadValue::Null> v25;
  TASSIGN(v25, v13);
  for (size_t v26 = v20; v26 < ((size_t) ((int32_t) ((uint32_t) v2 + (uint32_t) v15) / v16)); v26 += v19) {
    for (size_t v27 = v20; v27 < ((size_t) ((int32_t) ((uint32_t) v6 + (uint32_t) v15) / v16)); v27 += v19) {
      int32_t v28 = (int32_t) ((uint32_t) ((int32_t) v26) * (uint32_t) v16);
      int32_t v29 = (int32_t) ((uint32_t) ((int32_t) v27) * (uint32_t) v16);
      unsigned v30 = (unsigned) v3;
      unsigned v31 = v10 * v30;
      pto::Shape<1, 1, 1, 64, 64> v32 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v33 = pto::Stride<-1, -1, -1, -1, 1>(v31, v31, v31, v30);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v34 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v28 * (unsigned) v3 + v12 * (unsigned) v18), v32, v33);
      unsigned v35 = (unsigned) v6;
      unsigned v36 = v10 * v35;
      pto::Shape<1, 1, 1, 64, 64> v37 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v38 = pto::Stride<-1, -1, -1, -1, 1>(v36, v36, v36, v35);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v39 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v12 + v12 * (unsigned) v6 + (unsigned) v29 * (unsigned) v18), v37, v38);
      TLOAD(v21, v34);
      TLOAD(v22, v39);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v23, v21);
      TMOV(v24, v22);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      TMATMUL(v25, v23, v24);
      set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
      for (size_t v40 = v19; v40 < ((size_t) ((int32_t) ((uint32_t) v3 + (uint32_t) v15) / v16)); v40 += v19) {
        int32_t v41 = (int32_t) ((uint32_t) ((int32_t) v40) * (uint32_t) v16);
        unsigned v42 = (unsigned) v3;
        unsigned v43 = v10 * v42;
        pto::Shape<1, 1, 1, 64, 64> v44 = pto::Shape<1, 1, 1, 64, 64>();
        pto::Stride<-1, -1, -1, -1, 1> v45 = pto::Stride<-1, -1, -1, -1, 1>(v43, v43, v43, v42);
        GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v46 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v28 * (unsigned) v3 + (unsigned) v41 * (unsigned) v18), v44, v45);
        unsigned v47 = (unsigned) v6;
        unsigned v48 = v10 * v47;
        pto::Shape<1, 1, 1, 64, 64> v49 = pto::Shape<1, 1, 1, 64, 64>();
        pto::Stride<-1, -1, -1, -1, 1> v50 = pto::Stride<-1, -1, -1, -1, 1>(v48, v48, v48, v47);
        GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v51 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v12 + (unsigned) v41 * (unsigned) v6 + (unsigned) v29 * (unsigned) v18), v49, v50);
        TLOAD(v21, v46);
        TLOAD(v22, v51);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TMOV(v23, v21);
        TMOV(v24, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v25, v25, v23, v24);
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
      };
      unsigned v52 = (unsigned) v9;
      unsigned v53 = v10 * v52;
      pto::Shape<1, 1, 1, 64, 64> v54 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v55 = pto::Stride<-1, -1, -1, -1, 1>(v53, v53, v53, v52);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v56 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v7 + (v12 + (unsigned) v28 * (unsigned) v9 + (unsigned) v29 * (unsigned) v18), v54, v55);
      set_flag(PIPE_M, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE3, EVENT_ID0);
      TSTORE(v56, v25);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
  }
  #endif // __DAV_CUBE__

  return;
}


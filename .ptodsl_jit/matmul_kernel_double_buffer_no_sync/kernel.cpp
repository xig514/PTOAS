#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void matmul_kernel_double_buffer_no_sync(__gm__ half* v1, int32_t v2, int32_t v3, __gm__ half* v4, int32_t v5, int32_t v6, __gm__ half* v7, int32_t v8, int32_t v9) {
  unsigned v10 = 64;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int64_t v13 = 0;
  int64_t v14 = 8192;
  int64_t v15 = 16384;
  int64_t v16 = 24576;
  int32_t v17 = 63;
  int32_t v18 = 64;
  int32_t v19 = 0;
  int32_t v20 = 1;
  int32_t v21 = 2;
  using T = float;
  size_t v22 = (size_t) v20;
  size_t v23 = (size_t) v19;

  #if defined(__DAV_CUBE__)
  int32_t v24 = (int32_t) ((uint32_t) v6 + (uint32_t) v17) / v18;
  for (size_t v25 = v23; v25 < ((size_t) ((int32_t) ((uint32_t) v2 + (uint32_t) v17) / v18)); v25 += v22) {
    int32_t v26 = (int32_t) v25;
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    for (size_t v27 = v23; v27 < ((size_t) v24); v27 += v22) {
      int32_t v28 = (int32_t) v27;
      int32_t v29 = (int32_t) ((uint32_t) v26 * (uint32_t) v18);
      int32_t v30 = (int32_t) ((uint32_t) v28 * (uint32_t) v18);
      int32_t v31 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v26 * (uint32_t) v24) + (uint32_t) v28) % v21;
      bool v32 = v31 == v19;
      if (v32) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      } else {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      };
      if (v32) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      };
      if (v32) {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      } else {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
      };
      unsigned v33 = (unsigned) v3;
      unsigned v34 = v10 * v33;
      pto::Shape<1, 1, 1, 64, 64> v35 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v36 = pto::Stride<-1, -1, -1, -1, 1>(v34, v34, v34, v33);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v37 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v29 * (unsigned) v3 + v12 * (unsigned) v20), v35, v36);
      unsigned v38 = (unsigned) v6;
      unsigned v39 = v10 * v38;
      pto::Shape<1, 1, 1, 64, 64> v40 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v41 = pto::Stride<-1, -1, -1, -1, 1>(v39, v39, v39, v38);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v42 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v12 + v12 * (unsigned) v6 + (unsigned) v30 * (unsigned) v20), v40, v41);
      int64_t v43 = v32 ? v13 : v14;
      Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v44;
      TASSIGN(v44, v43);
      TLOAD(v44, v37);
      int64_t v45 = v32 ? v15 : v16;
      Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v46;
      TASSIGN(v46, v45);
      TLOAD(v46, v42);
      Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v47;
      TASSIGN(v47, v43);
      Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v48;
      TASSIGN(v48, v43);
      if (v32) {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      };
      if (v32) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      };
      TMOV(v47, v48);
      Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v49;
      TASSIGN(v49, v43);
      Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v50;
      TASSIGN(v50, v45);
      if (v32) {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      };
      if (v32) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      };
      TMOV(v49, v50);
      int64_t v51 = v32 ? v13 : v15;
      Tile<TileType::Acc, float, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 1024, PadValue::Null> v52;
      TASSIGN(v52, v51);
      Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v53;
      TASSIGN(v53, v43);
      Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v54;
      TASSIGN(v54, v43);
      if (v32) {
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      };
      if (v32) {
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      };
      TMATMUL(v52, v53, v54);
      for (size_t v55 = v22; v55 < ((size_t) ((int32_t) ((uint32_t) v3 + (uint32_t) v17) / v18)); v55 += v22) {
        bool v56 = (int32_t) ((uint32_t) v31 + (uint32_t) v20) % v21 == v19;
        if (v56) {
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        } else {
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        };
        if (v56) {
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        } else {
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        };
        int32_t v57 = (int32_t) ((uint32_t) ((int32_t) v55) * (uint32_t) v18);
        unsigned v58 = (unsigned) v3;
        unsigned v59 = v10 * v58;
        pto::Shape<1, 1, 1, 64, 64> v60 = pto::Shape<1, 1, 1, 64, 64>();
        pto::Stride<-1, -1, -1, -1, 1> v61 = pto::Stride<-1, -1, -1, -1, 1>(v59, v59, v59, v58);
        GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v62 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v29 * (unsigned) v3 + (unsigned) v57 * (unsigned) v20), v60, v61);
        unsigned v63 = (unsigned) v6;
        unsigned v64 = v10 * v63;
        pto::Shape<1, 1, 1, 64, 64> v65 = pto::Shape<1, 1, 1, 64, 64>();
        pto::Stride<-1, -1, -1, -1, 1> v66 = pto::Stride<-1, -1, -1, -1, 1>(v64, v64, v64, v63);
        GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v67 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v12 + (unsigned) v57 * (unsigned) v6 + (unsigned) v30 * (unsigned) v20), v65, v66);
        int64_t v68 = v56 ? v13 : v14;
        Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v69;
        TASSIGN(v69, v68);
        TLOAD(v69, v62);
        int64_t v70 = v56 ? v15 : v16;
        Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v71;
        TASSIGN(v71, v70);
        TLOAD(v71, v67);
        Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v72;
        TASSIGN(v72, v68);
        Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v73;
        TASSIGN(v73, v68);
        if (v56) {
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        } else {
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        };
        if (v56) {
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        } else {
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        };
        TMOV(v72, v73);
        Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v74;
        TASSIGN(v74, v68);
        Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v75;
        TASSIGN(v75, v70);
        if (v56) {
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        } else {
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        };
        if (v56) {
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        } else {
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        };
        TMOV(v74, v75);
        Tile<TileType::Acc, float, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 1024, PadValue::Null> v76;
        TASSIGN(v76, v51);
        Tile<TileType::Acc, float, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 1024, PadValue::Null> v77;
        TASSIGN(v77, v51);
        Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v78;
        TASSIGN(v78, v68);
        Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v79;
        TASSIGN(v79, v68);
        if (v56) {
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        } else {
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        };
        if (v56) {
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        } else {
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        };
        TMATMUL_ACC(v76, v77, v78, v79);
        if (v56) {
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        } else {
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        };
        if (v56) {
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        } else {
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        };
      };
      unsigned v80 = (unsigned) v9;
      unsigned v81 = v10 * v80;
      pto::Shape<1, 1, 1, 64, 64> v82 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v83 = pto::Stride<-1, -1, -1, -1, 1>(v81, v81, v81, v80);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v84 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v7 + (v12 + (unsigned) v29 * (unsigned) v9 + (unsigned) v30 * (unsigned) v20), v82, v83);
      Tile<TileType::Acc, float, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 1024, PadValue::Null> v85;
      TASSIGN(v85, v51);
      if (v32) {
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      } else {
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
      };
      if (v32) {
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      } else {
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
      };
      TSTORE(v84, v85);
      if (v32) {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      } else {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
      };
      if (v32) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      } else {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      };
      if (v32) {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      };
    };
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  }
  #endif // __DAV_CUBE__

  return;
}


#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void multicore_fa_mb_kernel(__gm__ half* v1, int32_t v2, int32_t v3, __gm__ half* v4, int32_t v5, int32_t v6, __gm__ half* v7, int32_t v8, int32_t v9, __gm__ half* v10, int32_t v11, int32_t v12, __gm__ float* v13, int32_t v14, int32_t v15, __gm__ half* v16, int32_t v17, int32_t v18, __gm__ float* v19, int32_t v20, int32_t v21, __gm__ int64_t* v22) {
  RoundMode v23 = RoundMode::CAST_RINT;
  unsigned v24 = 32;
  unsigned v25 = 64;
  unsigned v26 = 1;
  unsigned v27 = 0;
  int32_t v28 = 63;
  int32_t v29 = 64;
  int64_t v30 = 0;
  int64_t v31 = 8192;
  int64_t v32 = 16384;
  int64_t v33 = 24576;
  int64_t v34 = 32768;
  int64_t v35 = 40960;
  int64_t v36 = 49152;
  int32_t v37 = 0;
  int32_t v38 = 1;
  int32_t v39 = 2;
  int64_t v40 = 28672;
  int64_t v41 = 57344;
  int64_t v42 = 65536;
  int32_t v43 = 32;
  float v44 = 0.125f;
  float v45 = 1.0f;
  using T = float;
  size_t v46 = (size_t) v38;
  uint64_t v47 = (uint64_t) v22;
  set_ffts_base_addr(v47);
  size_t v48 = (size_t) ((int32_t) ((uint32_t) v2 + (uint32_t) v28) / v29);
  size_t v49 = (size_t) ((int32_t) ((uint32_t) v5 + (uint32_t) v28) / v29);
  int64_t v50 = get_block_num();
  size_t v51 = (size_t) ((int32_t) (int64_t) v50);

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v52;
  TASSIGN(v52, v30);
  Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v53;
  TASSIGN(v53, v30);
  int64_t v54 = get_block_idx();
  int32_t v55 = (int32_t) ((int64_t) v54);
  int32_t v56 = (int32_t) ((uint32_t) v55 * (uint32_t) v29);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  for (size_t v57 = (size_t) v55; v57 < v48; v57 += v51) {
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    unsigned v58 = (unsigned) v3;
    unsigned v59 = v25 * v58;
    pto::Shape<1, 1, 1, 64, 64> v60 = pto::Shape<1, 1, 1, 64, 64>();
    pto::Stride<-1, -1, -1, -1, 1> v61 = pto::Stride<-1, -1, -1, -1, 1>(v59, v59, v59, v58);
    GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v62 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v27 + (unsigned) ((int32_t) (uint32_t) ((int32_t) v57) * (uint32_t) v29) * (unsigned) v3 + v27 * (unsigned) v38), v60, v61);
    TLOAD(v52, v62);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v53, v52);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    for (size_t v63 = (size_t) v37; v63 < v49; v63 += v46) {
      int32_t v64 = (int32_t) v63;
      int32_t v65 = (int32_t) ((uint32_t) v64 * (uint32_t) v29);
      bool v66 = v64 % v39 == v37;
      if (v66) {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      } else {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
      };
      if (v66) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      } else {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      };
      if (v66) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      };
      int64_t v67 = v66 ? v31 : v32;
      Tile<TileType::Mat, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v68;
      TASSIGN(v68, v67);
      int64_t v69 = v66 ? v30 : v31;
      Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v70;
      TASSIGN(v70, v69);
      int64_t v71 = v66 ? v30 : v32;
      Tile<TileType::Acc, float, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 1024, PadValue::Null> v72;
      TASSIGN(v72, v71);
      int64_t v73 = v66 ? v33 : v34;
      Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v74;
      TASSIGN(v74, v73);
      Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v75;
      TASSIGN(v75, v67);
      int64_t v76 = v66 ? v35 : v36;
      Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v77;
      TASSIGN(v77, v76);
      int64_t v78 = v66 ? v32 : v33;
      Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v79;
      TASSIGN(v79, v78);
      int64_t v80 = v66 ? v34 : v36;
      Tile<TileType::Acc, float, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 1024, PadValue::Null> v81;
      TASSIGN(v81, v80);
      pto::Shape<1, 1, 1, 64, 64> v82 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<64, 64, 64, 1, -1> v83 = pto::Stride<64, 64, 64, 1, -1>((unsigned) v6);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<64, 64, 64, 1, -1>, pto::Layout::DN> v84 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<64, 64, 64, 1, -1>, pto::Layout::DN>(v4 + (v27 + v27 * (unsigned) v38 + (unsigned) v65 * (unsigned) v6), v82, v83);
      TLOAD(v68, v84);
      if (v66) {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      };
      if (v66) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      };
      TMOV(v70, v68);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      TMATMUL(v72, v53, v70);
      unsigned v85 = (unsigned) v15;
      unsigned v86 = v25 * v85;
      pto::Shape<1, 1, 1, 64, 64> v87 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v88 = pto::Stride<-1, -1, -1, -1, 1>(v86, v86, v86, v85);
      GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v89 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v13 + (v27 + (unsigned) v56 * (unsigned) v15 + (unsigned) v65 * (unsigned) v38), v87, v88);
      if (v66) {
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      } else {
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
      };
      if (v66) {
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      } else {
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
      };
      TSTORE(v89, v72);
      uint16_t v90 = getFFTSMsg(FFTS_MODE_VAL, v37);
      ffts_cross_core_sync(PIPE_FIX, v90);
      wait_flag_dev(1);
      unsigned v91 = (unsigned) v18;
      unsigned v92 = v25 * v91;
      pto::Shape<1, 1, 1, 64, 64> v93 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v94 = pto::Stride<-1, -1, -1, -1, 1>(v92, v92, v92, v91);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v95 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v16 + (v27 + (unsigned) v56 * (unsigned) v18 + (unsigned) v65 * (unsigned) v38), v93, v94);
      TLOAD(v74, v95);
      if (v66) {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      };
      if (v66) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      };
      TMOV(v75, v74);
      unsigned v96 = (unsigned) v9;
      unsigned v97 = v25 * v96;
      pto::Shape<1, 1, 1, 64, 64> v98 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v99 = pto::Stride<-1, -1, -1, -1, 1>(v97, v97, v97, v96);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v100 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v7 + (v27 + (unsigned) v65 * (unsigned) v9 + v27 * (unsigned) v38), v98, v99);
      TLOAD(v77, v100);
      if (v66) {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      };
      if (v66) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      };
      TMOV(v79, v77);
      if (v66) {
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      };
      if (v66) {
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      };
      TMATMUL(v81, v75, v79);
      unsigned v101 = (unsigned) v21;
      unsigned v102 = v25 * v101;
      pto::Shape<1, 1, 1, 64, 64> v103 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v104 = pto::Stride<-1, -1, -1, -1, 1>(v102, v102, v102, v101);
      GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v105 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v19 + (v27 + (unsigned) v56 * (unsigned) v21 + v27 * (unsigned) v38), v103, v104);
      if (v66) {
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      } else {
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
      };
      if (v66) {
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      } else {
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
      };
      TSTORE(v105, v81);
      uint16_t v106 = getFFTSMsg(FFTS_MODE_VAL, v39);
      ffts_cross_core_sync(PIPE_FIX, v106);
      if (v66) {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      } else {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
      };
      if (v66) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      } else {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      };
      if (v66) {
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
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_CUBE__


  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, 32, 64, SLayout::NoneBox, 512, PadValue::Null> v107;
  TASSIGN(v107, v30);
  Tile<TileType::Vec, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::NoneBox, 512, PadValue::Null> v108;
  TASSIGN(v108, v33);
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, 32, 64, SLayout::NoneBox, 512, PadValue::Null> v109;
  TASSIGN(v109, v32);
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, 32, 1, SLayout::NoneBox, 512, PadValue::Null> v110;
  TASSIGN(v110, v34);
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, 32, 64, SLayout::NoneBox, 512, PadValue::Null> v111;
  TASSIGN(v111, v35);
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, 32, 64, SLayout::NoneBox, 512, PadValue::Null> v112;
  TASSIGN(v112, v36);
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, 32, 64, SLayout::NoneBox, 512, PadValue::Null> v113;
  TASSIGN(v113, v41);
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, 32, 64, SLayout::NoneBox, 512, PadValue::Null> v114;
  TASSIGN(v114, v42);
  int64_t v115 = get_block_idx();
  int32_t v116 = (int32_t) ((int64_t) v115);
  int64_t v117 = get_subblockid();
  int32_t v118 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v117) * (uint32_t) v43);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (size_t v119 = (size_t) v116; v119 < v48; v119 += v51) {
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag_dev(0);
    int32_t v120 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v116 * (uint32_t) v29) + (uint32_t) v118);
    unsigned v121 = (unsigned) v15;
    unsigned v122 = v24 * v121;
    pto::Shape<1, 1, 1, 32, 64> v123 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<-1, -1, -1, -1, 1> v124 = pto::Stride<-1, -1, -1, -1, 1>(v122, v122, v122, v121);
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v125 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v13 + (v27 + (unsigned) v120 * (unsigned) v15 + v27 * (unsigned) v38), v123, v124);
    TLOAD(v107, v125);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWMAX(v110, v107, v109);
    TROWEXPAND(v111, v110);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSUB(v109, v107, v111);
    TMULS(v109, v109, v44);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TEXP(v107, v109);
    TROWSUM(v110, v107, v109);
    TROWEXPAND(v112, v110);
    TCVT(v108, v107, v23);
    unsigned v126 = (unsigned) v18;
    unsigned v127 = v24 * v126;
    pto::Shape<1, 1, 1, 32, 64> v128 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<-1, -1, -1, -1, 1> v129 = pto::Stride<-1, -1, -1, -1, 1>(v127, v127, v127, v126);
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v130 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v16 + (v27 + (unsigned) v120 * (unsigned) v18 + v27 * (unsigned) v38), v128, v129);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v130, v108);
    uint16_t v131 = getFFTSMsg(FFTS_MODE_VAL, v38);
    ffts_cross_core_sync(PIPE_MTE3, v131);
    wait_flag_dev(2);
    unsigned v132 = (unsigned) v21;
    unsigned v133 = v24 * v132;
    pto::Shape<1, 1, 1, 32, 64> v134 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<-1, -1, -1, -1, 1> v135 = pto::Stride<-1, -1, -1, -1, 1>(v133, v133, v133, v132);
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v136 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v19 + (v27 + (unsigned) v120 * (unsigned) v21 + v27 * (unsigned) v38), v134, v135);
    TLOAD(v113, v136);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (size_t v137 = v46; v137 < v49; v137 += v46) {
      int32_t v138 = (int32_t) v137;
      int32_t v139 = (int32_t) ((uint32_t) v138 * (uint32_t) v29);
      bool v140 = v138 % v39 == v37;
      if (v140) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      };
      if (v140) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      } else {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      };
      int64_t v141 = v140 ? v30 : v31;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, 32, 64, SLayout::NoneBox, 512, PadValue::Null> v142;
      TASSIGN(v142, v141);
      int64_t v143 = v140 ? v33 : v40;
      Tile<TileType::Vec, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::NoneBox, 512, PadValue::Null> v144;
      TASSIGN(v144, v143);
      wait_flag_dev(0);
      unsigned v145 = (unsigned) v15;
      unsigned v146 = v24 * v145;
      pto::Shape<1, 1, 1, 32, 64> v147 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v148 = pto::Stride<-1, -1, -1, -1, 1>(v146, v146, v146, v145);
      GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v149 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v13 + (v27 + (unsigned) v120 * (unsigned) v15 + (unsigned) v139 * (unsigned) v38), v147, v148);
      TLOAD(v142, v149);
      if (v140) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      };
      if (v140) {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      };
      TROWMAX(v110, v142, v109);
      TROWEXPAND(v109, v110);
      TMAX(v109, v109, v111);
      TSUB(v114, v111, v109);
      TMULS(v114, v114, v44);
      TEXP(v114, v114);
      TMUL(v112, v112, v114);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TMUL(v113, v113, v114);
      TMULS(v111, v109, v45);
      if (v140) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      };
      if (v140) {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      };
      TSUB(v109, v142, v111);
      TMULS(v109, v109, v44);
      if (v140) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      };
      if (v140) {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      };
      TEXP(v142, v109);
      TROWSUM(v110, v142, v109);
      TROWEXPAND(v109, v110);
      TADD(v112, v112, v109);
      TCVT(v144, v142, v23);
      unsigned v150 = (unsigned) v18;
      unsigned v151 = v24 * v150;
      pto::Shape<1, 1, 1, 32, 64> v152 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<-1, -1, -1, -1, 1> v153 = pto::Stride<-1, -1, -1, -1, 1>(v151, v151, v151, v150);
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v154 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v16 + (v27 + (unsigned) v120 * (unsigned) v18 + (unsigned) v139 * (unsigned) v38), v152, v153);
      if (v140) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      } else {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      };
      if (v140) {
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      } else {
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      };
      TSTORE(v154, v144);
      uint16_t v155 = getFFTSMsg(FFTS_MODE_VAL, v38);
      ffts_cross_core_sync(PIPE_MTE3, v155);
      wait_flag_dev(2);
      if (v140) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      } else {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      };
      if (v140) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      } else {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      };
      TLOAD(v142, v136);
      if (v140) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      };
      if (v140) {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      };
      TADD(v113, v113, v142);
      if (v140) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      };
      if (v140) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      } else {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      };
    };
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TDIV(v113, v113, v112);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TCVT(v108, v113, v23);
    unsigned v156 = (unsigned) v12;
    unsigned v157 = v24 * v156;
    pto::Shape<1, 1, 1, 32, 64> v158 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<-1, -1, -1, -1, 1> v159 = pto::Stride<-1, -1, -1, -1, 1>(v157, v157, v157, v156);
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v160 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v10 + (v27 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) v119) * (uint32_t) v29) + (uint32_t) v118) * (unsigned) v12 + v27 * (unsigned) v38), v158, v159);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v160, v108);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}


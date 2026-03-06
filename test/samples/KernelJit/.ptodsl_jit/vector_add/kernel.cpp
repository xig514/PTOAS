#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void vector_add(__gm__ half* v1, int32_t v2, int32_t v3, __gm__ half* v4, int32_t v5, int32_t v6, __gm__ half* v7, int32_t v8, int32_t v9) {
  unsigned v10 = 32;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int64_t v13 = 32;
  int32_t v14 = 1;
  int64_t v15 = 0;
  int32_t v16 = 32;
  using T = float;
  int64_t v17 = get_block_idx();
  int32_t v18 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v17) * (uint64_t) v13);
  unsigned v19 = (unsigned) v3;
  unsigned v20 = v10 * v19;
  pto::Shape<1, 1, 1, 32, 32> v21 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<-1, -1, -1, -1, 1> v22 = pto::Stride<-1, -1, -1, -1, 1>(v20, v20, v20, v19);
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v23 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v18 * (unsigned) v3 + v12 * (unsigned) v14), v21, v22);
  unsigned v24 = (unsigned) v6;
  unsigned v25 = v10 * v24;
  pto::Shape<1, 1, 1, 32, 32> v26 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<-1, -1, -1, -1, 1> v27 = pto::Stride<-1, -1, -1, -1, 1>(v25, v25, v25, v24);
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v28 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v12 + (unsigned) v18 * (unsigned) v6 + v12 * (unsigned) v14), v26, v27);
  unsigned v29 = (unsigned) v9;
  unsigned v30 = v10 * v29;
  pto::Shape<1, 1, 1, 32, 32> v31 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<-1, -1, -1, -1, 1> v32 = pto::Stride<-1, -1, -1, -1, 1>(v30, v30, v30, v29);
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v33 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v7 + (v12 + (unsigned) v18 * (unsigned) v9 + v12 * (unsigned) v14), v31, v32);
  Tile<TileType::Vec, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v34;
  TASSIGN(v34, v15);
  Tile<TileType::Vec, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v35;
  TASSIGN(v35, v15);
  Tile<TileType::Vec, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v36;
  TASSIGN(v36, v15);
  TLOAD(v34, v23);
  TLOAD(v35, v28);
  TADD(v36, v34, v35);
  TSTORE(v33, v36);
  return;
}


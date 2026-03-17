# AI Assistant Rules for PTOAS

This directory contains project-specific rules for AI-assisted development.

## Scope

When you change any user-visible behavior, keep these layers synchronized:

1. **ODS / Dialect definitions**: `include/PTO/IR/*.td`
2. **C++ implementation & verifiers**: `lib/PTO/IR`, `lib/PTO/Transforms`
3. **CLI / tool behavior**: `tools/ptoas`
4. **Python bindings / samples** (if affected): `python/`, `test/samples`
5. **Docs**: `README.md`, `docs/`
6. **Tests**: `test/`

## Architecture
- `include/PTO/IR` - pto mlir
- `frontend` - the front end code of pto mlir
- `test/samples/TiledTensor/LAYOUT_V2_DESIGN.md` - design of the TileLayout

## Build Commands
### Full configure, compile，build, install
```bash
export HOME=/data/g00895580
source compile.sh
```

## Test Commands
### Run a frontend testcase
```bash
python3 test/samples/frontend/test_jit_launch.py
```
### mlir python path
/data/g00895580/mlir/llvm-project/build-shared/tools/mlir/python_packages/mlir_core

## Pipeline and sync
- TLOAD is PIPE_MTE2, TSTORE_ACC is PIPE_FIX, TMOV_M2L and TMOV_M2B are MTE1, TMOV_M2S and TMOV_V2M are PIPE_FIX, TMOV_M2V is PIPV, TMATMUL is PIPE_M, TVEC and TVECWAIT_EVENT is PIPE_V
- When a buffer is used within a loop, backward synchronization is needed: at the start of each iteration, execution must wait for all associated pipelines from the previous iteration to have completed before the buffer can be reused.
## Ascend Npu Harward Core information
- For AscendNPU, there are multiple cores, each processing a chunk of data. The MatMul operation is computed on Cube cores, while most other operations are computed on Vector cores. The ratio of Cube cores to Vector cores is 1:2.
- For Cube-Only or Vector-Only operation, use pto.get_block_idx() to get current Cube or vector core index, use pto.get_block_num() to get total living Cube number. Use pto.get_subblock_idx() to get current Vector sub block idx, which is 0 or 1. 
- For Mix(which contais both Cube and Vector) operation, use pto.get_block_idx() // 2 to get the corresponding Cube core index.

## Rules
- matmul kernel code need to begin with pto.section_cube():, and other vector kernel need to begin with pto.section_vector():.
- You can only edit folder frontend and test. Require for approve when you want to edit other source files.
- The frontend representation must offer strong ergonomics and high expressiveness.
- Temp files should be put in /data/g00895580/tmp
- Refer to "Pipeline and sync" to get information about how to add sync op.
- Every time after finished editting the code. Run the following testcase:
- pto.VEC is 192KB on a2 and a3, and 248KB on a5; pto.MAT is 512KB on a2/a3/a5; pto.LEFT and pto.RIGHT are 64KB; pto.ACC is 128KB on a2 and a3 and 256KB on a5.
```bash
export HOME=/data/g00895580
source compile.sh
python3 test/samples/frontend/test_jit_launch.py
```

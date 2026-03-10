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
### Run a ptoas testcase
```bash
ptoas test/samples/FlashAttention/flash_attention_softmax.pto
```

### Run a python ir and ptoas testcase
```bash
cd test/samples/AddPtr
python3 addptr.py >a.pto
ptoas a.pto --pto-level=level3
```

## Rules
- matmul的代码需要加with pto.section_cube():, 其他vector代码需要加with pto.section_vector():.
- cmake command is shown in "Full configure"
- Make sure the test case in "Run a python ir and ptoas testcase" can be executed without error.
- Your testcase should also use ptoas to compile the ir which is generated from your new frontend. 
- Create a new directory for all new code and donot contaminate original code. (If there is some bug in original code, it's ok to fix it). Supply compile and install scripts to install the dialect.
- The frontend representation must offer strong ergonomics and high expressiveness.
- ptoas command needs to add argument "--pto-level=level3"
- Temp files should be put in /data/g00895580/tmp
- 

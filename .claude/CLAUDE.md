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

## Rules

See `.claude/rules/` for specific guidance:

- `cross-layer-sync.md`
- `testing-and-examples.md`

## Build Commands
### Full configure
```bash
export WORKSPACE_DIR=/home/code/ptoas/
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install

export LLVM_SOURCE_DIR=/home/pkg/mlir/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared

cmake -G Ninja -S . -B build -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir -DPython3_EXECUTABLE=$(which python3) -DPython3_FIND_STRATEGY=LOCATION -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DMLIR_PYTHON_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"

ninja -C build -j4
ninja -C build install

export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH
export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH
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
ptoas a.pto
```

## Rules
- cmake command is shown in "Full configure"
- Make sure the test case in "Run a python ir and ptoas testcase" can be executed without error.
- Your testcase should also use ptoas to compile the ir which is generated from your new frontend. 
- Create a new directory for all new code and donot contaminate original code. (If there is some bug in original code, it's ok to fix it). Supply compile and install scripts to install the dialect.

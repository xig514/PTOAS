export python3=/data/g00895580/miniconda3/bin/python3.12
export WORKSPACE_DIR=/data/g00895580/ptoas/
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install

export LLVM_SOURCE_DIR=/data/g00895580/mlir/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)
export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH
export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH
cmake -G Ninja -S . -B build -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir -DPython3_EXECUTABLE=$(which python3) -DPython3_FIND_STRATEGY=LOCATION -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DMLIR_PYTHON_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"

ninja -C build -j128
ninja -C build install


name=g00895580
source /data/$name/Ascend/cann/bin/setenv.bash
source /data/$name/Ascend/cann-8.5.0/set_env.sh
source /data/$name/Ascend/ascend-toolkit/set_env.sh

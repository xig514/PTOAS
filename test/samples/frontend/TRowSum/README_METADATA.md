# TRowSum with @pto_meta_data

This example demonstrates the new `@pto_meta_data` decorator for clean separation of kernel logic and metadata configuration.

## Quick Start

```bash
# Set up environment
export LLVM_BUILD_DIR=/home/pkg/mlir/llvm-project/build-shared
export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
export PTO_INSTALL_DIR=/home/code/ptoas/PTOAS/install
export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH
export PATH=/home/code/ptoas/PTOAS/build/tools/ptoas:/usr/local/Ascend/cann-8.5.0/x86_64-linux/bin:$PATH
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest

# Run tests
python3 test_trowsum_dynamic_v2.py --ir-only      # IR generation only
python3 test_trowsum_dynamic_v2.py --cpp-only     # IR + C++ generation
python3 test_trowsum_dynamic_v2.py                # Full compilation
python3 test_trowsum_dynamic_v2.py --case 1 2 3   # Specific cases
```

## Key Features

### 1. Clean Kernel Signature

**Before** (old style):
```python
@pto.kernel(metadata=dict(phys_row=128, phys_col=64, dtype=pto.float32))
def my_kernel(src: pto.Tensor(...), out: pto.Tensor(...), *, phys_row, phys_col, dtype):
    # Metadata pollutes signature
```

**After** (new style):
```python
@pto_meta_data
def meta():
    return {"phys_row": 128, "phys_col": 64, "dtype": pto.float32}

@pto.kernel(metadata=meta)
def my_kernel(src: pto.Tensor(...), out: pto.Tensor(...)):
    # Clean signature - only tensor parameters
```

### 2. Static + Dynamic Metadata

```python
@pto_meta_data
def meta():
    return {
        "phys_row": 128,        # Static: compile-time constant
        "phys_col": 64,         # Static: compile-time constant
        "dtype": pto.float32,   # Static: compile-time constant
        "valid_row": "dynamic", # Dynamic: runtime from tensor.shape
        "valid_col": "dynamic", # Dynamic: runtime from tensor.shape
    }

@pto.kernel(metadata=meta)
def my_kernel(src: pto.Tensor(...), out: pto.Tensor(...)):
    # Extract static metadata
    static = meta.get_static_metadata()
    phys_row = static["phys_row"]

    # Extract dynamic metadata from tensor
    valid_row = src.shape[0]
    valid_col = src.shape[1]
```

### 3. Important: Static Partition Sizes

**ptoas currently requires static partition sizes**:

```python
# ✓ CORRECT: Use static physical sizes
src_part = src.partition(offsets=[0, 0], sizes=[phys_row, phys_col])

# ✗ WRONG: Dynamic sizes cause ptoas error
src_part = src.partition(offsets=[0, 0], sizes=[src.shape[0], src.shape[1]])
```

### 4. Dynamic Valid Shapes in Tiles

Tile valid shapes can be dynamic:

```python
tile = pto.alloc_tile(
    addr=0,
    physical_shape=(phys_row, phys_col),  # Static
    valid_row=valid_row.ssa,              # Dynamic
    valid_col=valid_col.ssa,              # Dynamic
    dtype=dtype,
)
```

## Files

- `test_trowsum_dynamic_v2.py` - New implementation with @pto_meta_data
- `test_trowsum_dynamic.py` - Old implementation (for comparison)
- `METADATA_DESIGN.md` - Detailed design documentation

## Test Cases

8 test cases covering different tile sizes and data types:

| Case | dtype    | phys_row | phys_col | dst_col |
|------|----------|----------|----------|---------|
| 1    | float32  | 128      | 64       | 1       |
| 2    | float32  | 64       | 64       | 1       |
| 3    | float32  | 32       | 128      | 1       |
| 4    | float32  | 16       | 192      | 1       |
| 5    | float32  | 8        | 448      | 1       |
| 6    | float16  | 256      | 16       | 1       |
| 7    | float32  | 32       | 256      | 1       |
| 8    | float32  | 64       | 128      | 1       |

## Implementation Details

See `METADATA_DESIGN.md` for:
- Architecture overview
- MetaDataFunction class implementation
- KernelFunction integration
- Usage patterns
- Constraints and limitations

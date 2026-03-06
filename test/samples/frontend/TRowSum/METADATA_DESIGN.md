# @pto_meta_data Design Document

## Overview

This document describes the design and implementation of the `@pto_meta_data` decorator for PTO frontend, which enables clean separation of kernel logic from metadata configuration.

## Motivation

**Before**: Kernel signatures were cluttered with metadata parameters:

```python
@pto.kernel(metadata=dict(phys_row=128, phys_col=64, dtype=pto.float32))
def my_kernel(
    src: pto.Tensor(pto.float32, 2),
    out: pto.Tensor(pto.float32, 2),
    *,
    phys_row, phys_col, dtype  # Metadata pollutes signature
):
    # Use metadata...
```

**After**: Clean kernel signatures with metadata in separate function:

```python
@pto_meta_data
def meta():
    return {
        "phys_row": 128,
        "phys_col": 64,
        "dtype": pto.float32,
        "valid_row": "dynamic",  # Runtime from tensor.shape
    }

@pto.kernel(metadata=meta)
def my_kernel(
    src: pto.Tensor(pto.float32, 2),
    out: pto.Tensor(pto.float32, 2),
):
    # Access metadata via meta.get_static_metadata()
    phys_row = meta.get_static_metadata()["phys_row"]
    valid_row = src.shape[0]  # Dynamic metadata
```

## Key Features

### 1. Static Metadata (Compile-Time Constants)

Static metadata values are known at kernel definition time:

```python
@pto_meta_data
def meta():
    return {
        "phys_row": 128,      # Physical tile row size
        "phys_col": 64,       # Physical tile column size
        "dtype": pto.float32, # Data type
        "dst_col": 1,         # Output column size
    }
```

These values are:
- Used for partition sizes (ptoas requires static partition sizes)
- Used for tile allocation physical dimensions
- Passed to kernel via `@pto.kernel(metadata=meta)`

### 2. Dynamic Metadata (Runtime Values)

Dynamic metadata is extracted from tensor shapes at runtime:

```python
@pto_meta_data
def meta():
    return {
        "phys_row": 128,
        "phys_col": 64,
        "valid_row": "dynamic",  # Mark as dynamic
        "valid_col": "dynamic",  # Mark as dynamic
    }

@pto.kernel(metadata=meta)
def my_kernel(src: pto.Tensor(...), out: pto.Tensor(...)):
    # Extract dynamic metadata from tensor.shape
    valid_row = src.shape[0]
    valid_col = src.shape[1]

    # Use in tile allocation
    tile = pto.alloc_tile(..., valid_row=valid_row.ssa, valid_col=valid_col.ssa)
```

### 3. Static vs Dynamic Partition Sizes

**Important**: ptoas currently requires **static partition sizes**:

```python
# ✓ CORRECT: Use static physical sizes for partition
src_part = src.partition(offsets=[0, 0], sizes=[phys_row, phys_col])

# ✗ WRONG: Dynamic sizes cause ptoas parse error
src_part = src.partition(offsets=[0, 0], sizes=[src.shape[0], src.shape[1]])
```

This generates:
- **Static**: `!pto.partition_tensor_view<128x64xf32>` ✓
- **Dynamic**: `!pto.partition_tensor_view<-1x-1xf32>` ✗ (ptoas error)

## Implementation

### MetaDataFunction Class

```python
class MetaDataFunction:
    def __init__(self, fn):
        self._fn = fn
        self._config = None

    def get_config(self):
        """Get full metadata configuration (cached)."""
        if self._config is None:
            self._config = self._fn()
        return self._config

    def get_static_metadata(self):
        """Extract only static (non-dynamic) metadata."""
        config = self.get_config()
        return {k: v for k, v in config.items() if v != "dynamic"}

    def get_dynamic_keys(self):
        """Get list of keys marked as 'dynamic'."""
        config = self.get_config()
        return [k for k, v in config.items() if v == "dynamic"]
```

### KernelFunction Integration

```python
class KernelFunction:
    def __init__(self, fn, name, metadata=None):
        # Support both dict and MetaDataFunction
        if isinstance(metadata, MetaDataFunction):
            self._metadata = metadata.get_static_metadata()
            self._metadata_func = metadata
        else:
            self._metadata = metadata or {}
            self._metadata_func = None

    def _trace(self, builder):
        # ...
        with InsertionPoint(entry):
            # If metadata comes from MetaDataFunction, don't pass as kwargs
            if self._metadata_func is not None:
                self._fn(*proxy_args)
            else:
                self._fn(*proxy_args, **self._metadata)
```

## Usage Pattern

### Complete Example: TRowSum

```python
@pto_meta_data
def meta_data():
    return {
        "phys_row": 128,
        "phys_col": 64,
        "dst_col": 1,
        "dtype": pto.float32,
        "valid_row": "dynamic",
        "valid_col": "dynamic",
    }

@pto.kernel(metadata=meta_data)
def trowsum_case(
    src: pto.Tensor(pto.float32, 2),
    out: pto.Tensor(pto.float32, 2),
):
    # Extract static metadata
    static_meta = meta_data.get_static_metadata()
    phys_row = static_meta["phys_row"]
    phys_col = static_meta["phys_col"]
    dst_col = static_meta["dst_col"]
    dtype = static_meta["dtype"]

    # Extract dynamic metadata from tensor.shape
    valid_row = src.shape[0]
    valid_col = src.shape[1]

    # Static partition (ptoas requirement)
    src_part = src.partition(offsets=[0, 0], sizes=[phys_row, phys_col])
    out_part = out.partition(offsets=[0, 0], sizes=[phys_row, dst_col])

    # Allocate tiles with dynamic valid shapes
    src_tile = pto.alloc_tile(
        addr=0,
        physical_shape=(phys_row, phys_col),
        valid_row=valid_row.ssa,
        valid_col=valid_col.ssa,
        dtype=dtype,
    )

    # Pipeline
    pto.tload(src_part, src_tile)
    pto.trowsum(src_tile, tmp_tile, dst_tile)
    pto.tstore(out_part, dst_tile)
```

## Benefits

1. **Clean Separation**: Kernel logic separated from configuration
2. **Type Safety**: Metadata defined in one place
3. **Flexibility**: Mix static and dynamic metadata
4. **Readability**: Kernel signatures only show tensor parameters
5. **Maintainability**: Easy to modify metadata without touching kernel code

## Constraints

1. **Partition sizes must be static**: Use physical dimensions, not tensor.shape
2. **Tile valid shapes can be dynamic**: Use tensor.shape for valid_row/valid_col
3. **Static metadata extracted at kernel definition time**
4. **Dynamic metadata extracted at kernel trace time**

## Files Modified

- `frontend/pto_frontend/_metadata.py` - New file with MetaDataFunction
- `frontend/pto_frontend/_kernel.py` - Support MetaDataFunction in KernelFunction
- `frontend/pto_frontend/__init__.py` - Export pto_meta_data decorator
- `test/samples/frontend/TRowSum/test_trowsum_dynamic_v2.py` - Example usage

## Testing

Run the test suite:

```bash
# IR generation only
python3 test/samples/frontend/TRowSum/test_trowsum_dynamic_v2.py --ir-only

# IR + C++ generation (requires ptoas)
python3 test/samples/frontend/TRowSum/test_trowsum_dynamic_v2.py --cpp-only

# Full compilation (requires ptoas + bisheng + ASCEND_TOOLKIT_HOME)
python3 test/samples/frontend/TRowSum/test_trowsum_dynamic_v2.py

# Single case
python3 test/samples/frontend/TRowSum/test_trowsum_dynamic_v2.py --case 1
```

## Future Enhancements

1. **Dynamic partition support in ptoas**: Allow `!pto.partition_tensor_view<-1x-1xf32>`
2. **Metadata validation**: Type checking and constraint validation
3. **Metadata inheritance**: Base metadata + case-specific overrides
4. **Auto-extraction**: Automatically detect static vs dynamic metadata

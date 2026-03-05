# N-D Tiling and Layout Implementation Summary

## Overview

Implemented CuTe-style Layout abstraction and N-dimensional tiling for PTO frontend, enabling complex multi-core distribution patterns for Ascend NPU.

## New Files

### 1. `frontend/pto_frontend/_layout.py`
CuTe-style Layout abstraction for coordinate mapping:
- `Layout(shape, stride)`: Maps N-D logical coordinates to linear offsets
- Supports both static (int) and dynamic (ScalarValue) coordinates
- `slice(dim, idx)`: Fix one dimension, return lower-rank layout
- `compose(inner)`: Compose two layouts

### 2. `frontend/pto_frontend/_tiled_tensor_nd.py`
N-dimensional tiling with multi-core distribution:
- `TiledTensorND`: Tiles multiple dimensions simultaneously
  - `tile_sizes`: Size for each tensor dimension (None = full extent)
  - `tile_dims`: Which dimensions to iterate over
  - `for_each(ranges)`: Nested scf.for loops for N-D iteration
- `DistributedTiledTensorND`: Distributes N-D tiles across N-D core grid
  - Decomposes linear core_id into N-D coordinates
  - Computes per-core tile ranges for each dimension

## Modified Files

### 1. `frontend/pto_frontend/_tensor.py`
Added `tile_nd()` method to `_TensorProxy`:
```python
def tile_nd(self, tile_sizes, tile_dims):
    """Tile tensor along multiple dimensions."""
    return TiledTensorND(self, tile_sizes, tile_dims)
```

### 2. `frontend/pto_frontend/__init__.py`
Exported new APIs:
- `Layout`
- `TiledTensorND`
- `DistributedTiledTensorND`

## Test Cases

### 1. `test/samples/TiledTensor/test_tiled_nd_basic.py`
Basic N-D tiling without distribution:
- 2D tensor tiling with custom ranges
- Full range iteration

### 2. `test/samples/TiledTensor/test_tiled_nd_query.py`
BSND Query tensor scenario (simplified to 2D):
- Multi-dimensional core grid distribution
- Demonstrates typical attention pattern

### 3. `test/samples/TiledTensor/test_tiled_nd_add3d.py`
3D Add operator scenario (simplified to 2D):
- Per-dimension tile distribution
- Each core processes assigned tiles

### 4. `test/samples/TiledTensor/test_flash_attention_nd.py`
FlashAttention with N-D tiling:
- Outer loop: distributed Q tiles across cores
- Inner loop: each core iterates all K/V tiles
- Demonstrates nested tiling pattern

### 5. `test/samples/TiledTensor/test_layout_usage.py`
Layout abstraction usage examples:
- Static and dynamic coordinate mapping
- Custom stride patterns
- Layout composition

## Usage Examples

### Basic N-D Tiling
```python
# Tile 2D tensor along both dimensions
tiled = tensor.tile_nd(
    tile_sizes=(32, 128),
    tile_dims=[0, 1]
)

# Iterate with custom ranges
with tiled.for_each(ranges=[(0, 2, 1), (0, 4, 1)]) as (tile_idx, partition):
    # tile_idx is (i, j) tuple
    # partition is [32, 128] view
    tile_buf = pto.make_tile((32, 128), pto.float16, pto.VEC, addr=0)
    pto.tload(partition, tile_buf)
    pto.tstore(tile_buf, partition)
```

### Multi-Core Distribution
```python
# Distribute across 2D core grid
q_tiled = query.tile_nd(
    tile_sizes=(128, 64),
    tile_dims=[0]
)
q_dist = q_tiled.distribute_nd(core_grid=(2,))

# Each core processes its assigned tiles
with q_dist.for_each() as (tile_idx, q_view):
    # Only this core's tiles
    pto.tload(q_view, tile_q)
```

### FlashAttention Pattern
```python
# Outer: distributed Q tiles
q_dist = q.tile_nd(...).distribute_nd(core_grid=(2,))

# Inner: all K/V tiles
k_tiled = k.tile_nd(...)

with q_dist.for_each() as (q_idx, q_view):
    pto.tload(q_view, tile_q)

    # Each Q tile attends to all K/V
    with k_tiled.for_each() as (k_idx, k_view):
        v_view = v_tiled[k_idx]
        pto.tload(k_view, tile_k)
        pto.tload(v_view, tile_v)
        # Compute attention...

    pto.tstore(tile_out, out_view)
```

## Implementation Details

### Nested Loop Generation
Uses recursive approach to build properly nested scf.for loops:
```python
def _build_nested_loops(self, ranges, depth, indices):
    if depth == len(ranges):
        # Base case: yield partition
        pv = self._make_partition(tuple(indices))
        yield tuple(indices), pv
        return

    # Create one loop level
    loop = scf.ForOp(start, end, step, [])
    ip = InsertionPoint(loop.body)
    ip.__enter__()
    try:
        idx = ScalarValue(loop.induction_variable)
        yield from self._build_nested_loops(ranges, depth + 1, indices + [idx])
        scf.YieldOp([])
    finally:
        ip.__exit__(None, None, None)
```

### Core Coordinate Decomposition
Linear core_id → N-D core coordinates:
```python
# For dimension i in N-D core grid:
stride = prod(core_grid[:i])
core_coord[i] = (core_id // stride) % core_grid[i]
```

### Tile Range Computation
Per-core tile ranges with even distribution:
```python
tiles_per_core = ceildiv(num_tiles, num_cores_dim)
start = min(core_coord * tiles_per_core, num_tiles)
end = min((core_coord + 1) * tiles_per_core, num_tiles)
```

## Test Results

All tests pass:
- ✓ `test_tiled_nd_basic.py`: IR generation and ptoas compilation
- ✓ `test_flash_attention_nd.py`: Nested loops with distribution
- ✓ Existing 1D tiling tests remain functional

## Limitations

1. **Rank-2 tile_buf constraint**: PTO dialect requires 2D tiles, so 4D BSND scenarios need workarounds
2. **Dynamic dimensions**: Pre-existing PTO parser limitation with `-1` in partition_tensor_view
3. **Simplified distribution**: Current implementation uses blocked distribution; more complex patterns (cyclic, etc.) not yet supported

## Future Enhancements

1. Support for different distribution strategies (cyclic, block-cyclic)
2. Layout-based tile indexing for non-contiguous access patterns
3. Integration with causal masking and other attention variants
4. Performance optimizations for tile range computation

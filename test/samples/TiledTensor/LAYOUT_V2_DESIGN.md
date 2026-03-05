# Layout V2 API 设计文档

## 核心设计理念

基于你的需求，重新设计了Layout系统，实现了更符合CuTe理念的坐标化访问模式。

### 关键概念

1. **TensorLayout**: Tensor携带的布局信息 (shape + stride)
2. **TileLayout**: 描述tile的形状和stride模式
3. **TiledView**: split操作的结果，记录tile坐标范围
4. **TileCoordinate**: tile的多维坐标
5. **坐标化访问**: 通过坐标而不是partition直接访问数据

## API设计

### 1. Tensor获取Layout

```python
@pto.kernel
def my_kernel(tensor: pto.Tensor(pto.float16, 2)):
    # 获取tensor的layout (shape + stride)
    tensor_layout = tensor.get_layout()
    # TensorLayout(shape=(dynamic, dynamic), stride=(dynamic, 1))
```

### 2. 定义TileLayout

```python
# 定义tile模式 (必须是静态大小)
tile_layout = pto.TileLayout(shape=(64, 128))
# 默认stride为row-major: (128, 1)

# 也可以指定自定义stride
tile_layout = pto.TileLayout(shape=(64, 128), stride=(1, 64))  # column-major
```

### 3. Split操作

提供三种split策略：

#### 3.1 均匀分配 (split_even)

```python
num_cores = pto.get_block_num()
core_id = pto.get_block_idx()

# 将tensor均匀分配到各个core
tiled = pto.split_even(tensor_layout, tile_layout, num_cores, core_id)
```

#### 3.2 顺序遍历 (split_sequential)

```python
# 不分配，遍历所有tiles (用于inner loop)
tiled = pto.split_sequential(tensor_layout, tile_layout)
```

#### 3.3 Causal分配 (split_causal)

```python
# 用于attention的causal masking场景
tiled = pto.split_causal(tensor_layout, tile_layout, num_cores, core_id)
```

### 4. 坐标化迭代和访问

```python
# 分配tile buffer
tile_buf = pto.make_tile((64, 128), pto.float16, pto.VEC, addr=0)

# 迭代分配的tiles
with tiled.for_each() as coord:
    # coord是TileCoordinate，包含(tile_i, tile_j)

    # 使用坐标加载数据
    pto.tload_tile(tensor, coord, tile_layout, tile_buf)

    # 处理数据...

    # 使用坐标存储数据
    pto.tstore_tile(tile_buf, out, coord, tile_layout)
```

## 完整示例

### 示例1: 简单的均匀分配

```python
@pto.kernel
def even_split_example(
    tensor: pto.Tensor(pto.float16, 2),
    out: pto.Tensor(pto.float16, 2),
):
    # 1. 获取layout
    tensor_layout = tensor.get_layout()

    # 2. 定义tile模式
    tile_layout = pto.TileLayout(shape=(64, 128))

    # 3. 获取core信息
    num_cores = pto.get_block_num()
    core_id = pto.get_block_idx()

    # 4. 均匀分配
    tiled = pto.split_even(tensor_layout, tile_layout, num_cores, core_id)

    # 5. 分配tile buffer
    tile_buf = pto.make_tile((64, 128), pto.float16, pto.VEC, addr=0)

    # 6. 坐标化迭代
    with tiled.for_each() as coord:
        pto.tload_tile(tensor, coord, tile_layout, tile_buf)
        pto.tstore_tile(tile_buf, out, coord, tile_layout)
```

### 示例2: FlashAttention

```python
@pto.kernel
def flash_attention(
    q: pto.Tensor(pto.float16, 2),  # [S_q, D]
    k: pto.Tensor(pto.float16, 2),  # [S_kv, D]
    v: pto.Tensor(pto.float16, 2),  # [S_kv, D]
    out: pto.Tensor(pto.float16, 2),
):
    # 获取layouts
    q_layout = q.get_layout()
    k_layout = k.get_layout()
    v_layout = v.get_layout()

    # 定义tile模式
    tile_layout = pto.TileLayout(shape=(128, 64))

    # 获取core信息
    num_cores = pto.get_block_num()
    core_id = pto.get_block_idx()

    # Q: 分配到各个core
    q_tiled = pto.split_even(q_layout, tile_layout, num_cores, core_id)

    # K/V: 每个core遍历所有tiles
    k_tiled = pto.split_sequential(k_layout, tile_layout)
    v_tiled = pto.split_sequential(v_layout, tile_layout)

    # 分配tile buffers
    tile_q = pto.make_tile((128, 64), pto.float16, pto.VEC, addr=0)
    tile_k = pto.make_tile((128, 64), pto.float16, pto.VEC, addr=0x10000)
    tile_v = pto.make_tile((128, 64), pto.float16, pto.VEC, addr=0x20000)
    tile_out = pto.make_tile((128, 64), pto.float16, pto.VEC, addr=0x30000)

    # 外层循环: 分配的Q tiles
    with q_tiled.for_each() as q_coord:
        pto.tload_tile(q, q_coord, tile_layout, tile_q)

        # 内层循环: 所有K/V tiles
        with k_tiled.for_each() as kv_coord:
            pto.tload_tile(k, kv_coord, tile_layout, tile_k)
            pto.tload_tile(v, kv_coord, tile_layout, tile_v)

            # Attention计算
            pto.tadd(tile_q, tile_k, tile_out)
            pto.tadd(tile_out, tile_v, tile_out)

        # 存储结果
        pto.tstore_tile(tile_out, out, q_coord, tile_layout)
```

### 示例3: Causal Masking

```python
@pto.kernel
def flash_attention_causal(
    q: pto.Tensor(pto.float16, 2),
    k: pto.Tensor(pto.float16, 2),
    v: pto.Tensor(pto.float16, 2),
    out: pto.Tensor(pto.float16, 2),
):
    q_layout = q.get_layout()
    k_layout = k.get_layout()
    v_layout = v.get_layout()

    tile_layout = pto.TileLayout(shape=(128, 64))

    num_cores = pto.get_block_num()
    core_id = pto.get_block_idx()

    # 使用causal分配策略
    q_tiled = pto.split_causal(q_layout, tile_layout, num_cores, core_id)
    k_tiled = pto.split_sequential(k_layout, tile_layout)
    v_tiled = pto.split_sequential(v_layout, tile_layout)

    tile_q = pto.make_tile((128, 64), pto.float16, pto.VEC, addr=0)
    tile_k = pto.make_tile((128, 64), pto.float16, pto.VEC, addr=0x10000)
    tile_v = pto.make_tile((128, 64), pto.float16, pto.VEC, addr=0x20000)
    tile_out = pto.make_tile((128, 64), pto.float16, pto.VEC, addr=0x30000)

    with q_tiled.for_each() as q_coord:
        pto.tload_tile(q, q_coord, tile_layout, tile_q)

        with k_tiled.for_each() as kv_coord:
            # 实际应用中，这里会检查 kv_coord[0] <= q_coord[0]
            # 只处理causal范围内的tiles
            pto.tload_tile(k, kv_coord, tile_layout, tile_k)
            pto.tload_tile(v, kv_coord, tile_layout, tile_v)

            pto.tadd(tile_q, tile_k, tile_out)
            pto.tadd(tile_out, tile_v, tile_out)

        pto.tstore_tile(tile_out, out, q_coord, tile_layout)
```

## 实现细节

### 新增文件

1. **`_layout_v2.py`**: 核心Layout类
   - `TensorLayout`: 动态shape + stride
   - `TileLayout`: 静态tile shape + stride
   - `TileCoordinate`: tile坐标
   - `TiledView`: split结果，支持for_each迭代

2. **`_split_utils.py`**: Split策略实现
   - `split_even()`: 均匀分配
   - `split_sequential()`: 顺序遍历
   - `split_causal()`: Causal分配

### 修改文件

1. **`_tensor.py`**:
   - 添加 `get_layout()` 方法
   - 添加 `partition_at_coord()` 方法

2. **`_ops.py`**:
   - 添加 `tload_tile()` 坐标化加载
   - 添加 `tstore_tile()` 坐标化存储

3. **`__init__.py`**:
   - 导出新的Layout API
   - 导出split工具函数

## 优势

1. **符合CuTe理念**: 坐标化访问，Layout作为一等公民
2. **类型安全**: TileCoordinate明确表示tile坐标
3. **灵活性**: 支持多种split策略，易于扩展
4. **动态shape支持**: TensorLayout支持动态维度
5. **清晰的抽象层次**:
   - TensorLayout描述数据布局
   - TileLayout描述切分模式
   - TiledView描述分配结果
   - TileCoordinate描述访问位置

## 测试结果

所有测试通过：
- ✓ `test_layout_v2_basic.py`: 基本Layout API
- ✓ `test_flash_attention_v2.py`: FlashAttention with Layout
- ✓ IR生成和ptoas编译都成功

## 与旧API对比

### 旧API (tile_nd + distribute_nd):
```python
q_tiled = q.tile_nd(tile_sizes=(128, 64), tile_dims=[0])
q_dist = q_tiled.distribute_nd(core_grid=(2,))
with q_dist.for_each() as (tile_idx, partition):
    pto.tload(partition, tile_buf)
```

### 新API (Layout + split):
```python
q_layout = q.get_layout()
tile_layout = pto.TileLayout(shape=(128, 64))
q_tiled = pto.split_even(q_layout, tile_layout, num_cores, core_id)
with q_tiled.for_each() as coord:
    pto.tload_tile(q, coord, tile_layout, tile_buf)
```

新API更加：
- **显式**: Layout信息明确可见
- **灵活**: 易于实现自定义split策略
- **统一**: 所有操作都基于坐标
- **可组合**: Layout可以进行各种变换和组合

# @pto_meta_data 实现总结

## 完成的工作

### 1. 新增 `@pto_meta_data` 装饰器

**文件**: `frontend/pto_frontend/_metadata.py`

实现了 `MetaDataFunction` 类，支持：
- 静态元数据（编译时常量）
- 动态元数据（运行时从 tensor.shape 提取）
- 元数据缓存和查询接口

```python
@pto_meta_data
def meta():
    return {
        "phys_row": 128,        # 静态
        "phys_col": 64,         # 静态
        "dtype": pto.float32,   # 静态
        "valid_row": "dynamic", # 动态
        "valid_col": "dynamic", # 动态
    }
```

### 2. 集成到 KernelFunction

**文件**: `frontend/pto_frontend/_kernel.py`

修改了 `KernelFunction` 类以支持 `MetaDataFunction`：
- 自动识别 `MetaDataFunction` 类型
- 提取静态元数据传递给 kernel
- 不将元数据作为 kwargs 传递给 kernel 函数

### 3. 导出到公共 API

**文件**: `frontend/pto_frontend/__init__.py`

添加了 `pto_meta_data` 到导出列表，用户可以通过 `import pto_frontend as pto` 使用。

### 4. 完整测试用例

**文件**: `test/samples/frontend/TRowSum/test_trowsum_dynamic_v2.py`

实现了 8 个 TRowSum 测试用例，展示：
- 清晰的 kernel 签名（只有 tensor 参数）
- 静态和动态元数据的混合使用
- 静态 partition 大小（ptoas 要求）
- 动态 tile valid shapes

## 关键设计决策

### 1. 静态 vs 动态元数据

**静态元数据**（编译时常量）：
- 物理 tile 尺寸 (phys_row, phys_col)
- 数据类型 (dtype)
- 输出列数 (dst_col)
- 用于 partition 大小（ptoas 要求静态）

**动态元数据**（运行时提取）：
- 有效行数 (valid_row = src.shape[0])
- 有效列数 (valid_col = src.shape[1])
- 用于 tile 的 valid_row/valid_col 参数

### 2. Partition 大小必须静态

**重要约束**: ptoas 目前不支持动态大小的 partition_tensor_view

```python
# ✓ 正确：使用静态物理尺寸
src_part = src.partition(offsets=[0, 0], sizes=[phys_row, phys_col])
# 生成: !pto.partition_tensor_view<128x64xf32>

# ✗ 错误：使用动态 tensor.shape
src_part = src.partition(offsets=[0, 0], sizes=[src.shape[0], src.shape[1]])
# 生成: !pto.partition_tensor_view<-1x-1xf32> (ptoas 解析错误)
```

### 3. Tile Valid Shapes 可以动态

Tile 的 valid_row/valid_col 可以使用动态值：

```python
tile = pto.alloc_tile(
    addr=0,
    physical_shape=(phys_row, phys_col),  # 静态
    valid_row=valid_row.ssa,              # 动态（从 src.shape[0]）
    valid_col=valid_col.ssa,              # 动态（从 src.shape[1]）
)
```

## 测试结果

所有 8 个测试用例通过：
- ✓ IR 生成成功
- ✓ C++ 代码生成成功（通过 ptoas）
- ✓ 编译成功（通过 bisheng）

```bash
# 测试命令
python3 test_trowsum_dynamic_v2.py --case 1 2 3 4 5 6 7 8

# 输出
PASS: Case 1 emit_ir()
PASS: Case 1 emit_cpp()
PASS: Case 1 compile()
...
All TRowSum @pto_meta_data tests passed!
```

## 使用示例

### 完整示例

```python
import pto_frontend as pto

# 1. 定义元数据
@pto.pto_meta_data
def meta_data():
    return {
        "phys_row": 128,
        "phys_col": 64,
        "dst_col": 1,
        "dtype": pto.float32,
        "valid_row": "dynamic",
        "valid_col": "dynamic",
    }

# 2. 定义 kernel（签名清晰，只有 tensor 参数）
@pto.kernel(metadata=meta_data)
def trowsum_kernel(
    src: pto.Tensor(pto.float32, 2),
    out: pto.Tensor(pto.float32, 2),
):
    # 提取静态元数据
    static = meta_data.get_static_metadata()
    phys_row = static["phys_row"]
    phys_col = static["phys_col"]
    dst_col = static["dst_col"]
    dtype = static["dtype"]

    # 提取动态元数据
    valid_row = src.shape[0]
    valid_col = src.shape[1]

    # 静态 partition（ptoas 要求）
    src_part = src.partition(offsets=[0, 0], sizes=[phys_row, phys_col])
    out_part = out.partition(offsets=[0, 0], sizes=[phys_row, dst_col])

    # 分配 tiles（动态 valid shapes）
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

# 3. 使用 kernel
trowsum_kernel.emit_ir()   # 生成 IR
trowsum_kernel.emit_cpp()  # 生成 C++
trowsum_kernel.compile()   # 编译成 .so
```

## 优势

1. **清晰的关注点分离**: kernel 逻辑与配置分离
2. **类型安全**: 元数据在一个地方定义
3. **灵活性**: 支持静态和动态元数据混合
4. **可读性**: kernel 签名只显示 tensor 参数
5. **可维护性**: 修改元数据不需要改动 kernel 代码

## 文档

- `METADATA_DESIGN.md` - 详细设计文档
- `README_METADATA.md` - 使用指南
- `test_trowsum_dynamic_v2.py` - 完整示例代码

## 环境要求

```bash
# 必需工具
- ptoas (在 PATH 中)
- bisheng (在 PATH 中，用于编译)
- ASCEND_TOOLKIT_HOME (环境变量)

# Python 环境
- MLIR Python bindings
- PTO frontend
```

## 未来改进

1. **ptoas 支持动态 partition**: 允许 `!pto.partition_tensor_view<-1x-1xf32>`
2. **元数据验证**: 类型检查和约束验证
3. **元数据继承**: 基础元数据 + case 特定覆盖
4. **自动检测**: 自动识别静态 vs 动态元数据

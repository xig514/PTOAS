# PTOAS JIT Implementation Status

## ✅ 已完成的功能

### 1. 核心 JIT 框架
- **`pto.compile(kernel_fn)`** - 将 `@pto.kernel` 编译为 `.so` 共享库
- **`pto.launch(compiled, *args)`** - 通过 ctypes 调用编译后的 kernel
- **`@pto.jit` 装饰器** - 标记 host 端函数

### 2. 参数验证与转换
- **类型检查**：验证 torch.Tensor 的 dtype 和 rank
- **Shape 验证**：支持静态 shape 和 DynVar 动态 shape 的一致性检查
- **ctypes 转换**：Tensor → `void* + int32_t dims...`，标量 → 对应 ctypes 类型

### 3. Pythonic 语法增强
- **`for i in pto.range(N)`** - 迭代器风格循环（替代 `with pto.for_range(...)`）
- **`Tensor[[M, N], dtype]`** - 支持动态 shape 注解
- **`DynVar("M")`** - 动态维度变量

### 4. 文件结构
```
frontend/pto_frontend/
├── jit.py              # compile(), launch(), @jit 装饰器
├── _kernel.py          # KernelFunction.compile() 实现
├── _control_flow.py    # range() 迭代器
├── _tensor.py          # Tensor[[M, N], dtype] 支持
└── __init__.py         # 导出 compile, launch
```

### 5. 测试用例
- **`test/samples/frontend/test_jit_launch.py`** - 端到端测试
- **`test/samples/frontend/test_dynamic_add.py`** - 动态 shape 测试

## ⚠️ 待解决的问题

### Bisheng 编译器配置

**错误信息**：
```
error: function type 'void (__ubuf__ half *, ...) noexcept' of 'vadd'
does not support the given target feature
```

**原因分析**：
1. `--npu-arch=dav-2201` 可能不是正确的架构字符串
2. 或者需要额外的编译器标志来启用 AICore 向量指令

**当前编译命令**：
```bash
bisheng -I$ASCEND_TOOLKIT_HOME/include \
  -fPIC -c -O2 -std=c++17 \
  -xcce --npu-arch=dav-2201 \
  -mllvm -cce-aicore-stack-size=0x8000 \
  -mllvm -cce-aicore-function-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false \
  -DMEMORY_BASE \
  kernel.cpp -o kernel.o
```

**需要验证**：
- 正确的 `--npu-arch` 值（可能是 `ascend910b`、`910b`、`dav910b` 等）
- 是否需要 `-D__CCE_AICORE__=XXX` 宏定义
- bisheng 编译器版本是否与 CANN 8.5.0 兼容

## 🔧 调试步骤

### 1. 检查可用的 NPU 架构
```bash
bisheng --help | grep -i npu
# 或
bisheng --version
```

### 2. 查看现有工作示例
如果你有其他能成功编译的 PTO kernel（例如通过 pypto 或 CANN 示例），检查它们的编译命令：
```bash
# 在编译日志中查找 bisheng 命令
```

### 3. 测试最小示例
```bash
cd /data/g00895580/ptoas/PTOAS/test/samples/frontend/.ptodsl_jit/dynamic_add_kernel
bisheng -I/usr/local/Ascend/cann-8.5.0/include \
  -fPIC -c -O2 -std=c++17 \
  -xcce --npu-arch=<正确的架构> \
  -DMEMORY_BASE \
  kernel.cpp -o kernel.o
```

### 4. 参考 pypto 实现
pypto 项目已经解决了这个问题，可以参考：
https://github.com/xig514/pypto/blob/main/python/pypto/frontend/jit.py

## 📝 使用示例（一旦编译问题解决）

```python
import pto_frontend as pto
import torch
import torch_npu

M = pto.DynVar("M")
N = pto.DynVar("N")

@pto.kernel
def add_kernel(
    x: pto.Tensor[[M, N], pto.float16],
    y: pto.Tensor[[M, N], pto.float16],
    z: pto.Tensor[[M, N], pto.float16],
):
    tile_a = pto.make_tile((64, 128), pto.float16, pto.VEC, addr=0)
    tile_b = pto.make_tile((64, 128), pto.float16, pto.VEC, addr=16384)
    tile_c = pto.make_tile((64, 128), pto.float16, pto.VEC, addr=32768)

    m_loops = (M + 63) // 64
    n_loops = (N + 127) // 128

    for i in pto.range(m_loops):
        for j in pto.range(n_loops):
            m_offset = i * 64
            n_offset = j * 128

            pv_x = x.partition(offsets=[m_offset, n_offset], sizes=[64, 128])
            pv_y = y.partition(offsets=[m_offset, n_offset], sizes=[64, 128])
            pv_z = z.partition(offsets=[m_offset, n_offset], sizes=[64, 128])

            pto.tload(pv_x, tile_a)
            pto.tload(pv_y, tile_b)
            pto.tadd(tile_a, tile_b, tile_c)
            pto.tstore(pv_z, tile_c)


@pto.jit
def test():
    compiled = pto.compile(add_kernel)

    x = torch.rand(128, 256, device="npu:0", dtype=torch.float16)
    y = torch.rand(128, 256, device="npu:0", dtype=torch.float16)
    z = torch.empty_like(x)

    pto.launch(compiled, x, y, z)
    torch.npu.synchronize()

    torch.testing.assert_close(z, x + y)
    print("✅ Test passed!")

test()
```

## 🎯 下一步

1. **解决编译器配置** - 确定正确的 `--npu-arch` 和编译标志
2. **验证 kernel 执行** - 确保生成的 `.so` 能在 NPU 上正确运行
3. **性能优化** - 添加编译缓存、并行编译等优化
4. **错误处理** - 改进编译错误的诊断信息

## 📚 参考资料

- CANN 开发文档：https://www.hiascend.com/document
- pypto 参考实现：https://github.com/xig514/pypto
- Bisheng 编译器文档：查看 CANN toolkit 安装目录

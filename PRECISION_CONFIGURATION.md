# 模型精度配置指南

## 概述

本指南介绍如何在 Embedding Model Provider API 中配置模型精度，以优化内存使用和性能。

## 支持的精度类型

### 浮点精度
- **fp32** (float32): 标准32位浮点数，最高精度，内存占用最大
- **fp16** (float16): 16位浮点数，内存减半，性能提升，精度略有损失
- **bf16** (bfloat16): 16位脑浮点数，专为AI设计，保持fp32的指数范围

### 量化精度
- **int8**: 8位整数量化，内存减少75%，性能显著提升
- **int4**: 4位整数量化，内存减少87.5%，最大压缩比

## 环境变量配置

### 基础精度配置

```bash
# 设置全局默认精度 (可选值: auto, fp32, fp16, bf16, int8, int4)
EMB_PROVIDER_MODEL_PRECISION=fp16

# 启用量化支持
EMB_PROVIDER_ENABLE_QUANTIZATION=true

# 设置量化方法 (int8 或 int4)
EMB_PROVIDER_QUANTIZATION_METHOD=int8

# 启用GPU内存优化
EMB_PROVIDER_ENABLE_GPU_MEMORY_OPTIMIZATION=true
```

### 模型特定精度覆盖

```bash
# 为特定模型设置精度 (JSON格式)
EMB_PROVIDER_MODEL_PRECISION_OVERRIDES='{
  "sentence-transformers/all-MiniLM-L6-v2": "fp16",
  "BAAI/bge-large-en-v1.5": "int8",
  "thenlper/gte-base": "bf16"
}'
```

## 配置优先级

精度选择遵循以下优先级：

1. **模型特定覆盖**: 通过 `EMB_PROVIDER_MODEL_PRECISION_OVERRIDES` 设置
2. **全局配置**: 通过 `EMB_PROVIDER_MODEL_PRECISION` 设置
3. **模型原生精度**: 从模型配置文件中检测
4. **设备兼容性**: 根据设备能力自动选择
5. **默认安全选择**: fp32 确保兼容性

## 设备兼容性

### CPU 设备
- 推荐使用: **fp32** (最佳兼容性)
- 支持: fp16 (需要CPU支持)
- 不支持: bf16, int8, int4 (需要GPU)

### GPU 设备
- **NVIDIA RTX 20/30/40 系列**: 支持所有精度
- **NVIDIA GTX 10 系列**: 支持 fp32, fp16
- **AMD GPU**: 支持 fp32, fp16 (需要ROCm)

### bfloat16 支持检测
- RTX 30 系列及更新 GPU 支持 bfloat16
- 系统会自动检测并选择合适的精度

## 性能与内存对比

| 精度 | 内存占用 | 推理速度 | 精度损失 | 适用场景 |
|------|----------|----------|----------|----------|
| fp32 | 100% | 基准 | 无 | 高精度要求，小模型 |
| fp16 | 50% | 2-3x | 轻微 | 大多数场景，平衡选择 |
| bf16 | 50% | 2-3x | 轻微 | AI专用，大模型训练 |
| int8 | 25% | 3-4x | 中等 | 内存受限，性能优先 |
| int4 | 12.5% | 4-5x | 显著 | 极端内存限制 |

## 使用示例

### 1. 高性能配置 (GPU)

```bash
# 使用 fp16 获得最佳性能平衡
EMB_PROVIDER_MODEL_PRECISION=fp16
EMB_PROVIDER_ENABLE_GPU_MEMORY_OPTIMIZATION=true
```

### 2. 内存优化配置

```bash
# 使用 int8 量化减少内存占用
EMB_PROVIDER_MODEL_PRECISION=int8
EMB_PROVIDER_ENABLE_QUANTIZATION=true
EMB_PROVIDER_ENABLE_GPU_MEMORY_OPTIMIZATION=true
```

### 3. 混合精度配置

```bash
# 为不同模型设置不同精度
EMB_PROVIDER_MODEL_PRECISION_OVERRIDES='{
  "small-models": "fp16",
  "large-models": "int8",
  "precision-critical": "fp32"
}'
```

### 4. 生产环境推荐

```bash
# 平衡性能和精度的生产配置
EMB_PROVIDER_MODEL_PRECISION=fp16
EMB_PROVIDER_ENABLE_GPU_MEMORY_OPTIMIZATION=true
EMB_PROVIDER_LOG_LEVEL=INFO
```

## 故障排除

### 常见问题

1. **精度不兼容错误**
   - 症状: `RuntimeError: expected scalar type Float but found Half`
   - 解决方案: 使用 `fp32` 或检查模型兼容性

2. **内存不足错误**
   - 症状: `CUDA out of memory`
   - 解决方案: 启用量化 (`int8`/`int4`) 或减少批大小

3. **bfloat16 不支持**
   - 症状: 自动回退到 fp16
   - 解决方案: 检查GPU是否支持 bfloat16

### 调试信息

启用调试日志查看精度选择过程：

```bash
EMB_PROVIDER_LOG_LEVEL=DEBUG
```

日志将显示：
- 检测到的设备能力
- 模型原生精度
- 最终选择的精度
- 量化配置信息

## 最佳实践

1. **测试不同精度**: 在生产部署前测试不同精度设置
2. **监控性能**: 使用性能监控工具评估不同配置
3. **模型特定优化**: 为不同模型类型选择最优精度
4. **渐进式部署**: 从 fp32 开始，逐步测试更激进的优化

## 相关配置

查看完整配置选项：

```python
from emb_model_provider.core.config import Config

config = Config.from_env()
print(config.model_precision)  # 当前精度配置
print(config.model_precision_overrides)  # 模型特定覆盖
```

## 技术支持

如有问题，请参考：
- [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers)
- [PyTorch 精度指南](https://pytorch.org/docs/stable/notes/cuda.html)
- [ModelScope 模型加载](https://modelscope.cn/docs)
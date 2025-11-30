# ModelScope 模型加载使用指南

## 概述

Embedding Model Provider API 支持从 ModelScope 平台加载模型，为中文用户提供更便捷的模型访问方式。本指南详细说明如何配置和使用 ModelScope 模型加载功能。

## 前置要求

### 安装 ModelScope 库

在使用 ModelScope 模型加载功能之前，需要安装 ModelScope Python 库：

```bash
pip install modelscope
```

或者使用 uv 安装：

```bash
uv add modelscope
```

### 支持的模型

ModelScope 提供了丰富的中文优化模型，以下是一些推荐的嵌入模型：

- **damo/nlp_gte_sentence-embedding_chinese-base**: 通用中文文本嵌入模型（384维）
- **damo/nlp_gte_sentence-embedding_chinese-large**: 高质量中文文本嵌入模型（1024维）
- **damo/nlp_corom_sentence-embedding_chinese-base**: 多语言中文优化嵌入模型
- **iic/nlp_gte_sentence-embedding_chinese-tiny**: 轻量级中文嵌入模型（适合资源受限环境）

## 配置说明

### 环境变量配置

在 `.env` 文件中添加以下配置来启用 ModelScope 模型加载：

```env
# =============================================================================
# ModelScope 模型加载配置
# =============================================================================

# 模型来源选择（可选值：huggingface, modelscope, auto）
# auto: 自动选择，优先使用 ModelScope（如果模型名称以 damo/ 或 iic/ 开头）
EMB_PROVIDER_MODEL_SOURCE=modelscope

# ModelScope 模型名称（当 EMB_PROVIDER_MODEL_SOURCE=modelscope 时使用）
EMB_PROVIDER_MODEL_NAME=damo/nlp_gte_sentence-embedding_chinese-base

# 模型精度配置（可选值：auto, fp32, fp16, bf16, int8, int4）
EMB_PROVIDER_MODEL_PRECISION=auto

# 是否启用量化（当精度为 int8 或 int4 时生效）
EMB_PROVIDER_ENABLE_QUANTIZATION=false

# 模型缓存目录（可选，留空使用默认缓存）
# Windows: EMB_PROVIDER_MODELSCOPE_CACHE_DIR=D:\cache\modelscope
# Linux/Mac: EMB_PROVIDER_MODELSCOPE_CACHE_DIR=/cache/modelscope
EMB_PROVIDER_MODELSCOPE_CACHE_DIR=

# 是否信任远程代码（某些模型需要设置为 true）
EMB_PROVIDER_MODELSCOPE_TRUST_REMOTE_CODE=false
```

### 多模型配置示例

使用多模型映射功能，可以同时支持 HuggingFace 和 ModelScope 模型：

```env
# 多模型映射配置（JSON格式）
EMB_PROVIDER_MODEL_MAPPING={
  "gte-chinese": {
    "name": "damo/nlp_gte_sentence-embedding_chinese-base",
    "source": "modelscope",
    "precision": "fp16"
  },
  "gte-english": {
    "name": "sentence-transformers/all-MiniLM-L12-v2", 
    "source": "huggingface",
    "precision": "fp16"
  },
  "multilingual": {
    "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "source": "huggingface",
    "precision": "fp32"
  }
}
```

## 使用方式

### 1. 基本使用

配置完成后，启动服务即可使用 ModelScope 模型：

```bash
uv run python -m emb_model_provider.main
```

### 2. API 调用示例

#### 查看可用模型

```bash
curl http://localhost:9000/v1/models
```

响应示例：
```json
{
  "object": "list",
  "data": [
    {
      "id": "damo/nlp_gte_sentence-embedding_chinese-base",
      "object": "model", 
      "created": 1700000000,
      "owned_by": "organization-owner"
    }
  ]
}
```

#### 创建文本嵌入

```bash
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "这是一个中文文本示例",
    "model": "damo/nlp_gte_sentence-embedding_chinese-base"
  }'
```

#### 使用 OpenAI Python 客户端

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",
    base_url="http://localhost:9000/v1"
)

response = client.embeddings.create(
    model="damo/nlp_gte_sentence-embedding_chinese-base",
    input="这是一个中文文本示例"
)

print(f"嵌入维度: {len(response.data[0].embedding)}")
print(f"嵌入向量: {response.data[0].embedding[:5]}...")
```

### 3. 多模型使用

当配置了多模型映射时，可以通过别名调用不同的模型：

```bash
# 使用中文优化模型
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "这是一个中文文本示例",
    "model": "gte-chinese"
  }'

# 使用英文模型  
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is an English text example",
    "model": "gte-english"
  }'
```

## 性能优化建议

### 1. 精度选择

- **fp32**: 最高精度，适合对准确性要求极高的场景
- **fp16**: 推荐配置，在保持较高精度的同时减少内存占用
- **int8**: 适合资源受限环境，精度略有损失
- **int4**: 极致压缩，适合移动设备或边缘计算

### 2. 批处理配置

根据模型大小和设备性能调整批处理参数：

```env
# 中文模型通常较小，可以设置较大的批处理大小
EMB_PROVIDER_MAX_BATCH_SIZE=64

# 启用动态批处理优化
EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING=true
EMB_PROVIDER_MAX_WAIT_TIME_MS=50
EMB_PROVIDER_MIN_BATCH_SIZE=8
```

### 3. 设备选择

```env
# CPU 环境（推荐配置）
EMB_PROVIDER_DEVICE=cpu
EMB_PROVIDER_MAX_BATCH_SIZE=32

# GPU 环境（如果有可用 GPU）
EMB_PROVIDER_DEVICE=cuda
EMB_PROVIDER_MAX_BATCH_SIZE=64
```

## 故障排除

### 1. 模型加载失败

**问题**: 模型加载失败，提示 "Model not found" 或 "ImportError"

**解决方案**:
- 检查 ModelScope 库是否正确安装：`pip show modelscope`
- 确认模型名称拼写正确
- 检查网络连接，确保可以访问 ModelScope 平台

### 2. 内存不足

**问题**: 加载模型时出现内存不足错误

**解决方案**:
- 降低模型精度：设置 `EMB_PROVIDER_MODEL_PRECISION=fp16`
- 启用量化：设置 `EMB_PROVIDER_ENABLE_QUANTIZATION=true`
- 减少批处理大小：设置 `EMB_PROVIDER_MAX_BATCH_SIZE=16`

### 3. 性能问题

**问题**: 嵌入生成速度较慢

**解决方案**:
- 启用动态批处理：设置 `EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING=true`
- 调整批处理参数：适当增加 `EMB_PROVIDER_MAX_BATCH_SIZE`
- 使用 GPU：设置 `EMB_PROVIDER_DEVICE=cuda`（如果可用）

## 最佳实践

### 1. 开发环境配置

```env
# 开发环境配置（快速启动）
EMB_PROVIDER_MODEL_SOURCE=modelscope
EMB_PROVIDER_MODEL_NAME=damo/nlp_gte_sentence-embedding_chinese-base
EMB_PROVIDER_MODEL_PRECISION=fp16
EMB_PROVIDER_MAX_BATCH_SIZE=16
EMB_PROVIDER_DEVICE=cpu
EMB_PROVIDER_LOG_LEVEL=INFO
```

### 2. 生产环境配置

```env
# 生产环境配置（高性能）
EMB_PROVIDER_MODEL_SOURCE=modelscope
EMB_PROVIDER_MODEL_NAME=damo/nlp_gte_sentence-embedding_chinese-base
EMB_PROVIDER_MODEL_PRECISION=fp16
EMB_PROVIDER_MAX_BATCH_SIZE=64
EMB_PROVIDER_DEVICE=cuda
EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING=true
EMB_PROVIDER_MAX_WAIT_TIME_MS=50
EMB_PROVIDER_LOG_LEVEL=WARNING
```

### 3. 多模型应用配置

```env
# 多模型应用配置
EMB_PROVIDER_MODEL_MAPPING={
  "chinese": {
    "name": "damo/nlp_gte_sentence-embedding_chinese-base",
    "source": "modelscope",
    "precision": "fp16"
  },
  "english": {
    "name": "sentence-transformers/all-MiniLM-L12-v2",
    "source": "huggingface", 
    "precision": "fp16"
  },
  "multilingual": {
    "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "source": "huggingface",
    "precision": "fp32"
  }
}
EMB_PROVIDER_MAX_BATCH_SIZE=32
EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING=true
```

## 相关文档

- [ModelScope 集成设计文档](modelscope-integration-design.md)
- [ModelScope 实现指南](modelscope-implementation-guide.md)
- [多模型支持文档](MULTI_MODEL_SUPPORT.md)
- [配置说明文档](MODEL_CONFIG.md)

## 技术支持

如果在使用过程中遇到问题，请参考：

1. 查看服务日志：`tail -f logs/emb_model_provider.log`
2. 检查 ModelScope 官方文档：https://modelscope.cn
3. 查看项目 GitHub Issues
4. 联系开发团队获取技术支持
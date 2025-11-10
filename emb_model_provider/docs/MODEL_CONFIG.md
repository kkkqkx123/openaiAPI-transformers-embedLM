# 嵌入模型提供商 - 配置和使用指南

## 概述

嵌入模型提供商是一个基于 FastAPI 的服务，提供与 OpenAI 兼容的嵌入 API 功能。它使用基于 Transformer 的模型生成高质量文本嵌入，并基于 PyTorch 和 Hugging Face Transformers 构建。

## 配置

### 环境变量

服务通过以 `EMB_PROVIDER_` 为前缀的环境变量进行配置。可用以下配置选项：

#### 模型配置
- `EMB_PROVIDER_MODEL_PATH` (默认值: `D:\models\multilingual-MiniLM-L12-v2`): 模型目录路径
- `EMB_PROVIDER_MODEL_NAME` (默认值: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`): 要使用的模型名称 (用于从 Hugging Face Hub 下载)

#### Transformers 模型加载配置
- `EMB_PROVIDER_LOAD_FROM_TRANSFORMERS` (默认值: `false`): 是否直接从 transformers 加载模型（不使用本地缓存）
- `EMB_PROVIDER_TRANSFORMERS_MODEL_NAME` (默认值: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`): 直接从 transformers 加载的模型名称
- `EMB_PROVIDER_TRANSFORMERS_CACHE_DIR` (默认值: `None`): Transformers 模型的自定义缓存目录
- `EMB_PROVIDER_TRANSFORMERS_TRUST_REMOTE_CODE` (默认值: `false`): 加载 transformers 时是否信任远程代码

#### 处理配置
- `EMB_PROVIDER_MAX_BATCH_SIZE` (默认值: `32`): 最大批处理大小 (1-512)
- `EMB_PROVIDER_MAX_CONTEXT_LENGTH` (默认值: `512`): 最大上下文长度（以 tokens 为单位）(1-2048)
- `EMB_PROVIDER_EMBEDDING_DIMENSION` (默认值: `384`): 嵌入向量的维度

#### 动态批处理配置
- `EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING` (默认值: `true`): 启用动态批处理以提高吞吐量
- `EMB_PROVIDER_MAX_WAIT_TIME_MS` (默认值: `100`): 动态批处理的最大等待时间（毫秒）(10-1000)
- `EMB_PROVIDER_MIN_BATCH_SIZE` (默认值: `1`): 动态批处理的最小批处理大小 (1-32)

#### 内存优化配置
- `EMB_PROVIDER_ENABLE_LENGTH_GROUPING` (默认值: `true`): 启用基于长度的分组以减少填充开销
- `EMB_PROVIDER_LENGTH_GROUP_TOLERANCE` (默认值: `0.2`): 长度分组容差 (0.1-0.5)

#### 资源配置
- `EMB_PROVIDER_MEMORY_LIMIT` (默认值: `2GB`): 服务的内存限制
- `EMB_PROVIDER_DEVICE` (默认值: `auto`): 运行模型的设备 (auto, cpu, cuda, mps)

#### API 配置
- `EMB_PROVIDER_HOST` (默认值: `localhost`): 绑定 API 服务器的主机
- `EMB_PROVIDER_PORT` (默认值: `9000`): 绑定 API 服务器的端口 (1-65535)

#### 日志配置
- `EMB_PROVIDER_LOG_LEVEL` (默认值: `INFO`): 日志级别 (DEBUG, INFO, WARNING, ERROR)
- `EMB_PROVIDER_LOG_TO_FILE` (默认值: `true`): 启用文件日志记录
- `EMB_PROVIDER_LOG_DIR` (默认值: `logs`): 存储日志文件的目录
- `EMB_PROVIDER_LOG_FILE_MAX_SIZE` (默认值: `10`): 单个日志文件的最大大小（MB）(1-100)
- `EMB_PROVIDER_LOG_RETENTION_DAYS` (默认值: `7`): 保留日志文件的天数 (1-30)
- `EMB_PROVIDER_LOG_CLEANUP_INTERVAL_HOURS` (默认值: `1`): 日志清理检查的间隔（小时）(1-24)
- `EMB_PROVIDER_LOG_MAX_DIR_SIZE_MB` (默认值: `50`): 清理前日志目录的最大总大小（MB）(1-1000)
- `EMB_PROVIDER_LOG_CLEANUP_TARGET_SIZE_MB` (默认值: `20`): 清理后的目标大小（MB）(1-100)
- `EMB_PROVIDER_LOG_CLEANUP_RETENTION_DAYS` (默认值: `7,3,1`): 清理时尝试的保留天数

### 配置文件

您可以基于项目根目录中提供的 `.env.example` 文件创建 `.env` 文件来设置配置值：

```bash
# 复制示例文件
copy .env.example .env

# 使用您首选的设置编辑 .env
```

## 模型加载机制

该服务支持多种加载嵌入模型的方法：

### 1. 本地模型加载
服务首先检查 `EMB_PROVIDER_MODEL_PATH` 指定的路径是否存在模型。如果找到，将直接从本地目录加载模型。

### 2. 从 Hugging Face Hub 下载
如果本地模型不可用，服务会自动从 Hugging Face Hub 使用 `EMB_PROVIDER_MODEL_NAME` 值下载模型。

### 3. 直接 Transformers 加载
如果 `EMB_PROVIDER_LOAD_FROM_TRANSFORMERS` 设置为 `true`，模型将直接从 transformers 库加载，而不使用本地缓存副本。

## 支持的模型

该服务支持任何可以使用 Hugging Face Transformers 库加载的基于 Transformer 的模型。一些流行的模型包括：

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (默认)
- `sentence-transformers/all-MiniLM-L12-v2`
- `sentence-transformers/all-mpnet-base-v2`
- `intfloat/e5-large-v2`
- `BAAI/bge-large-en-v1.5`

## API 使用

该服务提供与 OpenAI 兼容的嵌入 API。

### 创建嵌入

```bash
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  }'
```

### 使用 Python OpenAI 客户端

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",  # 不需要真实 API 密钥
    base_url="http://localhost:9000/v1"
)

response = client.embeddings.create(
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    input="Hello, world!"
)
```

### 请求参数

- `input`: 必需。要生成嵌入的字符串或字符串数组。
- `model`: 必需。模型名称（必须与您配置的模型匹配）。
- `encoding_format`: 可选。嵌入的格式（默认为 "float"）。
- `user`: 可选。用户标识符。

### 响应格式

响应遵循 OpenAI 嵌入 API 格式：

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    }
  ],
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3
  }
}
```

## 性能优化

该服务包含几个性能优化功能：

### 动态批处理
启用后，服务将传入请求分组为批次以更高效地处理它们。`max_wait_time_ms` 设置控制服务等待收集更多请求后再处理批次的时间。

### 基于长度的分组
启用后，服务将相似长度的输入分组在一起，以减少填充开销并提高计算效率。

### GPU 内存优化
当 CUDA 可用时，服务会根据可用 GPU 内存自动优化批处理大小。

## 运行服务

### 使用 uv (推荐)

```bash
# 安装依赖
uv sync

# 运行服务
uv run python -m emb_model_provider.main
```

### 使用 pip

```bash
# 安装依赖
pip install -e .

# 运行服务
uvicorn emb_model_provider.main:app --host localhost --port 9000
```

### Docker

```bash
# 构建镜像
docker build -t emb-model-provider .

# 运行容器
docker run -p 9000:9000 -v /path/to/models:/models emb-model-provider
```

## 监控和健康检查

### 健康检查端点

服务提供健康检查端点：
```
GET http://localhost:9000/health
```

### 性能指标

您可以在此处检索性能指标：
```
GET http://localhost:9000/v1/performance
```

并在此处重置指标：
```
POST http://localhost:9000/v1/performance/reset
```

## 日志

该服务使用 JSON 格式的结构化日志记录。日志同时写入控制台和文件（如果启用）。每个日志条目都包含一个唯一的请求 ID，用于跨请求跟踪。

## 错误处理

该服务遵循 OpenAI 的 API 错误格式。常见错误包括：

- `context_length_exceeded`: 输入超出模型的最大上下文长度
- `batch_size_exceeded`: 请求包含的输入超过最大批处理大小
- `model_not_found`: 请求的模型不可用
- `invalid_request_error`: 无效的请求参数
- `internal_server_error`: 内部服务器错误

## 最佳实践

1. **选择合适的设备**: 如果您有兼容的 GPU，请使用 `cuda`，否则使用 `cpu` 或 `mps`（Mac）。
2. **优化批处理大小**: 根据可用内存和预期的请求模式调整 `max_batch_size`。
3. **监控资源使用情况**: 根据系统资源设置适当的 `memory_limit` 值。
4. **使用适当的模型**: 选择符合您准确性和性能要求的模型。
5. **启用动态批处理**: 对于具有可变请求模式的服务，启用动态批处理可以显著提高吞吐量。
6. **适当配置日志**: 根据调试和合规性需求调整日志级别和保留策略。

## 故障排除

### 常见问题

1. **模型未找到**: 确保您的模型名称正确，如果未使用本地路径，则模型可以通过 Hugging Face Hub 访问。

2. **内存问题**: 如果遇到内存错误，请减少 `max_batch_size` 或使用不同的设备。

3. **性能问题**: 为可变输入长度启用 `enable_length_grouping` 和 `enable_dynamic_batching` 以获得更好的性能。

4. **连接问题**: 在容器化环境中，验证主机和端口配置是否符合您的期望。

### 验证安装

1. 检查所有依赖项是否已安装：
   ```bash
   uv run python -c "import torch; import transformers; import fastapi; print('Dependencies OK')"
   ```

2. 测试服务：
   ```bash
   curl -X POST http://localhost:9000/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input": "test", "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}'
   ```
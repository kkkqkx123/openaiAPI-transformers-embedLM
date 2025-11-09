# Embedding Model Provider API

一个基于 FastAPI 的 OpenAI 兼容嵌入模型 API 服务，使用 `all-MiniLM-L12-v2` 模型提供高质量的文本嵌入向量。

## 功能特性

- 🚀 **高性能**: 基于 FastAPI 和 PyTorch 的高效推理
- 🔄 **OpenAI 兼容**: 完全兼容 OpenAI embeddings API 格式
- 📦 **自动模型管理**: 支持本地模型加载和从 Hugging Face Hub 自动下载
- 🛡️ **错误处理**: 完善的错误处理机制，遵循 OpenAI API 错误格式
- 📊 **结构化日志**: JSON 格式的结构化日志输出
- ⚙️ **灵活配置**: 支持环境变量和配置文件的灵活配置
- 🧪 **全面测试**: 包含单元测试、集成测试、性能测试和兼容性测试

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- uv 包管理器（推荐）或 pip
- 至少 2GB 可用内存

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/example/emb-model-provider.git
cd emb-model-provider
```

2. 使用 uv 安装依赖（推荐）：
```bash
uv sync
```

或使用 pip 安装：
```bash
pip install -e .
```

3. 模型会自动下载到默认路径 `D:\models\all-MiniLM-L12-v2`（Windows）或 `/models/all-MiniLM-L12-v2`（Linux/Mac）。

### 运行服务

1. 启动服务：
```bash
uv run python -m emb_model_provider.main
```

或使用 uvicorn：
```bash
uvicorn emb_model_provider.main:app --host localhost --port 9000
```

2. 服务将在 `http://localhost:9000` 启动。

3. 访问 API 文档：
   - Swagger UI: `http://localhost:9000/docs`
   - ReDoc: `http://localhost:9000/redoc`

## 使用示例

### 使用 curl

```bash
# 获取可用模型
curl http://localhost:9000/v1/models

# 创建嵌入向量
curl -X POST "http://localhost:9000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "all-MiniLM-L12-v2"
  }'
```

### 使用 Python requests

```python
import requests

# 获取可用模型
models_response = requests.get("http://localhost:9000/v1/models")
models = models_response.json()
print(models)

# 创建嵌入向量
embeddings_response = requests.post(
    "http://localhost:9000/v1/embeddings",
    json={
        "input": "Hello, world!",
        "model": "all-MiniLM-L12-v2"
    }
)
embeddings = embeddings_response.json()
print(embeddings)
```

### 使用 OpenAI Python 客户端

```python
from openai import OpenAI

# 配置客户端使用本地 API
client = OpenAI(
    api_key="dummy-key",  # 不需要真实密钥
    base_url="http://localhost:9000/v1"
)

# 创建嵌入向量
response = client.embeddings.create(
    model="all-MiniLM-L12-v2",
    input="Hello, world!"
)

print(response.data[0].embedding)
```

### 批量处理

```python
import requests

# 批量创建嵌入向量
response = requests.post(
    "http://localhost:9000/v1/embeddings",
    json={
        "input": [
            "First sentence",
            "Second sentence",
            "Third sentence"
        ],
        "model": "all-MiniLM-L12-v2"
    }
)

data = response.json()
for i, embedding_data in enumerate(data["data"]):
    print(f"Sentence {i}: {embedding_data['embedding'][:5]}...")  # 只显示前5个维度
```

## 配置

### 环境变量

可以通过环境变量配置服务：

```bash
# 模型配置
export EMB_PROVIDER_MODEL_PATH="/path/to/model"
export EMB_PROVIDER_MODEL_NAME="all-MiniLM-L12-v2"

# 处理配置
export EMB_PROVIDER_MAX_BATCH_SIZE=32
export EMB_PROVIDER_MAX_CONTEXT_LENGTH=512
export EMB_PROVIDER_EMBEDDING_DIMENSION=384

# 资源配置
export EMB_PROVIDER_MEMORY_LIMIT="2GB"
export EMB_PROVIDER_DEVICE="auto"  # auto, cpu, cuda

# API 配置
export EMB_PROVIDER_HOST="localhost"
export EMB_PROVIDER_PORT=9000

# 日志配置
export EMB_PROVIDER_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

### 配置文件

创建 `.env` 文件：

```env
# 模型配置
EMB_PROVIDER_MODEL_PATH=D:\models\all-MiniLM-L12-v2
EMB_PROVIDER_MODEL_NAME=all-MiniLM-L12-v2

# 处理配置
EMB_PROVIDER_MAX_BATCH_SIZE=32
EMB_PROVIDER_MAX_CONTEXT_LENGTH=512
EMB_PROVIDER_EMBEDDING_DIMENSION=384

# 资源配置
EMB_PROVIDER_MEMORY_LIMIT=2GB
EMB_PROVIDER_DEVICE=auto

# API 配置
EMB_PROVIDER_HOST=localhost
EMB_PROVIDER_PORT=9000

# 日志配置
EMB_PROVIDER_LOG_LEVEL=INFO
```

## API 参考

### 端点

#### `GET /v1/models`

列出可用的模型。

**响应示例：**
```json
{
  "object": "list",
  "data": [
    {
      "id": "all-MiniLM-L12-v2",
      "object": "model",
      "created": 1677610602,
      "owned_by": "organization-owner"
    }
  ]
}
```

#### `POST /v1/embeddings`

为给定的输入文本创建嵌入向量。

**请求体：**
```json
{
  "input": "Your text here",
  "model": "all-MiniLM-L12-v2",
  "encoding_format": "float",
  "user": "optional-user-id"
}
```

**响应示例：**
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
  "model": "all-MiniLM-L12-v2",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

#### `GET /health`

健康检查端点。

**响应示例：**
```json
{
  "status": "healthy"
}
```

### 错误处理

API 遵循 OpenAI 的错误响应格式：

```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "param": "parameter_name",
    "code": "error_code"
  }
}
```

常见错误类型：
- `invalid_request_error`: 请求无效（400）
- `context_length_exceeded`: 上下文长度超限（429）
- `batch_size_exceeded`: 批处理大小超限（429）
- `model_not_found`: 模型未找到（404）
- `internal_server_error`: 内部服务器错误（500）

## 性能优化

### 批处理

使用批量请求可以提高吞吐量：

```python
# 不推荐：多个单独请求
for text in texts:
    response = requests.post("/v1/embeddings", json={"input": text, "model": "all-MiniLM-L12-v2"})

# 推荐：单个批量请求
response = requests.post("/v1/embeddings", json={"input": texts, "model": "all-MiniLM-L12-v2"})
```

### 配置调优

根据硬件资源调整配置：

- **CPU 环境**: 设置 `EMB_PROVIDER_DEVICE=cpu`，减小 `EMB_PROVIDER_MAX_BATCH_SIZE`
- **GPU 环境**: 设置 `EMB_PROVIDER_DEVICE=cuda`，增大 `EMB_PROVIDER_MAX_BATCH_SIZE`
- **内存受限**: 减小 `EMB_PROVIDER_MAX_BATCH_SIZE` 和 `EMB_PROVIDER_MEMORY_LIMIT`
直接使用auto也可以

## 开发

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest tests/test_e2e.py
uv run pytest tests/test_performance.py
uv run pytest tests/test_openai_compatibility.py

# 运行测试并生成覆盖率报告
uv run pytest --cov=emb_model_provider
```

### 代码格式化

```bash
# 格式化代码
uv run black .
uv run isort .

# 检查代码质量
uv run flake8 emb_model_provider
uv run mypy emb_model_provider
```

### 项目结构

```
emb_model_provider/
├── __init__.py
├── main.py              # FastAPI 应用入口
├── api/                 # API 路由
│   ├── __init__.py
│   ├── embeddings.py    # 嵌入端点
│   ├── models.py        # 模型端点
│   ├── exceptions.py    # 异常定义
│   └── middleware.py    # 中间件
├── core/                # 核心业务逻辑
│   ├── __init__.py
│   ├── config.py        # 配置管理
│   ├── logging.py       # 日志配置
│   └── model_manager.py # 模型管理
└── services/            # 服务层
    ├── __init__.py
    └── embedding_service.py # 嵌入服务
```

## Docker 部署

### 构建镜像

```bash
docker build -t emb-model-provider .
```

### 运行容器

```bash
docker run -p 9000:9000 \
  -v /path/to/models:/models \
  -e EMB_PROVIDER_MODEL_PATH=/models/all-MiniLM-L12-v2 \
  emb-model-provider
```

### 使用 Docker Compose

```yaml
version: '3.8'
services:
  emb-model-provider:
    build: .
    ports:
      - "9000:9000"
    volumes:
      - ./models:/models
    environment:
      - EMB_PROVIDER_MODEL_PATH=/models/all-MiniLM-L12-v2
      - EMB_PROVIDER_LOG_LEVEL=INFO
```

## 故障排除

### 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 确保有足够的磁盘空间
   - 手动下载模型到指定路径

2. **内存不足**
   - 减小 `EMB_PROVIDER_MAX_BATCH_SIZE`
   - 设置 `EMB_PROVIDER_DEVICE=cpu`
   - 增加 `EMB_PROVIDER_MEMORY_LIMIT`

3. **响应时间慢**
   - 使用 GPU 加速：`EMB_PROVIDER_DEVICE=cuda`
   - 增加批处理大小：`EMB_PROVIDER_MAX_BATCH_SIZE`
   - 检查系统资源使用情况

### 日志分析

启用 DEBUG 级别日志获取详细信息：

```bash
export EMB_PROVIDER_LOG_LEVEL=DEBUG
```

日志以 JSON 格式输出，包含：
- 请求 ID 跟踪
- 性能指标
- 错误详情
- 模型加载事件

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 支持

如有问题或建议，请：

1. 查看 [FAQ](#故障排除)
2. 搜索 [Issues](https://github.com/example/emb-model-provider/issues)
3. 创建新的 Issue

## 更新日志

### v0.1.0
- 初始版本发布
- OpenAI 兼容的 embeddings API
- 支持单个和批量文本嵌入
- 完整的错误处理和日志记录
- 全面的测试套件
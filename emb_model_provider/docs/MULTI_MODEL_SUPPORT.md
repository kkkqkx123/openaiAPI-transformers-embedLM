# 多模型支持方案

## 概述

本文档描述了 Embedding Model Provider API 的多模型支持功能，允许通过环境变量配置多个模型及其别名映射。

## 当前实现状态分析

### 已支持的功能
1. **基础模型配置**：在 [`emb_model_provider/core/config.py`](emb_model_provider/core/config.py) 中已有基础模型配置
2. **单模型别名**：支持通过 `model_aliases` 配置单个模型的别名

### 需要改进的功能
1. **多模型配置**：支持配置多个不同的模型
2. **JSON格式映射**：使用JSON格式配置模型别名映射
3. **API模型验证**：修改嵌入服务以支持多模型验证
4. **模型列表端点**：更新模型列表API以返回所有可用模型

## 新的配置格式

采用JSON格式来配置模型映射，这样更加清晰和易于维护：

```env
# 多模型JSON配置格式（简单格式）
EMB_PROVIDER_MODEL_MAPPING={"mini": "sentence-transformers/all-MiniLM-L12-v2", "mpnet": "sentence-transformers/all-mpnet-base-v2"}

# 多模型JSON配置格式（包含路径信息）
EMB_PROVIDER_MODEL_MAPPING={
  "mini": {
    "name": "sentence-transformers/all-MiniLM-L12-v2",
    "path": "D:\\models\\all-MiniLM-L12-v2"
  },
  "multilingual": {
    "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "path": "D:\\models\\multilingual-MiniLM-L12-v2"
  }
}
```

## 技术实现方案

### 1. 修改配置类

在 [`emb_model_provider/core/config.py`](emb_model_provider/core/config.py) 中添加JSON解析功能：

```python
import json

class Config(BaseSettings):
    # ... 现有配置 ...
    
    # 多模型映射配置
    model_mapping: str = Field(
        default="{}",
        description="JSON string mapping model aliases to actual model names"
    )
    
    def get_model_mapping(self) -> dict:
        """
        Parse the model mapping JSON string into a dictionary.
        
        Returns:
            dict: Dictionary mapping aliases to actual model names
        """
        if not self.model_mapping or self.model_mapping == "{}":
            return {}
            
        try:
            return json.loads(self.model_mapping)
        except json.JSONDecodeError:
            logger.warning("Failed to parse model mapping JSON")
            return {}
```

### 2. 修改嵌入服务

修改 [`emb_model_provider/services/embedding_service.py`](emb_model_provider/services/embedding_service.py) 以支持多模型：

```python
def validate_request(self, request: EmbeddingRequest) -> None:
    """
    验证请求参数（支持多模型）
    """
    # 获取模型映射
    model_mapping = self.config.get_model_mapping()
    
    # 解析模型名称（支持别名）
    actual_model_name = model_mapping.get(request.model, request.model)
    
    # 检查模型是否可用
    available_models = list(model_mapping.values()) + [self.config.model_name]
    if actual_model_name not in available_models:
        raise ModelNotFoundError(model_name=request.model)
```

### 3. 更新模型列表端点

更新 [`emb_model_provider/api/models.py`](emb_model_provider/api/models.py) 以返回所有可用模型：

```python
@router.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    列出所有可用模型的API端点（包括别名）
    """
    from emb_model_provider.core.config import config
    
    model_info_list = []
    
    # 添加主模型
    if config.model_name:
        model_info_list.append(ModelInfo(
            id=config.model_name,
            owned_by="organization-owner"
        ))
    
    # 添加映射的模型和别名
    model_mapping = config.get_model_mapping()
    for alias, actual_model in model_mapping.items():
        # 添加别名
        model_info_list.append(ModelInfo(
            id=alias,
            owned_by="organization-owner"
        ))
        # 添加实际模型（如果尚未添加）
        if actual_model != config.model_name:
            model_info_list.append(ModelInfo(
                id=actual_model,
                owned_by="organization-owner"
            ))
    
    return ModelsResponse(data=model_info_list)
```

## 配置示例

### 基础配置
```env
# 主模型配置
EMB_PROVIDER_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMB_PROVIDER_MODEL_PATH=D:\models\multilingual-MiniLM-L12-v2

# 多模型映射配置
EMB_PROVIDER_MODEL_MAPPING={
  "mini": "sentence-transformers/all-MiniLM-L12-v2",
  "mpnet": "sentence-transformers/all-mpnet-base-v2"
}
```

### 完整配置示例
```env
# 模型配置
EMB_PROVIDER_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMB_PROVIDER_MODEL_PATH=D:\models\multilingual-MiniLM-L12-v2

# 多模型映射配置
EMB_PROVIDER_MODEL_MAPPING={
  "mini": "sentence-transformers/all-MiniLM-L12-v2",
  "mpnet": "sentence-transformers/all-mpnet-base-v2",
  "large": "sentence-transformers/all-mpnet-base-v2"
}

# 处理配置
EMB_PROVIDER_MAX_BATCH_SIZE=32
EMB_PROVIDER_MAX_CONTEXT_LENGTH=512
```

## 测试用例

### 单元测试
在 `tests/test_config.py` 中添加测试用例：

```python
def test_get_model_mapping(self):
    """测试获取模型映射"""
    config = Config(model_mapping='{"mini": "sentence-transformers/all-MiniLM-L12-v2"}')
    mapping = config.get_model_mapping()
    assert "mini" in mapping
    assert mapping["mini"] == "sentence-transformers/all-MiniLM-L12-v2"

def test_get_model_mapping_empty(self):
    """测试获取空模型映射"""
    config = Config(model_mapping="")
    mapping = config.get_model_mapping()
    assert mapping == {}

def test_get_model_mapping_invalid_json(self):
    """测试获取无效JSON的模型映射"""
    config = Config(model_mapping="{invalid-json}")
    mapping = config.get_model_mapping()
    assert mapping == {}
```

### API 测试
在 `tests/test_api_endpoints.py` 中添加测试用例：

```python
def test_models_endpoint_with_multi_models(client_with_multi_models):
    """测试模型端点返回多模型信息"""
    response = client_with_multi_models.get("/v1/models")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "data" in data
    
    # 检查是否包含所有模型和别名
    model_ids = [model["id"] for model in data["data"]]
    assert "mini" in model_ids
    assert "sentence-transformers/all-MiniLM-L12-v2" in model_ids

def test_embeddings_endpoint_with_multi_models(client_with_multi_models):
    """测试使用不同模型调用嵌入端点"""
    # 测试使用别名
    request_data = {
        "input": "Hello world",
        "model": "mini"
    }
    
    response = client_with_multi_models.post("/v1/embeddings", json=request_data)
    assert response.status_code == 200
    
    # 测试使用实际模型名
    request_data["model"] = "sentence-transformers/all-MiniLM-L12-v2"
    response = client_with_multi_models.post("/v1/embeddings", json=request_data)
    assert response.status_code == 200
```

## 部署说明

### 1. 更新 `.env` 文件

在 `.env` 文件中添加多模型映射配置：

```env
EMB_PROVIDER_MODEL_MAPPING={
  "mini": "sentence-transformers/all-MiniLM-L12-v2",
  "mpnet": "sentence-transformers/all-mpnet-base-v2"
}
```

### 2. 启动服务

使用以下命令启动服务：

```bash
uv run python -m emb_model_provider.main
```

### 3. 验证功能

使用以下命令验证多模型功能：

```bash
# 查看模型列表（应该包含所有模型和别名）
curl http://localhost:9000/v1/models

# 使用别名调用嵌入API
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "mini"}'

# 使用实际模型名调用嵌入API
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "sentence-transformers/all-MiniLM-L12-v2"}'
```

## 兼容性考虑

1. **向后兼容**：现有的单模型配置和别名配置仍然正常工作
2. **错误处理**：无效的JSON配置会记录警告并返回空映射
3. **响应格式**：API响应格式保持不变，符合OpenAI兼容性标准

## 性能影响

- **内存开销**：模型映射存储在内存中，开销极小
- **处理延迟**：JSON解析几乎不增加处理延迟
- **网络开销**：响应格式保持不变，无额外网络开销

## 安全考虑

- **JSON验证**：模型映射经过严格的JSON格式验证
- **注入防护**：配置值经过适当的转义和验证
- **错误信息**：不泄露内部模型路径信息
# 模型别名支持方案

## 概述

本方案为 Embedding Model Provider API 提供模型别名支持功能，允许通过 `.env` 配置文件为模型设置别名，并通过别名调用相应的模型。

## 注意事项

此文档描述的是旧的模型别名配置方式。我们已经引入了新的多模型支持功能，请参考 [`MULTI_MODEL_SUPPORT.md`](MULTI_MODEL_SUPPORT.md) 文档了解新的配置方式。

## 当前实现状态分析

### 已支持的功能
1. **配置系统**：在 [`emb_model_provider/core/config.py`](emb_model_provider/core/config.py:44-48) 中已有 `model_aliases` 配置项
2. **别名解析**：配置类中已有 [`get_model_aliases()`](emb_model_provider/core/config.py:360-380) 方法来解析别名映射
3. **多模型支持**：新增了 `model_mapping` 配置项支持JSON格式的多模型配置

### 需要改进的功能
1. **API 模型验证**：需要修改 [`emb_model_provider/services/embedding_service.py`](emb_model_provider/services/embedding_service.py:53-54) 中的 `validate_request()` 方法以支持别名解析
2. **模型列表端点**：需要更新 [`emb_model_provider/api/models.py`](emb_model_provider/api/models.py:23-50) 中的 `list_models()` 函数以返回所有可用模型（包括别名）
3. **测试用例**：需要添加相应的测试用例

## 配置格式

在 `.env` 文件中使用以下格式配置模型别名：

```env
# 模型别名配置（逗号分隔的 alias:actual_model_name 对）
EMB_PROVIDER_MODEL_ALIASES=mini:sentence-transformers/all-MiniLM-L12-v2,mpnet:sentence-transformers/all-mpnet-base-v2,multilingual:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 配置示例

```env
# 模型配置
EMB_PROVIDER_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMB_PROVIDER_MODEL_PATH=D:\models\multilingual-MiniLM-L12-v2

# 模型别名配置
EMB_PROVIDER_MODEL_ALIASES=mini:sentence-transformers/all-MiniLM-L12-v2,mpnet:sentence-transformers/all-mpnet-base-v2,multilingual:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## 技术实现方案

### 1. 修改模型验证逻辑

在 [`emb_model_provider/services/embedding_service.py`](emb_model_provider/services/embedding_service.py:53-54) 中修改 `validate_request()` 方法：

```python
def validate_request(self, request: EmbeddingRequest) -> None:
    """
    验证请求参数（支持别名解析）
    """
    # 获取模型别名映射
    model_aliases = self.config.get_model_aliases()
    
    # 检查模型名称是否匹配（支持别名解析）
    actual_model_name = model_aliases.get(request.model, request.model)
    if actual_model_name != self.config.model_name:
        raise ModelNotFoundError(model_name=request.model)
    
    # ... 其他验证逻辑保持不变
```

### 2. 更新模型列表端点

在 [`emb_model_provider/api/models.py`](emb_model_provider/api/models.py:23-50) 中修改 `list_models()` 函数：

```python
@router.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    列出可用模型的API端点（包括别名）
    对应需求: UB-1.4
    """
    # 延迟导入，避免循环导入
    from emb_model_provider.core.config import config
    from emb_model_provider.api.exceptions import ModelNotFoundError
    
    # 验证模型是否存在
    model_path = config.model_path
    import os
    if not os.path.exists(model_path):
        from emb_model_provider.core.model_manager import ModelManager
        try:
            # 尝试初始化模型管理器，这会触发模型下载
            model_manager = ModelManager(model_path)
        except Exception:
            # 如果模型不存在且无法下载，抛出异常
            raise ModelNotFoundError(config.model_name)
    
    # 获取所有模型信息（包括别名）
    model_info_list = []
    
    # 添加实际模型
    model_info = ModelInfo(
        id=config.model_name,
        owned_by="organization-owner"
    )
    model_info_list.append(model_info)
    
    # 添加别名模型
    model_aliases = config.get_model_aliases()
    for alias_name, actual_model_name in model_aliases.items():
        if actual_model_name == config.model_name:
            alias_info = ModelInfo(
                id=alias_name,
                owned_by="organization-owner"
            )
            model_info_list.append(alias_info)
    
    response = ModelsResponse(data=model_info_list)
    return response
```

### 3. 确保API响应一致性

在嵌入处理过程中，确保响应中的 `model` 字段使用原始请求的模型名称（可能是别名），而不是解析后的实际模型名称。

## 测试用例

### 单元测试

在 `tests/test_embedding_service.py` 中添加测试用例：

```python
def test_validate_request_with_alias(self, config_with_aliases):
    """测试使用别名验证请求"""
    service = EmbeddingService(config_with_aliases)
    request = EmbeddingRequest(
        input="Hello world",
        model="mini"  # 使用别名
    )
    
    # 应该不会抛出异常
    service.validate_request(request)

def test_validate_request_with_invalid_alias(self, config_with_aliases):
    """测试使用无效别名验证请求"""
    service = EmbeddingService(config_with_aliases)
    request = EmbeddingRequest(
        input="Hello world",
        model="invalid-alias"  # 无效别名
    )
    
    # 应该抛出 ModelNotFoundError
    with pytest.raises(ModelNotFoundError):
        service.validate_request(request)
```

### API 测试

在 `tests/test_api_endpoints.py` 中添加测试用例：

```python
def test_models_endpoint_with_aliases(client_with_aliases):
    """测试模型端点返回别名信息"""
    response = client_with_aliases.get("/v1/models")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "data" in data
    assert len(data["data"]) >= 2  # 至少有一个实际模型和一个别名
    
    # 检查是否包含别名
    model_ids = [model["id"] for model in data["data"]]
    assert "mini" in model_ids
    assert "sentence-transformers/all-MiniLM-L12-v2" in model_ids

def test_embeddings_endpoint_with_alias(client_with_aliases):
    """测试使用别名调用嵌入端点"""
    request_data = {
        "input": "Hello world",
        "model": "mini"  # 使用别名
    }
    
    response = client_with_aliases.post("/v1/embeddings", json=request_data)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "data" in data
    assert "model" in data
    assert data["model"] == "mini"  # 响应中应该返回使用的别名
```

## 部署说明

### 1. 更新 `.env` 文件

在 `.env` 文件中添加模型别名配置：

```env
EMB_PROVIDER_MODEL_ALIASES=mini:sentence-transformers/all-MiniLM-L12-v2,mpnet:sentence-transformers/all-mpnet-base-v2
```

### 2. 启动服务

使用以下命令启动服务：

```bash
uv run python -m emb_model_provider.main
```

### 3. 验证功能

使用以下命令验证别名功能：

```bash
# 查看模型列表（应该包含别名）
curl http://localhost:9000/v1/models

# 使用别名调用嵌入API
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "mini"}'
```

## 兼容性考虑

1. **向后兼容**：现有的使用完整模型名称的API调用仍然正常工作
2. **错误处理**：无效的别名会返回标准的 `ModelNotFoundError`
3. **响应格式**：API响应格式保持不变，符合OpenAI兼容性标准

## 性能影响

- **内存开销**：别名映射存储在内存中，开销极小
- **处理延迟**：别名解析几乎不增加处理延迟
- **网络开销**：响应格式保持不变，无额外网络开销

## 安全考虑

- **别名验证**：确保别名只能映射到已配置的实际模型
- **注入防护**：别名配置经过严格的格式验证
- **错误信息**：不泄露内部模型路径信息
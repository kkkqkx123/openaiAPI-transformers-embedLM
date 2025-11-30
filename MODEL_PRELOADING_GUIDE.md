# 模型预加载与白名单功能说明文档

## 概述

本项目支持模型预加载和白名单功能，允许管理员控制哪些模型可以在服务中使用。通过配置，您可以：

1. 预先加载指定的模型，提高响应速度
2. 禁用动态模型加载，只允许预加载的模型被使用
3. 创建模型白名单，防止加载不需要的模型

## 配置说明

### 1. 预加载模型配置

使用 `EMB_PROVIDER_PRELOAD_MODELS` 环境变量指定需要在服务启动时预加载的模型：

```bash
EMB_PROVIDER_PRELOAD_MODELS="all-MiniLM-L12-v2,sentence-transformers/all-MiniLM-L6-v2"
```

- 多个模型名称用逗号分隔
- 模型名称应与API请求中的模型名称一致
- 预加载的模型在服务启动时即被加载到内存中

### 2. 动态加载控制

使用 `EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING` 控制是否允许动态加载模型：

```bash
EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING=false
```

- 设置为 `false` 时，只允许预加载列表中的模型被使用
- 设置为 `true` 时，允许动态加载任何模型（默认行为）
- 当禁用动态加载时，请求非预加载模型将返回错误

### 3. 模型映射配置

使用 `EMB_PROVIDER_MODEL_MAPPING` 配置多模型映射：

```json
{
  "all-MiniLM-L12-v2": {
    "name": "sentence-transformers/all-MiniLM-L12-v2",
    "path": "",
    "source": "transformers"
  },
  "all-MiniLM-L6-v2": {
    "name": "sentence-transformers/all-MiniLM-L6-v2",
    "path": "",
    "source": "transformers"
  }
}
```

## 模型加载时机

### 1. 服务启动时
- 如果配置了 `EMB_PROVIDER_PRELOAD_MODELS`，指定的模型会在服务启动时被加载
- 预加载过程在 `preload_models()` 函数中执行

### 2. 请求处理时
- 当 `EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING=true` 时，首次请求某个模型时会动态加载
- 当 `EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING=false` 时，只有预加载列表中的模型可以被加载

## 实现机制

### 1. 配置检查

在 `config.py` 中实现了以下关键方法：

```python
def get_preload_models(self) -> List[str]:
    """获取预加载模型列表"""
    pass

def is_model_preloaded(self, alias: str) -> bool:
    """检查模型是否在预加载列表中"""
    pass
```

### 2. 模型管理

在 `model_manager.py` 中：

- `get_model_manager()` 函数检查模型是否可以加载
- 如果动态加载被禁用且模型未预加载，会抛出错误

### 3. API 请求处理

在 `embedding_service.py` 中：

- `_get_model_manager()` 方法检查模型是否允许使用
- `validate_request()` 方法验证模型是否存在且允许使用

## 使用示例

### 示例 1: 仅允许预加载模型

```bash
# .env 配置
EMB_PROVIDER_PRELOAD_MODELS="all-MiniLM-L12-v2,sentence-transformers/all-MiniLM-L6-v2"
EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING=false
EMB_PROVIDER_MODEL_MAPPING="{\"all-MiniLM-L12-v2\": {\"name\": \"sentence-transformers/all-MiniLM-L12-v2\", \"path\": \"\", \"source\": \"transformers\"}, \"all-MiniLM-L6-v2\": {\"name\": \"sentence-transformers/all-MiniLM-L6-v2\", \"path\": \"\", \"source\": \"transformers\"}}"
```

在这种配置下：
- 服务启动时会预加载 `all-MiniLM-L12-v2` 和 `all-MiniLM-L6-v2` 两个模型
- 只有这两个模型可以被使用
- 请求其他模型（如 `text-embedding-ada-002`）会返回错误

### 示例 2: 允许动态加载（默认行为）

```bash
# .env 配置
EMB_PROVIDER_PRELOAD_MODELS=""
EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING=true
```

在这种配置下：
- 没有预加载任何模型
- 所有模型都可以动态加载
- 首次请求某个模型时会加载该模型

## 错误处理

当请求的模型不在预加载列表中且动态加载被禁用时，服务会返回 `ModelNotFoundError` 错误：

```json
{
  "error": {
    "message": "Model 'unknown-model' not found",
    "type": "model_not_found",
    "param": null,
    "code": "model_not_found"
  }
}
```

## 最佳实践

1. **生产环境**: 建议禁用动态加载，只预加载需要的模型，以控制资源使用和安全
2. **开发环境**: 可以启用动态加载，方便测试不同模型
3. **资源管理**: 根据可用内存合理配置预加载模型数量
4. **监控**: 使用性能监控功能跟踪模型加载和使用情况

## 故障排除

### 问题: 服务启动时预加载模型失败

**原因**: 模型名称错误、网络问题或模型文件损坏
**解决方案**: 检查模型名称和网络连接，确保模型可以正常下载

### 问题: 请求预加载的模型返回错误

**原因**: 模型名称不匹配
**解决方案**: 确保请求中的模型名称与预加载配置中的名称完全一致

### 问题: 内存不足

**原因**: 预加载了过多模型
**解决方案**: 减少预加载模型数量或增加系统内存
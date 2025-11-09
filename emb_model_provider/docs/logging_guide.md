# 日志系统使用指南

## 概述

本项目提供了完整的日志系统，支持控制台输出和文件输出，具有自动轮转、压缩和清理功能。日志采用结构化JSON格式，便于分析和监控。

## 功能特性

### 1. 多输出支持
- **控制台输出**：实时显示日志信息，适用于开发和调试
- **文件输出**：持久化存储日志，适用于生产环境和问题排查

### 2. 按级别分离
日志文件按级别分别存储：
- `app-YYYY-MM-DD-debug.log` - 调试信息
- `app-YYYY-MM-DD-info.log` - 一般信息
- `app-YYYY-MM-DD-warning.log` - 警告信息
- `app-YYYY-MM-DD-error.log` - 错误信息

### 3. 自动轮转和压缩
- 当单个日志文件达到配置的最大大小时，自动创建新文件
- 旧日志文件自动压缩为 `.gz` 格式以节省空间
- 保留指定数量的压缩备份文件

### 4. 智能清理策略
- 定期检查日志目录总大小
- 当超过限制时，按策略逐步清理旧日志
- 支持多级清理策略（7天→3天→1天）

## 配置参数

### 基础配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `EMB_PROVIDER_LOG_LEVEL` | INFO | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| `EMB_PROVIDER_LOG_TO_FILE` | true | 是否启用文件日志 |
| `EMB_PROVIDER_LOG_DIR` | logs | 日志文件目录 |

### 文件管理配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `EMB_PROVIDER_LOG_FILE_MAX_SIZE` | 10 | 单个日志文件最大大小（MB） |
| `EMB_PROVIDER_LOG_RETENTION_DAYS` | 7 | 默认日志保留天数 |
| `EMB_PROVIDER_LOG_CLEANUP_INTERVAL_HOURS` | 1 | 清理检查间隔（小时） |
| `EMB_PROVIDER_LOG_MAX_DIR_SIZE_MB` | 50 | 日志目录最大大小（MB），超过此值触发清理 |
| `EMB_PROVIDER_LOG_CLEANUP_TARGET_SIZE_MB` | 20 | 清理后目标大小（MB），应小于最大大小 |
| `EMB_PROVIDER_LOG_CLEANUP_RETENTION_DAYS` | 7,3,1 | 清理策略保留天数（逗号分隔） |

## 使用示例

### 开发环境配置

```bash
# 启用详细日志和较小的文件大小
EMB_PROVIDER_LOG_LEVEL=DEBUG
EMB_PROVIDER_LOG_TO_FILE=true
EMB_PROVIDER_LOG_FILE_MAX_SIZE=5
EMB_PROVIDER_LOG_RETENTION_DAYS=3
```

### 生产环境配置

```bash
# 使用INFO级别和较大的保留策略
EMB_PROVIDER_LOG_LEVEL=INFO
EMB_PROVIDER_LOG_TO_FILE=true
EMB_PROVIDER_LOG_FILE_MAX_SIZE=20
EMB_PROVIDER_LOG_RETENTION_DAYS=7
EMB_PROVIDER_LOG_MAX_DIR_SIZE_MB=20
```

### 容器环境配置

```bash
# 仅使用控制台输出，便于日志收集
EMB_PROVIDER_LOG_TO_FILE=false
EMB_PROVIDER_LOG_LEVEL=INFO
```

## 日志格式

所有日志采用结构化JSON格式：

```json
{
  "timestamp": "2025-11-09T12:30:45.123456",
  "level": "INFO",
  "message": "Request completed: POST /embeddings - Status: 200 - Time: 0.123s",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### DEBUG级别额外信息

```json
{
  "timestamp": "2025-11-09T12:30:45.123456",
  "level": "DEBUG",
  "message": "Processing batch request",
  "details": {
    "module": "embedding_service",
    "function": "process_batch",
    "line": 123,
    "thread": 140234567890,
    "process": 12345
  }
}
```

### 异常信息

```json
{
  "timestamp": "2025-11-09T12:30:45.123456",
  "level": "ERROR",
  "message": "API error: ValueError: Invalid input format",
  "exception": "Traceback (most recent call last):\n  File \"api.py\", line 45, in process_request\n    validate_input(data)\nValueError: Invalid input format"
}
```

## 代码中使用日志

### 基本用法

```python
from emb_model_provider.core.logging import get_logger

logger = get_logger("my_module")
logger.info("Application started")
logger.error("Something went wrong", exc_info=True)
```

### 带请求ID的日志

```python
from emb_model_provider.core.logging import log_with_request_id

log_with_request_id(
    logger,
    logging.INFO,
    "Processing user request",
    request_id="user-123"
)
```

### 模型事件日志

```python
from emb_model_provider.core.logging import log_model_event

log_model_event(
    "load",
    "all-MiniLM-L12-v2",
    details={"source": "local", "path": "/models/test"},
    request_id="req-456"
)
```

### API错误日志

```python
from emb_model_provider.core.logging import log_api_error

try:
    # API处理逻辑
    pass
except Exception as e:
    log_api_error(e, request=request, request_id="req-789")
```

## 日志文件管理

### 文件命名规则

```
logs/
├── app-2025-11-09-debug.log
├── app-2025-11-09-info.log
├── app-2025-11-09-warning.log
├── app-2025-11-09-error.log
├── app-2025-11-08-info.log.1.gz
├── app-2025-11-08-info.log.2.gz
└── cleanup.log
```

### 清理策略

1. **触发条件**：日志目录总大小超过 `EMB_PROVIDER_LOG_MAX_DIR_SIZE_MB`
2. **清理顺序**：
   - 首先尝试删除超过 `EMB_PROVIDER_LOG_RETENTION_DAYS` 的文件
   - 如果仍超限，尝试删除超过3天的文件
   - 如果仍超限，尝试删除超过1天的文件
3. **清理记录**：所有清理操作记录在 `cleanup.log` 中

### 监控建议

1. **磁盘空间监控**：定期检查日志目录大小
2. **错误日志监控**：设置ERROR级别日志告警
3. **性能监控**：监控日志写入性能，避免影响应用性能

## 故障排查

### 常见问题

1. **日志文件未创建**
   - 检查 `EMB_PROVIDER_LOG_TO_FILE` 是否为 `true`
   - 检查日志目录权限
   - 检查磁盘空间

2. **日志文件过大**
   - 调整 `EMB_PROVIDER_LOG_FILE_MAX_SIZE` 参数
   - 检查清理策略是否正常工作
   - 考虑调整日志级别

3. **清理不工作**
   - 检查 `EMB_PROVIDER_LOG_CLEANUP_INTERVAL_HOURS` 设置
   - 查看 `cleanup.log` 了解清理过程
   - 检查文件权限

### 调试方法

1. **启用DEBUG日志**：
   ```bash
   EMB_PROVIDER_LOG_LEVEL=DEBUG
   ```

2. **查看清理日志**：
   ```bash
   tail -f logs/cleanup.log
   ```

3. **检查日志配置**：
   ```python
   from emb_model_provider.core.config import config
   print(config.get_logging_config())
   ```

## 最佳实践

1. **生产环境**：使用INFO级别，启用文件日志
2. **开发环境**：使用DEBUG级别，较小的文件大小
3. **容器环境**：禁用文件日志，使用外部日志收集
4. **监控**：定期检查日志大小和错误率
5. **备份**：重要日志定期备份到外部存储
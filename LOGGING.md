# 日志系统用户指南

## 概述

本项目提供了完整的日志系统，支持控制台输出和文件输出，具有自动轮转、压缩和清理功能。日志采用结构化JSON格式，便于分析和监控。

## 快速开始

### 基本使用

```python
from emb_model_provider.core.logging import get_logger

# 获取日志器
logger = get_logger("my_module")

# 记录不同级别的日志
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
```

### 带请求ID的日志

```python
from emb_model_provider.core.logging import log_with_request_id

log_with_request_id(
    logger,
    logging.INFO,
    "处理用户请求",
    request_id="req-12345"
)
```

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

## 配置指南

### 环境变量配置

在 `.env` 文件中设置以下变量：

```bash
# 基础配置
EMB_PROVIDER_LOG_LEVEL=INFO                    # 日志级别
EMB_PROVIDER_LOG_TO_FILE=true                  # 启用文件日志
EMB_PROVIDER_LOG_DIR=logs                      # 日志目录

# 文件管理
EMB_PROVIDER_LOG_FILE_MAX_SIZE=10              # 单文件最大大小(MB)
EMB_PROVIDER_LOG_RETENTION_DAYS=7              # 默认保留天数
EMB_PROVIDER_LOG_CLEANUP_INTERVAL_HOURS=1      # 清理检查间隔(小时)
EMB_PROVIDER_LOG_MAX_DIR_SIZE_MB=50            # 目录最大大小(MB)，超过此值触发清理
EMB_PROVIDER_LOG_CLEANUP_TARGET_SIZE_MB=20     # 清理后目标大小(MB)
EMB_PROVIDER_LOG_CLEANUP_RETENTION_DAYS=7,3,1  # 清理策略
```

### 配置场景

#### 开发环境
```bash
EMB_PROVIDER_LOG_LEVEL=DEBUG
EMB_PROVIDER_LOG_TO_FILE=true
EMB_PROVIDER_LOG_FILE_MAX_SIZE=5
EMB_PROVIDER_LOG_RETENTION_DAYS=3
EMB_PROVIDER_LOG_MAX_DIR_SIZE_MB=3
```

#### 生产环境
```bash
EMB_PROVIDER_LOG_LEVEL=INFO
EMB_PROVIDER_LOG_TO_FILE=true
EMB_PROVIDER_LOG_FILE_MAX_SIZE=20
EMB_PROVIDER_LOG_RETENTION_DAYS=7
EMB_PROVIDER_LOG_MAX_DIR_SIZE_MB=20
```

#### 容器环境
```bash
EMB_PROVIDER_LOG_TO_FILE=false
EMB_PROVIDER_LOG_LEVEL=INFO
```

### 日志级别说明

| 级别 | 用途 | 建议 |
|------|------|------|
| DEBUG | 详细调试信息 | 开发环境使用 |
| INFO | 一般信息 | 生产环境推荐 |
| WARNING | 警告信息 | 需要关注的问题 |
| ERROR | 错误信息 | 需要立即处理 |

## 日志格式

### 标准格式
```json
{
  "timestamp": "2025-11-09T12:30:45.123456",
  "level": "INFO",
  "message": "Request completed: POST /embeddings - Status: 200",
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
  "exception": "Traceback (most recent call last):\n..."
}
```

## 文件管理

### 目录结构
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
1. **触发条件**：日志目录总大小超过配置限制
2. **清理顺序**：
   - 首先删除超过默认保留天数的文件
   - 如果仍超限，删除超过3天的文件
   - 如果仍超限，删除超过1天的文件
3. **清理记录**：所有清理操作记录在 `cleanup.log` 中

## 监控和维护

### 磁盘空间监控
```bash
# 检查日志目录大小
du -sh logs/

# 查看日志文件数量
find logs -name "*.log*" | wc -l
```

### 日志分析
```bash
# 查看最近的错误日志
tail -f logs/app-$(date +%Y-%m-%d)-error.log

# 统计错误数量
grep '"level": "ERROR"' logs/app-*.log | wc -l

# 查找特定请求的日志
grep '"request_id": "req-12345"' logs/app-*.log
```

### 性能监控
- 监控日志写入性能，避免影响应用性能
- 定期检查磁盘空间使用情况
- 设置ERROR级别日志告警

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

3. **检查配置**：
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

## API参考

### 核心函数

#### `get_logger(name)`
获取指定名称的日志器。

**参数：**
- `name` (str): 日志器名称

**返回：**
- `logging.Logger`: 日志器实例

#### `log_with_request_id(logger, level, message, request_id, **kwargs)`
记录带请求ID的日志。

**参数：**
- `logger` (logging.Logger): 日志器
- `level` (int): 日志级别
- `message` (str): 日志消息
- `request_id` (str, optional): 请求ID
- `**kwargs`: 额外参数

#### `log_model_event(event_type, model_name, details, request_id)`
记录模型事件。

**参数：**
- `event_type` (str): 事件类型
- `model_name` (str): 模型名称
- `details` (dict, optional): 详细信息
- `request_id` (str, optional): 请求ID

#### `log_api_error(error, request, request_id)`
记录API错误。

**参数：**
- `error` (Exception): 异常对象
- `request` (Request, optional): 请求对象
- `request_id` (str, optional): 请求ID

## 更新历史

- **v1.0.0** - 初始版本，支持基本日志功能
- **v1.1.0** - 添加文件输出和轮转功能
- **v1.2.0** - 添加智能清理策略
- **v1.3.0** - 添加按级别分离和压缩功能
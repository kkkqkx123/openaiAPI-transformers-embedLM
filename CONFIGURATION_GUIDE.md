# 配置系统使用说明

## 概述

本项目使用基于环境变量的配置系统，所有配置项均以 `EMB_PROVIDER_` 为前缀。配置系统基于 Pydantic 设置模型，支持类型验证和默认值设置。

## 配置项列表

### 模型配置

#### `EMB_PROVIDER_MODEL_PATH`
- **类型**: 字符串
- **默认值**: `D:\models\multilingual-MiniLM-L12-v2`
- **描述**: 模型目录路径
- **示例**: `EMB_PROVIDER_MODEL_PATH=/models/all-MiniLM-L12-v2`

#### `EMB_PROVIDER_MODEL_NAME`
- **类型**: 字符串
- **默认值**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **描述**: 模型名称

#### `EMB_PROVIDER_MODEL_ALIASES`
- **类型**: 字符串
- **默认值**: `""`
- **描述**: 模型别名列表，格式为 `alias1:actual_model_name1,alias2:actual_model_name2`
- **示例**: `EMB_PROVIDER_MODEL_ALIASES="mini:sentence-transformers/all-MiniLM-L12-v2,large:sentence-transformers/all-mpnet-base-v2"`

#### `EMB_PROVIDER_MODEL_MAPPING`
- **类型**: 字符串 (JSON格式)
- **默认值**: `{}`
- **描述**: 模型映射配置，将模型别名映射到实际模型配置
- **示例**:
```json
{
  "all-MiniLM-L12-v2": {
    "name": "sentence-transformers/all-MiniLM-L12-v2",
    "path": "",
    "source": "transformers",
    "precision": "fp16"
  }
}
```

### Transformers 模型加载配置

#### `EMB_PROVIDER_LOAD_FROM_TRANSFORMERS`
- **类型**: 布尔值
- **默认值**: `false`
- **描述**: 是否直接从 transformers 加载模型（不使用本地缓存）

#### `EMB_PROVIDER_TRANSFORMERS_MODEL_NAME`
- **类型**: 字符串
- **默认值**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **描述**: 从 transformers 直接加载的模型名称

#### `EMB_PROVIDER_TRANSFORMERS_TRUST_REMOTE_CODE`
- **类型**: 布尔值
- **默认值**: `false`
- **描述**: 是否信任远程代码（某些模型需要设置为 true）

### ModelScope 模型加载配置

#### `EMB_PROVIDER_MODELSCOPE_MODEL_NAME`
- **类型**: 字符串
- **默认值**: `""`
- **描述**: ModelScope 模型名称

#### `EMB_PROVIDER_MODELSCOPE_TRUST_REMOTE_CODE`
- **类型**: 布尔值
- **默认值**: `false`
- **描述**: 是否信任 ModelScope 模型的远程代码

#### `EMB_PROVIDER_MODELSCOPE_REVISION`
- **类型**: 字符串
- **默认值**: `master`
- **描述**: ModelScope 模型版本

#### `EMB_PROVIDER_MODELSCOPE_MODEL_PROVIDER`
- **类型**: 字符串
- **默认值**: `modelscope`
- **描述**: 模型提供者，可选值：`huggingface`, `modelscope`

#### `EMB_PROVIDER_MODELSCOPE_FALLBACK_TO_HUGGINGFACE`
- **类型**: 布尔值
- **默认值**: `true`
- **描述**: ModelScope 加载失败时是否回退到 Hugging Face

#### `EMB_PROVIDER_MODEL_SOURCE`
- **类型**: 字符串
- **默认值**: `huggingface`
- **描述**: 模型来源，可选值：`huggingface`, `modelscope`

### 多源加载配置

#### `EMB_PROVIDER_ENABLE_MULTI_SOURCE_LOADING`
- **类型**: 布尔值
- **默认值**: `false`
- **描述**: 是否启用多源模型加载支持

#### `EMB_PROVIDER_PRELOAD_MODELS`
- **类型**: 字符串
- **默认值**: `""`
- **描述**: 预加载模型列表（逗号分隔），这些模型将在服务启动时加载
- **示例**: `EMB_PROVIDER_PRELOAD_MODELS="all-MiniLM-L12-v2,all-mpnet-base-v2"`

#### `EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING`
- **类型**: 布尔值
- **默认值**: `true`
- **描述**: 是否启用动态模型加载。当设置为 `false` 时，只允许预加载列表中的模型被使用

#### `EMB_PROVIDER_ENABLE_OFFLINE_MODE`
- **类型**: 布尔值
- **默认值**: `false`
- **描述**: 是否启用离线模式（仅使用本地模型，不进行网络下载）

#### `EMB_PROVIDER_ENABLE_PATH_PRIORITY`
- **类型**: 布尔值
- **默认值**: `true`
- **描述**: 当指定了路径时，是否优先从本地路径加载模型

### 处理配置

#### `EMB_PROVIDER_MAX_BATCH_SIZE`
- **类型**: 整数
- **默认值**: `32`
- **范围**: 1-512
- **描述**: 最大批处理大小（同时处理的最大文本数量）

#### `EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING`
- **类型**: 布尔值
- **默认值**: `true`
- **描述**: 是否启用动态批处理以提高吞吐量

#### `EMB_PROVIDER_MAX_WAIT_TIME_MS`
- **类型**: 整数
- **默认值**: `100`
- **范围**: 10-1000
- **描述**: 动态批处理的最大等待时间（毫秒）

#### `EMB_PROVIDER_HARD_TIMEOUT_ADDITIONAL_SECONDS`
- **类型**: 浮点数
- **默认值**: `1.0`
- **范围**: 0.1-10.0
- **描述**: 在 max_wait_time 之后强制处理小批次的额外超时时间

#### `EMB_PROVIDER_MIN_BATCH_SIZE`
- **类型**: 整数
- **默认值**: `1`
- **范围**: 1-32
- **描述**: 动态批处理的最小批次大小

### 内存优化配置

#### `EMB_PROVIDER_ENABLE_LENGTH_GROUPING`
- **类型**: 布尔值
- **默认值**: `true`
- **描述**: 是否启用基于长度的分组以减少 padding 开销

#### `EMB_PROVIDER_LENGTH_GROUP_TOLERANCE`
- **类型**: 浮点数
- **默认值**: `0.2`
- **范围**: 0.1-0.5
- **描述**: 长度分组容忍度（20% 表示组内长度差异不超过 20%）

#### `EMB_PROVIDER_MAX_CONTEXT_LENGTH`
- **类型**: 整数
- **默认值**: `512`
- **范围**: 1-2048
- **描述**: 最大上下文长度（token 数量）

#### `EMB_PROVIDER_EMBEDDING_DIMENSION`
- **类型**: 整数
- **默认值**: `384`
- **范围**: 1+
- **描述**: 嵌入向量维度

### 资源配置

#### `EMB_PROVIDER_MEMORY_LIMIT`
- **类型**: 字符串
- **默认值**: `2GB`
- **描述**: 服务内存限制
- **示例**: `1GB`, `2GB`, `512MB`

#### `EMB_PROVIDER_DEVICE`
- **类型**: 字符串
- **默认值**: `auto`
- **描述**: 模型运行设备，可选值：`auto`, `cpu`, `cuda`

### 精度配置

#### `EMB_PROVIDER_MODEL_PRECISION`
- **类型**: 字符串
- **默认值**: `auto`
- **描述**: 模型精度，可选值：`auto`, `fp32`, `fp16`, `bf16`, `int8`, `int4`

#### `EMB_PROVIDER_MODEL_PRECISION_OVERRIDES`
- **类型**: 字符串 (JSON格式)
- **默认值**: `{}`
- **描述**: 模型特定精度覆盖配置
- **示例**:
```json
{
  "all-MiniLM-L12-v2": "fp16",
  "all-mpnet-base-v2": "fp32"
}
```

#### `EMB_PROVIDER_ENABLE_QUANTIZATION`
- **类型**: 布尔值
- **默认值**: `false`
- **描述**: 是否启用量化支持（int8/int4）

#### `EMB_PROVIDER_QUANTIZATION_METHOD`
- **类型**: 字符串
- **默认值**: `bitsandbytes`
- **描述**: 量化方法，可选值：`bitsandbytes`, `gptq`, `awq`

#### `EMB_PROVIDER_ENABLE_GPU_MEMORY_OPTIMIZATION`
- **类型**: 布尔值
- **默认值**: `true`
- **描述**: 是否启用 GPU 内存优化技术

### API 配置

#### `EMB_PROVIDER_HOST`
- **类型**: 字符串
- **默认值**: `localhost`
- **描述**: API 服务器绑定主机
- **示例**: `localhost`, `0.0.0`

#### `EMB_PROVIDER_PORT`
- **类型**: 整数
- **默认值**: `9000`
- **范围**: 1-65535
- **描述**: API 服务器端口

### 日志配置

#### `EMB_PROVIDER_LOG_LEVEL`
- **类型**: 字符串
- **默认值**: `INFO`
- **描述**: 日志级别，可选值：`DEBUG`, `INFO`, `WARNING`, `ERROR`

#### `EMB_PROVIDER_LOG_TO_FILE`
- **类型**: 布尔值
- **默认值**: `true`
- **描述**: 是否启用文件日志输出

#### `EMB_PROVIDER_LOG_DIR`
- **类型**: 字符串
- **默认值**: `logs`
- **描述**: 日志文件存储目录

#### `EMB_PROVIDER_LOG_FILE_MAX_SIZE`
- **类型**: 整数
- **默认值**: `10`
- **范围**: 1-100
- **描述**: 单个日志文件最大大小（MB）

#### `EMB_PROVIDER_LOG_RETENTION_DAYS`
- **类型**: 整数
- **默认值**: `7`
- **范围**: 1-30
- **描述**: 日志文件保留天数

#### `EMB_PROVIDER_LOG_CLEANUP_INTERVAL_HOURS`
- **类型**: 整数
- **默认值**: `1`
- **范围**: 1-24
- **描述**: 日志清理检查间隔（小时）

#### `EMB_PROVIDER_LOG_MAX_DIR_SIZE_MB`
- **类型**: 整数
- **默认值**: `50`
- **范围**: 1-1000
- **描述**: 日志目录最大总大小（MB）

#### `EMB_PROVIDER_LOG_CLEANUP_TARGET_SIZE_MB`
- **类型**: 整数
- **默认值**: `20`
- **范围**: 1-100
- **描述**: 清理后的目标大小（MB），应小于最大大小

#### `EMB_PROVIDER_LOG_CLEANUP_RETENTION_DAYS`
- **类型**: 字符串
- **默认值**: `7,3,1`
- **描述**: 清理策略保留天数（逗号分隔）

## 配置示例

### 开发环境配置
```bash
# 基础模型配置
EMB_PROVIDER_MODEL_PATH=D:\models\all-MiniLM-L12-v2
EMB_PROVIDER_MODEL_NAME=sentence-transformers/all-MiniLM-L12-v2
EMB_PROVIDER_LOAD_FROM_TRANSFORMERS=true

# 处理配置
EMB_PROVIDER_MAX_BATCH_SIZE=8
EMB_PROVIDER_MAX_WAIT_TIME_MS=50
EMB_PROVIDER_MIN_BATCH_SIZE=1

# 日志配置
EMB_PROVIDER_LOG_LEVEL=DEBUG
EMB_PROVIDER_LOG_TO_FILE=true
EMB_PROVIDER_LOG_RETENTION_DAYS=3

# 设备配置
EMB_PROVIDER_DEVICE=cpu
```

### 生产环境配置
```bash
# 模型配置
EMB_PROVIDER_LOAD_FROM_TRANSFORMERS=true
EMB_PROVIDER_TRANSFORMERS_MODEL_NAME=sentence-transformers/all-MiniLM-L12-v2

# 预加载配置
EMB_PROVIDER_PRELOAD_MODELS="all-MiniLM-L12-v2,all-mpnet-base-v2"
EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING=false

# 处理配置
EMB_PROVIDER_MAX_BATCH_SIZE=64
EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING=true
EMB_PROVIDER_MAX_WAIT_TIME_MS=150
EMB_PROVIDER_MIN_BATCH_SIZE=4

# 资源配置
EMB_PROVIDER_DEVICE=cuda
EMB_PROVIDER_MEMORY_LIMIT=4GB

# API 配置
EMB_PROVIDER_HOST=0.0.0.0
EMB_PROVIDER_PORT=9000

# 日志配置
EMB_PROVIDER_LOG_LEVEL=INFO
EMB_PROVIDER_LOG_FILE_MAX_SIZE=20
EMB_PROVIDER_LOG_RETENTION_DAYS=14
EMB_PROVIDER_LOG_MAX_DIR_SIZE_MB=100
EMB_PROVIDER_LOG_CLEANUP_TARGET_SIZE_MB=50
```

### 资源受限环境配置
```bash
# 模型配置
EMB_PROVIDER_LOAD_FROM_TRANSFORMERS=true
EMB_PROVIDER_TRANSFORMERS_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# 处理配置
EMB_PROVIDER_MAX_BATCH_SIZE=4
EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING=false
EMB_PROVIDER_ENABLE_LENGTH_GROUPING=false

# 资源配置
EMB_PROVIDER_DEVICE=cpu
EMB_PROVIDER_MEMORY_LIMIT=1GB

# 精度配置
EMB_PROVIDER_MODEL_PRECISION=int8
EMB_PROVIDER_ENABLE_QUANTIZATION=true

# 日志配置
EMB_PROVIDER_LOG_FILE_MAX_SIZE=5
EMB_PROVIDER_LOG_RETENTION_DAYS=2
EMB_PROVIDER_LOG_MAX_DIR_SIZE_MB=5
```

## 注意事项

1. **环境变量优先级**: 环境变量的值会覆盖配置文件中的默认值
2. **路径格式**: 在 Windows 系统中使用双反斜杠或正斜杠表示路径
3. **JSON 格式**: 包含 JSON 格式的配置项需要正确转义引号和特殊字符
4. **安全考虑**: 设置 `*_TRUST_REMOTE_CODE=true` 时需确保模型来源可信
5. **性能调优**: 根据硬件资源调整批处理大小和内存限制
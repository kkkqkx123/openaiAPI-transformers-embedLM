# 模型源配置使用指南

## 概述

本项目支持三种模型加载源：
- **transformers**: 从 Hugging Face Hub 加载模型
- **modelscope**: 从 ModelScope Hub 加载模型
- **local**: 从本地路径加载模型

## 本地模型配置优先级

### 优先级机制

项目采用基于本地路径存在的优先级机制：

1. **本地路径存在** → 使用本地加载器（LocalModelLoader）
2. **本地路径不存在** → 按配置的模型源加载（transformers 或 modelscope）

这种设计的目的是：
- 避免重复下载已存在的模型
- 提高加载速度
- 节省网络带宽和存储空间

### 配置参数

- `EMB_PROVIDER_ENABLE_PATH_PRIORITY`: 控制是否启用路径优先级（默认为 `true`）
- `EMB_PROVIDER_MODEL_PATH`: 指定本地模型路径
- `EMB_PROVIDER_MODEL_SOURCE`: 指定首选模型源（transformers/modelscope）

## 本地模型配置详解

### 1. 本地模型配置

当配置本地模型时，您需要：

1. **确保模型已下载到本地路径**
2. **配置正确的模型路径**

```bash
# 配置本地模型
EMB_PROVIDER_MODEL_PATH=/path/to/local/model
EMB_PROVIDER_MODEL_NAME=local-model-name
EMB_PROVIDER_MODEL_SOURCE=transformers  # 或 modelscope，用于备用
```

### 2. 禁用远程拉取

本地模型配置实际上已经"禁用"了远程拉取，因为：
- 如果本地路径存在，系统会直接使用本地模型
- 不会尝试从远程下载或拉取模型

### 3. 重新下载模型

当您需要重新下载或更新模型时，需要：

1. **删除本地模型目录**
2. **使用下载脚本重新下载模型**

## 模型下载方法

### 方法1: 使用下载脚本

项目提供了 `download_model.py` 脚本用于下载模型：

```bash
# 修改 download_model.py 中的模型名称和路径
python download_model.py
```

示例脚本内容：
```python
from transformers import AutoTokenizer, AutoModel
import os

def download_model():
    model_name = "Qwen/Qwen3-Embedding-0.6B"  # 修改为需要的模型
    model_path = "D:\\models\\qwen-embedding-0.6b" # 修改为本地路径
    
    print(f"正在下载模型 {model_name} 到 {model_path}")
    
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # 下载tokenizer
        print("下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(model_path)
        
        # 下载模型
        print("下载模型...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model.save_pretrained(model_path)
        
        print(f"模型下载完成，保存在: {model_path}")
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False
    
    return True
```

### 方法2: 使用 ModelScope 命令

参考 `modelscope下载模型.md` 文件：

```bash
# 下载 ModelScope 模型
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir D:\models\qwen-embedding-0.6b

# 或下载其他模型
modelscope download --model jinaai/jina-embeddings-v2-base-code --local_dir D:\models\jina-embeddings
```

## 配置示例

### 1. 纯本地模型配置

```bash
# 仅使用本地模型（不尝试远程拉取）
EMB_PROVIDER_MODEL_PATH=/models/all-MiniLM-L12-v2
EMB_PROVIDER_MODEL_NAME=sentence-transformers/all-MiniLM-L12-v2
EMB_PROVIDER_MODEL_SOURCE=local
EMB_PROVIDER_ENABLE_PATH_PRIORITY=true
EMB_PROVIDER_ENABLE_OFFLINE_MODE=true
```

### 2. 本地优先，远程备选配置

```bash
# 优先使用本地模型，本地不存在时从远程拉取
EMB_PROVIDER_MODEL_PATH=/models/all-MiniLM-L12-v2
EMB_PROVIDER_MODEL_NAME=sentence-transformers/all-MiniLM-L12-v2
EMB_PROVIDER_MODEL_SOURCE=transformers
EMB_PROVIDER_ENABLE_PATH_PRIORITY=true
EMB_PROVIDER_LOAD_FROM_TRANSFORMERS=true
```

### 3. ModelScope 模型配置

```bash
# 优先使用本地模型，本地不存在时从 ModelScope 拉取
EMB_PROVIDER_MODEL_PATH=/models/gte-chinese-base
EMB_PROVIDER_MODEL_NAME=damo/nlp_gte_sentence-embedding_chinese-base
EMB_PROVIDER_MODEL_SOURCE=modelscope
EMB_PROVIDER_ENABLE_PATH_PRIORITY=true
EMB_PROVIDER_MODELSCOPE_FALLBACK_TO_HUGGINGFACE=true
```

## 模型管理最佳实践

### 1. 更新模型流程

1. **停止服务**
2. **删除本地模型目录**
   ```bash
   rm -rf /path/to/model/directory  # Linux/Mac
   rmdir /path/to/model/directory /s  # Windows
   ```
3. **下载新模型**
   ```bash
   python download_model.py
   # 或使用 ModelScope 命令
   modelscope download --model MODEL_NAME --local_dir /path/to/model/directory
   ```
4. **重启服务**

### 2. 离线模式配置

```bash
# 确保只使用本地模型，不进行任何网络请求
EMB_PROVIDER_ENABLE_OFFLINE_MODE=true
EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING=false
EMB_PROVIDER_PRELOAD_MODELS="local-model1,local-model2"
EMB_PROVIDER_MODEL_MAPPING='{
  "local-model1": {
    "name": "sentence-transformers/all-MiniLM-L12-v2",
    "path": "/models/all-MiniLM-L12-v2",
    "source": "local"
  }
}'
```

## 注意事项

1. **路径存在检查**: 系统仅检查 `EMB_PROVIDER_MODEL_PATH` 指定的路径是否存在，不验证模型文件完整性
2. **模型文件完整性**: 请确保本地模型目录包含必要的模型文件（`config.json`, `pytorch_model.bin`, `tokenizer.json`, `tokenizer_config.json` 等）
3. **权限问题**: 确保应用有权限读取本地模型目录
4. **存储空间**: 本地模型通常占用大量存储空间，请确保有足够的磁盘空间

## 故障排除

### 问题1: 模型未从预期源加载

**原因**: 本地路径存在，系统优先使用本地模型
**解决方案**: 
- 检查 `EMB_PROVIDER_MODEL_PATH` 配置
- 如需强制使用远程源，删除本地模型目录

### 问题2: 本地模型加载失败

**原因**: 模型文件不完整或损坏
**解决方案**:
- 验证模型目录包含必要的文件
- 重新下载模型到本地

### 问题3: 离线模式下模型不可用

**原因**: 配置了远程模型但本地不存在
**解决方案**:
- 确保离线模式下使用本地模型路径
- 或预先下载所需模型到本地
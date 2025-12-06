# 量化模块使用示例

## 概述

本文档提供了在emb-model-provider项目中使用各种量化技术的详细示例，包括GPTQ、AutoAWQ和SINQ的实际应用场景。

## 安装量化依赖

首先安装量化相关依赖：

```bash
# 安装量化依赖组
uv sync --group quan

# 或者单独安装
pip install gptqmodel autoawq sinq safetensors gemlite
```

## 基本使用示例

### 1. GPTQ量化示例

#### 简单GPTQ量化

```python
import torch
from transformers import AutoTokenizer
from emb_model_provider.loaders import QuantizedModelLoader

# 使用GPTQ量化模型
loader = QuantizedModelLoader(
    model_name="meta-llama/Llama-2-7B-hf",
    quantization_method="gptq",
    quantization_config={
        "bits": 4,
        "group_size": 128,
        "desc_act": False
    }
)

# 加载量化模型
model, tokenizer = loader.load_model()

# 测试推理
prompt = "Quantization is important for"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"输入: {prompt}")
print(f"输出: {result}")
```

#### 使用预量化GPTQ模型

```python
from emb_model_provider.loaders import QuantizedModelLoader

# 加载预量化的GPTQ模型
loader = QuantizedModelLoader(
    model_name="TheBloke/Llama-2-7B-Chat-GPTQ",
    model_path="path/to/prequantized/model",  # 可选，如果模型在本地
    quantization_method="gptq"
)

model, tokenizer = loader.load_model()

# 检查模型信息
model_info = loader.get_model_info()
print(f"模型信息: {model_info}")
print(f"是否量化: {model_info['is_quantized']}")
print(f"量化方法: {model_info['quantization_method']}")
```

#### GPTQ动态配置

```python
from emb_model_provider.loaders import QuantizedModelLoader

# 使用动态量化配置
dynamic_config = {
    "bits": 4,
    "group_size": 128,
    "dynamic": {
        # 第19层gate模块使用4位
        r"+:.*\.18\..*gate.*": {"bits": 4, "group_size": 32},
        # 第20层gate模块使用8位
        r".*\.19\..*gate.*": {"bits": 8, "group_size": 64},
        # 跳过第21层gate模块
        r"-:.*\.20\..*gate.*": {},
        # 跳过所有down模块
        r"-:.*down.*": {},
    }
}

loader = QuantizedModelLoader(
    model_name="meta-llama/Llama-2-7B-hf",
    quantization_method="gptq",
    quantization_config=dynamic_config
)

model, tokenizer = loader.load_model()
```

### 2. AutoAWQ量化示例

#### 基本AWQ量化

```python
import torch
from transformers import AutoTokenizer
from emb_model_provider.loaders import QuantizedModelLoader

# 使用AWQ量化模型
loader = QuantizedModelLoader(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    quantization_method="awq",
    quantization_config={
        "bits": 4,
        "group_size": 128,
        "version": "GEMM"  # 或 "GEMV"
    }
)

model, tokenizer = loader.load_model()

# 测试推理
prompt = "What is the difference between GEMM and GEMV?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"输入: {prompt}")
print(f"输出: {result}")
```

#### AWQ融合模块优化

```python
from emb_model_provider.loaders import QuantizedModelLoader

# 使用融合模块优化
loader = QuantizedModelLoader(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    quantization_method="awq",
    quantization_config={
        "bits": 4,
        "group_size": 128,
        "version": "GEMM",
        "fuse_layers": True  # 启用融合模块
    }
)

model, tokenizer = loader.load_model()

# 预热模型
_ = model.generate(torch.tensor([[1]]), max_new_tokens=1)

# 现在模型应该运行得更快
```

### 3. SINQ量化示例

#### 基本SINQ量化

```python
import torch
from transformers import AutoTokenizer
from emb_model_provider.loaders import QuantizedModelLoader

# 使用SINQ量化模型
loader = QuantizedModelLoader(
    model_name="Qwen/Qwen3-1.7B",
    quantization_method="sinq",
    quantization_config={
        "nbits": 4,
        "group_size": 64,
        "tiling_mode": "1D",
        "method": "sinq"  # 或 "asinq" 用于校准版本
    }
)

model, tokenizer = loader.load_model()

# 测试推理
prompt = "Explain the concept of Sinkhorn normalization"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=80)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"输入: {prompt}")
print(f"输出: {result}")
```

#### SINQ推理优化

```python
from emb_model_provider.loaders import QuantizedModelLoader

# 加载SINQ模型
loader = QuantizedModelLoader(
    model_name="Qwen/Qwen3-1.7B",
    quantization_method="sinq",
    quantization_config={
        "nbits": 4,
        "group_size": 64,
        "tiling_mode": "1D",
        "method": "sinq"
    }
)

model, tokenizer = loader.load_model()

# 预热以初始化CUDA图
_ = model.forward(torch.tensor([[0]], device=model.device))

# 编译以加速推理
model.forward = torch.compile(
    model.forward,
    dynamic=True,
    fullgraph=False,
    backend="inductor",
    mode="reduce-overhead",
)

# 现在推理应该更快
```

#### 加载预量化SINQ模型

```python
from emb_model_provider.loaders import QuantizedModelLoader

# 从Hugging Face Hub加载预量化SINQ模型
loader = QuantizedModelLoader(
    model_name="huawei-csl/Qwen3-1.7B-SINQ-4bit",
    quantization_method="sinq"
)

model, tokenizer = loader.load_model()
```

## 高级使用示例

### 1. 批量量化多个模型

```python
from emb_model_provider.loaders import QuantizedModelLoader
import json

# 定义要量化的模型列表
models_to_quantize = [
    {
        "name": "meta-llama/Llama-2-7B-hf",
        "method": "gptq",
        "config": {"bits": 4, "group_size": 128}
    },
    {
        "name": "Qwen/Qwen3-1.7B",
        "method": "sinq",
        "config": {"nbits": 4, "group_size": 64}
    },
    {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "method": "awq",
        "config": {"bits": 4, "group_size": 128}
    }
]

# 批量量化
quantized_models = {}
for model_info in models_to_quantize:
    print(f"正在量化模型: {model_info['name']}")
    
    loader = QuantizedModelLoader(
        model_name=model_info["name"],
        quantization_method=model_info["method"],
        quantization_config=model_info["config"]
    )
    
    try:
        model, tokenizer = loader.load_model()
        quantized_models[model_info["name"]] = {
            "model": model,
            "tokenizer": tokenizer,
            "info": loader.get_model_info()
        }
        print(f"✓ 成功量化 {model_info['name']}")
    except Exception as e:
        print(f"✗ 量化失败 {model_info['name']}: {e}")

# 保存量化结果
with open("quantization_results.json", "w") as f:
    results = {
        name: {
            "method": info["info"]["quantization_method"],
            "config": info["info"]["quantization_config"],
            "memory_usage": info["info"].get("memory_usage", {})
        }
        for name, info in quantized_models.items()
    }
    json.dump(results, f, indent=2)
```

### 2. 性能基准测试

```python
import time
import torch
from transformers import AutoTokenizer
from emb_model_provider.loaders import QuantizedModelLoader, BaseModelLoader
from emb_model_provider.loaders.huggingface_loader import HuggingFaceLoader

def benchmark_model(model_name, quantization_method=None, quantization_config=None):
    """基准测试模型性能"""
    
    # 加载模型
    if quantization_method:
        loader = QuantizedModelLoader(
            model_name=model_name,
            quantization_method=quantization_method,
            quantization_config=quantization_config
        )
    else:
        loader = HuggingFaceLoader(model_name=model_name)
    
    model, tokenizer = loader.load_model()
    
    # 准备测试数据
    test_prompts = [
        "The future of artificial intelligence is",
        "Quantization techniques help to",
        "Large language models can be",
        "Memory efficiency is important for",
        "Model compression allows us to"
    ]
    
    # 预热
    for prompt in test_prompts[:2]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)
    
    # 基准测试
    times = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    # 获取内存使用情况
    memory_info = loader.get_memory_usage()
    
    return {
        "avg_time": sum(times) / len(times),
        "memory_usage": memory_info,
        "model_info": loader.get_model_info()
    }

# 比较不同量化方法
model_name = "Qwen/Qwen3-1.7B"
results = {}

# 原始模型
results["original"] = benchmark_model(model_name)

# GPTQ量化
results["gptq"] = benchmark_model(
    model_name, 
    "gptq", 
    {"bits": 4, "group_size": 128}
)

# SINQ量化
results["sinq"] = benchmark_model(
    model_name, 
    "sinq", 
    {"nbits": 4, "group_size": 64}
)

# 打印结果
print("性能基准测试结果:")
print("-" * 50)
for method, result in results.items():
    print(f"{method}:")
    print(f"  平均推理时间: {result['avg_time']:.3f}秒")
    print(f"  GPU内存使用: {result['memory_usage']['gpu_memory']:.2f}GB")
    print(f"  模型信息: {result['model_info'].get('quantization_method', 'N/A')}")
    print()
```

### 3. 内存优化示例

```python
import torch
from emb_model_provider.loaders import QuantizedModelLoader

def memory_efficient_inference(model_name, quantization_method, quantization_config):
    """内存高效的推理示例"""
    
    # 加载量化模型
    loader = QuantizedModelLoader(
        model_name=model_name,
        quantization_method=quantization_method,
        quantization_config=quantization_config
    )
    
    model, tokenizer = loader.load_model()
    
    # 启用梯度检查点以减少内存使用
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # 使用较小的批大小
    batch_size = 1
    
    # 准备输入
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "How does photosynthesis work?",
        "What are the benefits of exercise?",
        "Describe the process of digestion."
    ]
    
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # 分词
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        for j, output in enumerate(outputs):
            result = tokenizer.decode(output, skip_special_tokens=True)
            results.append(result)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    
    return results

# 使用示例
results = memory_efficient_inference(
    model_name="Qwen/Qwen3-1.7B",
    quantization_method="sinq",
    quantization_config={"nbits": 4, "group_size": 64}
)

for i, result in enumerate(results):
    print(f"结果 {i+1}: {result}\n")
```

### 4. 动态量化配置示例

```python
from emb_model_provider.core.quantization_config import QuantizationConfig
from emb_model_provider.loaders import QuantizedModelLoader

# 从配置文件创建量化配置
config = QuantizationConfig(
    method="gptq",
    enabled=True,
    gptq_bits=4,
    gptq_group_size=128,
    gptq_desc_act=False
)

# 使用配置加载模型
loader = QuantizedModelLoader(
    model_name="meta-llama/Llama-2-7B-hf",
    quantization_method=config.method,
    quantization_config=config.get_quantizer_config()
)

model, tokenizer = loader.load_model()

# 保存配置
with open("quantization_config.json", "w") as f:
    import json
    json.dump(config.dict(), f, indent=2)
```

### 5. 错误处理和回退示例

```python
import logging
from emb_model_provider.loaders import QuantizedModelLoader, HuggingFaceLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_with_fallback(model_name, quantization_method, quantization_config):
    """带回退机制的模型加载"""
    
    try:
        # 尝试加载量化模型
        logger.info(f"尝试加载{quantization_method}量化模型: {model_name}")
        loader = QuantizedModelLoader(
            model_name=model_name,
            quantization_method=quantization_method,
            quantization_config=quantization_config
        )
        
        if loader.is_model_available():
            model, tokenizer = loader.load_model()
            logger.info(f"成功加载{quantization_method}量化模型")
            return model, tokenizer, quantization_method
        else:
            raise Exception("量化模型不可用")
            
    except Exception as e:
        logger.warning(f"加载{quantization_method}量化模型失败: {e}")
        
        # 回退到原始模型
        logger.info("回退到原始模型")
        try:
            loader = HuggingFaceLoader(model_name=model_name)
            model, tokenizer = loader.load_model()
            logger.info("成功加载原始模型")
            return model, tokenizer, "original"
        except Exception as fallback_error:
            logger.error(f"加载原始模型也失败: {fallback_error}")
            raise

# 使用示例
try:
    model, tokenizer, actual_method = load_model_with_fallback(
        model_name="Qwen/Qwen3-1.7B",
        quantization_method="sinq",
        quantization_config={"nbits": 4, "group_size": 64}
    )
    
    print(f"成功加载模型，使用方法: {actual_method}")
    
    # 测试推理
    prompt = "Test prompt"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"推理结果: {result}")
    
except Exception as e:
    print(f"模型加载失败: {e}")
```

## 最佳实践

### 1. 选择合适的量化方法

- **GPTQ**：适合需要高质量量化的场景，支持多种后端
- **SINQ**：适合免校准场景，速度快且质量高
- **AutoAWQ**：虽然已被弃用，但在某些场景下仍有用

### 2. 优化量化参数

```python
# 根据模型大小调整参数
def get_optimal_quant_config(model_name, method):
    if "7B" in model_name:
        if method == "gptq":
            return {"bits": 4, "group_size": 128}
        elif method == "sinq":
            return {"nbits": 4, "group_size": 64}
    elif "13B" in model_name or "14B" in model_name:
        if method == "gptq":
            return {"bits": 4, "group_size": 128}
        elif method == "sinq":
            return {"nbits": 4, "group_size": 128}
    else:
        # 默认配置
        return {"bits": 4, "group_size": 128}
```

### 3. 监控资源使用

```python
import psutil
import torch

def monitor_resources():
    """监控系统资源使用"""
    # CPU使用率
    cpu_percent = psutil.cpu_percent()
    
    # 内存使用
    memory = psutil.virtual_memory()
    
    # GPU内存使用
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_percent = (gpu_memory / gpu_total) * 100
    else:
        gpu_memory = gpu_percent = 0
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "gpu_memory_gb": gpu_memory,
        "gpu_percent": gpu_percent
    }
```

## 故障排除

### 常见问题和解决方案

1. **内存不足错误**
   ```python
   # 解决方案：减少批大小或使用梯度检查点
   model.gradient_checkpointing_enable()
   ```

2. **量化精度下降**
   ```python
   # 解决方案：调整量化参数
   quant_config = {"bits": 8, "group_size": 64}  # 使用更高精度
   ```

3. **依赖冲突**
   ```bash
   # 解决方案：使用虚拟环境
   python -m venv quant_env
   source quant_env/bin/activate  # Linux/Mac
   quant_env\Scripts\activate  # Windows
   ```

## 参考资料

- [GPTQ使用指南](./gptq_guide.md)
- [AutoAWQ使用指南](./autoawq_guide.md)
- [SINQ使用指南](./sinq_guide.md)
- [集成指南](./integration_guide.md)
- [性能对比](./performance_comparison.md)
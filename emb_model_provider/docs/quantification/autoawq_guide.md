# AutoAWQ量化技术指南

## 概述

AutoAWQ是一个用于4-bit量化的易用包，实现了Activation-aware Weight Quantization (AWQ)算法。AutoAWQ能够将模型速度提升3倍，内存需求减少3倍（相比FP16）。需要注意的是，AutoAWQ已于2025年5月11日被官方弃用，不再维护。

## 技术原理

AWQ算法基于以下核心原理：
- **激活感知权重量化**：利用激活值的重要性来指导权重量化
- **通道级缩放**：为每个通道应用不同的缩放因子
- **混合精度**：在关键部分保持较高精度

## 系统要求

### NVIDIA GPU
- 计算能力7.5或更高（Turing架构及以后）
- CUDA 11.8或更高版本

### AMD GPU
- ROCm版本需与Triton兼容

### Intel CPU/GPU
- torch和intel_extension_for_pytorch版本至少2.4.0
- 或使用intel-xpu-backend-for-triton

## 安装

### 基础安装

```bash
pip install autoawq
```

### 带内核的安装

```bash
pip install autoawq[kernels]
```

### Intel CPU优化安装

```bash
pip install autoawq[cpu]
```

## 基本使用

### 模型量化

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 模型路径
model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
quant_path = 'mistral-instruct-v0.2-awq'

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 量化模型
model.quantize(tokenizer, quant_config=quant_config)

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### 针对嵌入模型的校准数据集

对于嵌入模型，AWQ需要使用与目标任务相关的校准数据集：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

# 嵌入模型路径
model_path = 'sentence-transformers/all-MiniLM-L6-v2'
quant_path = 'all-MiniLM-L6-v2-awq-4bit'

# 准备校准数据集
def prepare_embedding_calibration_data():
    """准备适合嵌入模型的校准数据"""
    
    # 选项1: 使用文本相似度数据集
    dataset = load_dataset("sentence-transformers/stsb", split="train")
    calibration_texts = []
    
    for item in dataset.select(range(512)):  # 使用512个样本
        calibration_texts.append(item["sentence1"])
        calibration_texts.append(item["sentence2"])
    
    return calibration_texts

# 选项2: 使用领域特定文本
def prepare_domain_calibration_data():
    """准备领域特定的校准数据"""
    
    # 示例：使用学术论文摘要
    dataset = load_dataset("scientific_papers", "arxiv", split="train")
    calibration_texts = dataset.select(range(1024))["abstract"]
    
    return calibration_texts

# 准备校准数据
calibration_texts = prepare_embedding_calibration_data()

# 将文本转换为token
def prepare_calibration_tokens(tokenizer, texts, max_length=512):
    """准备校准token数据"""
    calibration_data = []
    
    for text in texts:
        # 分词并截断
        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        calibration_data.append(tokens["input_ids"])
    
    # 合并为批次
    calibration_data = torch.cat(calibration_data, dim=0)
    return calibration_data

# 加载模型和分词器
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 准备校准数据
calibration_tokens = prepare_calibration_tokens(tokenizer, calibration_texts)

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 使用校准数据量化模型
model.quantize(calibration_tokens, quant_config=quant_config)

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### 使用自定义校准数据

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

# 自定义校准文本
custom_texts = [
    "This is an example sentence for embedding model quantization.",
    "Embedding models convert text into numerical representations.",
    "Quantization reduces model size while preserving performance.",
    # 添加更多与你的应用相关的文本...
]

# 模型路径
model_path = 'your-embedding-model'
quant_path = 'your-embedding-model-awq-4bit'

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 准备校准数据
calibration_tokens = prepare_calibration_tokens(tokenizer, custom_texts)

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 量化模型
model.quantize(calibration_tokens, quant_config=quant_config)

# 保存模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### 校准数据集最佳实践

1. **数据选择原则**：
   - 选择与实际应用场景相似的文本
   - 确保文本覆盖模型将遇到的各种情况
   - 避免使用与目标任务无关的通用文本

2. **数据量建议**：
   - 小模型：128-256个样本
   - 中等模型：512-1024个样本
   - 大模型：1024-2048个样本

3. **文本预处理**：
   - 清理特殊字符和格式
   - 保持一致的文本长度
   - 考虑模型的输入限制

### 模型推理

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from awq.utils.utils import get_best_device

# 获取最佳设备
device = get_best_device()
quant_path = "TheBloke/zephyr-7B-beta-AWQ"

# 加载量化模型
model = AutoAWQForCausalLM.from_quantized(
    quant_path, 
    fuse_layers=True, 
    device=device
)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

# 准备输入
prompt_template = """\
<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>
"""

prompt = "You're standing on the surface of the Earth. You walk one mile south, one mile west and one mile north. You end up exactly where you started. Where are you?"

tokens = tokenizer(
    prompt_template.format(prompt=prompt), 
    return_tensors='pt'
).input_ids.to(device)

# 生成输出
generation_output = model.generate(
    tokens, 
    streamer=streamer, 
    max_seq_len=512
)
```

## GEMM vs GEMV

AutoAWQ提供两种量化版本：

### GEMV（量化）
- 比GEMM快20%
- 仅支持批大小为1
- 不适合大上下文场景

### GEMM（量化）
- 在批大小小于8时比FP16快得多
- 适合大上下文场景
- 推荐用于大多数应用

### 计算限制 vs 内存限制

- **小批大小+小模型**：内存限制，量化模型更快
- **大批大小**：计算限制，量化可能不会带来加速

## 融合模块

融合模块是AutoAWQ加速的关键：

```python
model = AutoAWQForCausalLM.from_quantized(
    quant_path, 
    fuse_layers=True,  # 启用融合模块
    max_seq_len=seq_len, 
    batch_size=batch_size
)
```

### 融合模块特点
- 将多个层合并为单个操作
- 实现自定义缓存
- 预分配基于批大小和序列长度
- 主要加速器来自FasterTransformer（仅Linux）

## 性能基准

### RTX 4090基准测试

| 模型 | 大小 | 版本 | 批大小 | 预填充tokens/s | 解码tokens/s | 内存(GB) |
|------|------|------|--------|----------------|--------------|----------|
| Mistral | 7B | GEMM | 1 | 1093.35 | 156.317 | 4.35 |
| Mistral | 7B | GEMV | 1 | 531.99 | 188.29 | 4.28 |
| Llama 2 | 13B | GEMM | 1 | 820.34 | 96.74 | 8.47 |
| CodeLlama | 34B | GEMM | 1 | 681.74 | 41.01 | 19.05 |

### CPU基准测试（48核Intel Xeon）

| 模型 | 版本 | 批大小 | 预填充tokens/s | 解码tokens/s | 内存(GB) |
|------|------|--------|----------------|--------------|----------|
| TinyLlama 1B | gemm | 1 | 817.86 | 70.93 | 1.94 |
| Mistral 7B | gemm | 1 | 343.08 | 28.46 | 9.74 |
| Llama 2 13B | gemm | 1 | 220.79 | 18.14 | 17.46 |

## 与当前项目集成

### 依赖配置

在`pyproject.toml`中添加AutoAWQ相关依赖：

```toml
[project.optional-dependencies]
quan = [
    "autoawq>=0.2.9",
    "transformers>=4.51.3",
    "torch>=2.6.0",
]
```

### 模型加载器扩展

```python
from awq import AutoAWQForCausalLM
from emb_model_provider.loaders.base_loader import BaseModelLoader

class AutoAWQLoader(BaseModelLoader):
    def load_model(self):
        model = AutoAWQForCausalLM.from_quantized(
            self.model_name,
            fuse_layers=True,
            device=self.get_device()
        )
        tokenizer = model.tokenizer
        return model, tokenizer
```

## 替代方案

由于AutoAWQ已被弃用，推荐考虑以下替代方案：

1. **vLLM项目**：已完全采用AutoAWQ
   - GitHub: https://github.com/vllm-project/llm-compressor

2. **MLX-LM**：支持Mac设备的AWQ
   - GitHub: https://github.com/ml-explore/mlx-lm

3. **GPTQModel**：继续维护的量化库
   - GitHub: https://github.com/ModelCloud/GPTQModel

## 注意事项

1. **弃用状态**：AutoAWQ已不再维护，建议使用替代方案
2. **版本兼容性**：最后测试配置为Torch 2.6.0和Transformers 4.51.3
3. **硬件要求**：需要支持的GPU架构和CUDA版本
4. **性能权衡**：根据使用场景选择GEMM或GEMV版本

## 参考资料

- [AutoAWQ GitHub](https://github.com/casper-hansen/AutoAWQ)
- [AWQ论文](https://arxiv.org/abs/2306.00978)
- [vLLM项目](https://github.com/vllm-project/llm-compressor)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
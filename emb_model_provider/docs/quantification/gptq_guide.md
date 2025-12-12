# GPTQ量化技术指南

## 概述

GPTQ (Generative Pre-trained Transformer Quantization) 是一种用于大型语言模型的后训练量化技术，能够将模型权重压缩到4位或更低精度，同时保持较高的模型性能。GPTQ通过分析模型的激活值来确定最优的量化参数，实现高效的模型压缩。

## 技术原理

GPTQ基于以下核心原理：
- **激活感知量化**：利用模型在典型输入下的激活分布来指导量化过程
- **分组量化**：将权重按组进行量化，通常组大小为128
- **二阶优化**：考虑量化误差的二阶信息，提高量化精度

## GPTQModel库介绍

GPTQModel是一个生产就绪的LLM模型压缩和量化工具包，支持通过HF、vLLM和SGLang进行CPU/GPU加速推理。

### 主要特性

- 支持多种量化位数（2-8位）
- 支持多种后端（vLLM、SGLang、BitBLAS、IPEX等）
- 支持动态量化配置
- 支持EoRA（Eigenspace Low-Rank Approximation）补偿
- 兼容HuggingFace和ModelScope模型库

## 安装

### 基础安装

```bash
pip install gptqmodel --no-build-isolation
```

### 带可选模块的安装

```bash
pip install gptqmodel[vllm,sglang,bitblas,ipex,auto_round] --no-build-isolation
```

### 从源码安装

```bash
git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel
pip install -v . --no-build-isolation
```

## 基本使用

### 模型量化

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

# 模型路径
model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

# 准备校准数据集
calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
).select(range(1024))["text"]

# 配置量化参数
quant_config = QuantizeConfig(bits=4, group_size=128)

# 加载模型
model = GPTQModel.load(model_id, quant_config)

# 执行量化
model.quantize(calibration_dataset, batch_size=1)

# 保存量化模型
model.save(quant_path)
```

### 针对嵌入模型的校准数据集

对于嵌入模型，使用与目标任务相关的校准数据集非常重要：

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

# 嵌入模型路径
model_id = "sentence-transformers/all-MiniLM-L6-v2"
quant_path = "all-MiniLM-L6-v2-gptq-4bit"

# 准备嵌入任务相关的校准数据集
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

# 选项3: 使用自定义文本数据
def prepare_custom_calibration_data(texts):
    """使用自定义文本数据作为校准数据"""
    
    # 确保文本长度适中
    processed_texts = []
    for text in texts:
        if len(text.split()) > 10:  # 过滤太短的文本
            processed_texts.append(text[:512])  # 截断到512字符
    
    return processed_texts[:1024]  # 限制到1024个样本

# 准备校准数据
calibration_dataset = prepare_embedding_calibration_data()

# 配置量化参数
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,  # 对于嵌入模型通常设为False
    sym=True,       # 对称量化
)

# 加载模型
model = GPTQModel.load(model_id, quant_config)

# 执行量化
model.quantize(calibration_dataset, batch_size=1)

# 保存量化模型
model.save(quant_path)
```

### 校准数据集最佳实践

1. **数据选择原则**：
   - 选择与目标任务相似的文本
   - 确保文本多样性
   - 避免包含过多特殊字符或格式

2. **数据量建议**：
   - 小模型（<1B参数）：128-256个样本
   - 中等模型（1B-7B参数）：512-1024个样本
   - 大模型（>7B参数）：1024-2048个样本

3. **文本长度**：
   - 保持与实际使用场景相似的文本长度
   - 避免过短（<10词）或过长（>512词）的文本
   - 考虑模型的输入限制

### 模型推理

```python
from gptqmodel import GPTQModel

# 加载量化模型
model = GPTQModel.load("ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v2.5")

# 执行推理
result = model.generate("Uncovering deep insights begins with")[0]
print(model.tokenizer.decode(result))
```

### 动态量化配置

```python
dynamic = { 
    # 正匹配：第19层，gate模块
    r"+:.*\.18\..*gate.*": {"bits": 4, "group_size": 32},  
    
    # 正匹配：第20层，gate模块
    r".*\.19\..*gate.*": {"bits": 8, "group_size": 64},  
    
    # 负匹配：跳过第21层，gate模块
    r"-:.*\.20\..*gate.*": {}, 
    
    # 负匹配：跳过所有层的down模块
    r"-:.*down.*": {},  
}

quant_config = QuantizeConfig(bits=4, group_size=128, dynamic=dynamic)
```

## 高级功能

### EoRA补偿

EoRA是一种训练无关的补偿方法，用于提高量化模型的性能：

```python
from gptqmodel import GPTQModel
from gptqmodel.adapter.adapter import Lora 

# 初始化EoRA适配器
eora = Lora(
    path='GPTQModel/examples/eora/Llama-3.2-3B-4bits-eora_rank64_c4',
    rank=64,
)

# 加载带EoRA的模型
model = GPTQModel.load(
    model_id_or_path='sliuau/Llama-3.2-3B_4bits_128group_size',
    adapter=eora,
)

# 执行推理
tokens = model.generate("Capital of France is")[0]
result = model.tokenizer.decode(tokens)
print(f"Result: {result}")
```

### 多后端支持

GPTQModel支持多种推理后端：

- **vLLM**：高性能推理引擎
- **SGLang**：结构化生成语言
- **BitBLAS**：位级线性代数库
- **IPEX**：Intel扩展PyTorch

## 性能优化

### 量化参数选择

- **bits**：通常选择4位，在性能和模型质量间取得平衡
- **group_size**：通常选择128，较大的组大小可以提高精度但降低压缩率
- **version**：推荐使用v2版本，提供更好的量化质量

### 校准数据集

- 选择与目标任务相似的数据集
- 通常使用512-1024个样本进行校准
- 确保数据集的多样性和代表性

## 与当前项目集成

### 依赖配置

在`pyproject.toml`中添加GPTQ相关依赖：

```toml
[project.optional-dependencies]
quan = [
    "gptqmodel>=0.9.0",
    "auto-gptq>=0.7.0",
    "optimum>=1.16.0",
]
```

### 模型加载器扩展

可以扩展现有的模型加载器以支持GPTQ量化：

```python
from gptqmodel import GPTQModel
from emb_model_provider.loaders.base_loader import BaseModelLoader

class GPTQModelLoader(BaseModelLoader):
    def load_model(self):
        model = GPTQModel.load(self.model_name)
        tokenizer = model.tokenizer
        return model, tokenizer
```

## 注意事项

1. **硬件要求**：需要足够的GPU内存进行量化过程
2. **兼容性**：确保PyTorch和transformers版本兼容
3. **模型质量**：量化可能会影响模型性能，建议进行充分测试
4. **许可证**：遵守相关模型的许可证要求

## 参考资料

- [GPTQModel GitHub](https://github.com/ModelCloud/GPTQModel)
- [GPTQ论文](https://arxiv.org/abs/2210.17323)
- [GPTQv2论文](https://arxiv.org/abs/2504.02692)
- [EoRA论文](https://arxiv.org/abs/2410.21271)
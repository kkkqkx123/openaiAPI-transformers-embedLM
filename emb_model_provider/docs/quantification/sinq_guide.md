# SINQ量化技术指南

## 概述

SINQ (Sinkhorn-Normalized Quantization) 是一种新颖、快速且高质量的量化方法，专为大型语言模型设计，能够在保持准确性的同时显著减小模型大小。SINQ是一种免校准的量化技术，通过引入双缩放因子和基于Sinkhorn-Knopp算法的归一化方法来解决传统量化中的异常值问题。

## 技术原理

### 核心创新

#### 1. 双缩放机制

传统量化方法每个权重维度只使用一个缩放因子，这使得模型容易受到异常值的影响。SINQ通过引入**双缩放**机制解决这个问题：

- **行缩放因子**：为每行权重设置独立的缩放因子
- **列缩放因子**：为每列权重设置独立的缩放因子

这种灵活性重新分配了异常值的影响，使量化误差更小且更平衡。

#### 2. Sinkhorn归一化优化

SINQ使用受Sinkhorn矩阵归一化启发的迭代算法，重新缩放行和列以平衡它们的方差。通过减少整体的**矩阵不平衡**，权重变得更容易量化，即使在极低位宽下也能保持一致的高精度。

#### 3. 更均匀的误差分布

与标准单缩放量化相比，SINQ的误差分布更加均匀且不那么严重，即使在3位精度下也能保持模型准确性。

## 主要特性

### 量化类型支持

- **对称和非对称量化**：同时支持两种量化方式
- **NF4支持**：支持非均匀4位量化
- **多种位宽**：支持2、3、4、5、6、8位量化

### 性能优势

- **免校准**：不需要校准数据集
- **快速量化**：比HQQ快约2倍，比AWQ快约4倍
- **高质量**：在相同精度下提供更好的模型性能
- **模型无关**：不需要了解特定的LLM架构

## 安装

### 基础安装

```bash
# 克隆仓库
git clone https://github.com/huawei-csl/SINQ.git
cd SINQ

# 安装依赖
pip install -r req.txt

# 安装SINQ
pip install .
```

### 可选依赖

```bash
# 用于保存和加载分片safetensors
pip install safetensors
pip install gemlite==0.5.1.post1
```

## 基本使用

### 模型量化

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig

model_name = "Qwen/Qwen3-1.7B"
device = "cuda:0"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 配置量化参数
quant_cfg = BaseQuantizeConfig(
    nbits=4,            # 量化位宽
    group_size=64,      # 组大小
    tiling_mode="1D",   # 平铺策略
    method="sinq"       # 量化方法（"asinq"用于校准版本）
)

# 执行量化
qmodel = AutoSINQHFModel.quantize_model(
    model,
    tokenizer=tokenizer,
    quant_config=quant_cfg,
    compute_dtype=torch.bfloat16,
    device=device
)
```

### 保存和加载模型

```python
# 保存为分片safetensors格式
save_dir = "qwen3-1.7b-sinq-4bit"
AutoSINQHFModel.save_quantized_safetensors(
    qmodel,
    tokenizer,
    save_dir,
    verbose=True,
    max_shard_size="4GB",
)

# 从分片safetensors加载
tokenizer = AutoTokenizer.from_pretrained(save_dir)
device = "cuda:0"
qmodel = AutoSINQHFModel.from_quantized_safetensors(
    save_dir,
    device=device,
    compute_dtype=torch.bfloat16,
)
```

### 推理加速

```python
# 预热以初始化CUDA图
_ = qmodel.forward(torch.tensor([[0]], device=device))

# 编译以加速推理
qmodel.forward = torch.compile(
    qmodel.forward,
    dynamic=True,
    fullgraph=False,
    backend="inductor",
    mode="reduce-overhead",
)
```

## 配置参数

### 主要参数

| 参数 | 描述 | 选项 | 默认值 |
|------|------|------|--------|
| `nbits` | 权重量化位宽 | 2, 3, 4, 5, 6, 8 | 4 |
| `tiling_mode` | 权重矩阵平铺策略 | 1D, 2D | 1D |
| `group_size` | 每个量化组的权重数 | 64, 128 | 64 |
| `method` | 量化方法 | sinq, asinq | sinq |

### 方法选择

- **sinq**：免校准版本，快速且高质量
- **asinq**：校准版本，结合AWQ校准以获得更高精度
- **sinq_nf4**：非均匀4位量化版本

## 性能基准

### 量化速度

- **Qwen3-14B**：约21秒
- **DeepSeekV2.5-236B**：约5分钟

### 内存节省

- **DeepSeekV2.5-236B**：从~472GB减少到~110GB
- **精度损失**：WikiText2和C4上< 1 ppl

### 与其他方法对比

| 特性 | SINQ | HQQ | A-SINQ | AWQ |
|------|------|-----|--------|-----|
| 校准 | 免校准 | 免校准 | 校准 | 校准 |
| 量化类型 | 对称&非对称 | 仅非对称 | 对称&非对称 | 对称&非对称 |
| NF4支持 | 是 | 否 | 是 | 否 |
| 量化速度 | 比HQQ快2倍 | 较慢 | 比AWQ快4倍 | 较慢 |
| 模型质量 | 更高 | 较低 | 更高 | 较低 |

## 高级功能

### 与lm-eval框架集成

```python
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# 包装量化模型和分词器
lm = HFLM(pretrained=qmodel, tokenizer=tokenizer, device=device)

# 评估
results = evaluator.simple_evaluate(
    model=lm,
    tasks=["lambada_openai"],
    device=device
)
```

### 从Hugging Face Hub加载预量化模型

```python
import torch
from transformers import AutoTokenizer
from sinq.patch_model import AutoSINQHFModel

model_name = "huawei-csl/<model_name>"  # 从集合中选择模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda:0"

qmodel = AutoSINQHFModel.from_quantized_safetensors(
    model_name,
    device=device,
    compute_dtype=torch.bfloat16,
)
```

## 与当前项目集成

### 依赖配置

在`pyproject.toml`中添加SINQ相关依赖：

```toml
[project.optional-dependencies]
quan = [
    "sinq @ git+https://github.com/huawei-csl/SINQ.git",
    "safetensors",
    "gemlite==0.5.1.post1",
]
```

### 模型加载器扩展

```python
from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig
from emb_model_provider.loaders.base_loader import BaseModelLoader

class SINQLoader(BaseModelLoader):
    def load_model(self):
        quant_cfg = BaseQuantizeConfig(
            nbits=4,
            group_size=64,
            tiling_mode="1D",
            method="sinq"
        )
        
        model = AutoSINQHFModel.quantize_model(
            self.model_name,
            quant_config=quant_cfg,
            device=self.get_device()
        )
        tokenizer = model.tokenizer
        return model, tokenizer
```

## 未来发展

### 即将推出的功能

- 🤗 与Hugging Face Transformers集成
- 支持Conv2D层和timm模型
- 混合精度量化支持
- vLLM、SGLang和llama.cpp框架支持

### 持续更新

- [2025/09/26] SINQ论文发布
- [2025/09/30] SINQ GitHub仓库公开
- [2025/10/02] 论文被Hugging Face Papers收录
- [2025/10/17] 首批预量化SINQ模型在Hugging Face Hub发布
- [2025/10/23] 使用gemlite后端实现更快推理

## 注意事项

1. **首次运行**：由于内核/图编译，首次运行会较慢，后续运行会快得多
2. **内存要求**：量化过程需要足够的GPU内存
3. **兼容性**：确保PyTorch版本兼容
4. **模型质量**：虽然SINQ提供高质量量化，但仍建议在目标任务上进行测试

## 参考资料

- [SINQ GitHub](https://github.com/huawei-csl/SINQ)
- [SINQ论文](https://arxiv.org/abs/2509.22944)
- [Hugging Face SINQ集合](https://huggingface.co/collections/huawei-csl/sinq)
- [Qwen3-Quantization](https://github.com/Efficient-ML/Qwen3-Quantization)
- [HQQ](https://github.com/mobiusml/hqq)

#!/usr/bin/env python3
"""
验证jina-code模型是否能正常加载的脚本
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_model_files(model_path: str) -> bool:
    """检查模型文件是否存在且完整"""
    logger.info(f"检查模型文件路径: {model_path}")
    
    model_dir = Path(model_path)
    if not model_dir.exists():
        logger.error(f"模型目录不存在: {model_path}")
        return False
    
    # 检查关键文件
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
            logger.warning(f"缺少文件: {file_name}")
        else:
            file_size = file_path.stat().st_size
            logger.info(f"找到文件: {file_name} ({file_size:,} bytes)")
    
    # 检查是否有其他可能的模型文件格式
    alternative_files = [
        "model.safetensors",
        "model.bin"
    ]
    
    found_alternative = False
    for file_name in alternative_files:
        file_path = model_dir / file_name
        if file_path.exists():
            found_alternative = True
            file_size = file_path.stat().st_size
            logger.info(f"找到替代模型文件: {file_name} ({file_size:,} bytes)")
    
    if not found_alternative and "pytorch_model.bin" in missing_files:
        logger.error("未找到任何模型权重文件")
        return False
    
    if missing_files:
        logger.warning(f"缺少一些文件，但可能仍能加载: {missing_files}")
    
    logger.info("模型文件检查完成")
    return True

def load_model_with_transformers(model_path: str, precision: str = "fp16") -> Optional[Any]:
    """使用transformers加载模型"""
    try:
        from transformers import AutoModel, AutoTokenizer
        
        logger.info(f"尝试加载模型: {model_path}")
        logger.info(f"精度设置: {precision}")
        
        # 设置torch_dtype
        torch_dtype = "float16" if precision == "fp16" else "float32"
        
        # 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        logger.info(f"Tokenizer加载成功: {type(tokenizer)}")
        
        # 加载模型
        logger.info("加载模型...")
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        logger.info(f"模型加载成功: {type(model)}")
        
        return model, tokenizer
        
    except ImportError as e:
        logger.error(f"缺少必要的库: {e}")
        logger.error("请安装transformers: pip install transformers torch")
        return None
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        return None

def test_model_functionality(model, tokenizer) -> bool:
    """测试模型基本功能"""
    try:
        import torch
        
        logger.info("测试模型基本功能...")
        
        # 测试文本
        test_texts = [
            "def hello_world():",
            "print('Hello, World!')",
            "class MyClass:",
            "import numpy as np"
        ]
        
        for text in test_texts:
            logger.info(f"测试文本: {text}")
            
            # 编码
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            logger.info(f"输入形状: {inputs['input_ids'].shape}")
            
            # 推理
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 获取embedding
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
                # 平均池化
                pooled_embeddings = embeddings.mean(dim=1)
                logger.info(f"输出embedding形状: {pooled_embeddings.shape}")
                logger.info(f"embedding范数: {pooled_embeddings.norm(dim=-1).item():.4f}")
            elif hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
                logger.info(f"输出embedding形状: {embeddings.shape}")
                logger.info(f"embedding范数: {embeddings.norm(dim=-1).item():.4f}")
            else:
                logger.warning("未找到预期的输出格式")
                logger.info(f"可用输出: {outputs.keys() if hasattr(outputs, 'keys') else type(outputs)}")
        
        logger.info("模型功能测试完成")
        return True
        
    except Exception as e:
        logger.error(f"测试模型功能时出错: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始验证jina-code模型...")
    
    # 从.env文件读取配置
    model_config = {
        "name": "jinaai/jina-code-embeddings-0.5b",
        "model_path": "D:/models/jina-code-emb-0.5b",
        "source": "local",
        "precision": "fp16"
    }
    
    logger.info(f"模型配置: {model_config}")
    
    # 步骤1: 检查模型文件
    if not check_model_files(model_config["model_path"]):
        logger.error("模型文件检查失败")
        return False
    
    # 步骤2: 加载模型
    result = load_model_with_transformers(
        model_config["model_path"], 
        model_config["precision"]
    )
    
    if result is None:
        logger.error("模型加载失败")
        return False
    
    model, tokenizer = result
    
    # 步骤3: 测试模型功能
    if not test_model_functionality(model, tokenizer):
        logger.error("模型功能测试失败")
        return False
    
    logger.info("✅ jina-code模型验证成功！")
    logger.info(f"模型名称: {model_config['name']}")
    logger.info(f"模型路径: {model_config['model_path']}")
    logger.info(f"精度: {model_config['precision']}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
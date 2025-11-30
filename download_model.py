#!/usr/bin/env python3
"""
下载模型的脚本（修正版）
"""

from transformers import AutoTokenizer, AutoModel
import os

def download_model():
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    model_path = "D:\\models\\qwen-embedding-0.6b"
    
    print(f"正在下载模型 {model_name} 到 {model_path}")
    
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # 下载tokenizer（加 trust_remote_code 更稳妥）
        print("下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True  # 新增：加载自定义 Tokenizer 配置（若有）
        )
        tokenizer.save_pretrained(model_path)
        
        # 下载模型（必须加 trust_remote_code，加载自定义架构）
        print("下载模型...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True  # 核心：加载自定义架构代码
        )
        model.save_pretrained(model_path)
        
        print(f"模型下载完成，保存在: {model_path}")
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_model()
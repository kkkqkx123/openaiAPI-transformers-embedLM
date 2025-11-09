#!/usr/bin/env python3
"""
下载模型的脚本
"""

from transformers import AutoTokenizer, AutoModel
import os

def download_model():
    # model_name = "sentence-transformers/all-MiniLM-L12-v2"
    # model_path = "D:\\models\\all-MiniLM-L12-v2"
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_path = "D:\\models\\multilingual-MiniLM-L12-v2"
    
    print(f"正在下载模型 {model_name} 到 {model_path}")
    
    # 创建目录
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # 下载tokenizer
        print("下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)
        
        # 下载模型
        print("下载模型...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(model_path)
        
        print(f"模型下载完成，保存在: {model_path}")
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_model()
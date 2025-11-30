#!/usr/bin/env python3
"""检查当前配置状态的脚本"""

import os
import sys

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emb_model_provider.core.config import config

def main():
    print("=== 当前配置状态检查 ===")
    
    # 检查model_mapping配置
    print(f"model_mapping 配置值: {repr(config.model_mapping)}")
    print(f"model_mapping 是否为空: {not config.model_mapping}")
    print(f"model_mapping 是否等于 '{{}}': {config.model_mapping == '{}'}")
    
    # 检查get_model_mapping方法
    model_mapping_result = config.get_model_mapping()
    print(f"get_model_mapping() 结果: {model_mapping_result}")
    
    # 检查get_model_info方法
    default_model_info = config.get_model_info('default')
    print(f"get_model_info('default') 结果: {default_model_info}")
    
    # 检查默认模型名称
    print(f"默认模型名称: {config.model_name}")
    
    # 检查预加载模型列表
    preload_models = config.get_preload_models()
    print(f"预加载模型列表: {preload_models}")
    
    # 检查is_model_preloaded方法
    is_default_preloaded = config.is_model_preloaded('default')
    print(f"is_model_preloaded('default') 结果: {is_default_preloaded}")
    
    print("=== 检查完成 ===")

if __name__ == "__main__":
    main()
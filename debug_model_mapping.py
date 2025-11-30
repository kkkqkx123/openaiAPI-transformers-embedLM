#!/usr/bin/env python3
"""详细调试模型映射解析过程的脚本"""

import os
import sys
import json

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emb_model_provider.core.config import Config

def main():
    print("=== 详细调试模型映射解析过程 ===")
    
    # 设置测试环境变量
    test_config = {
        "EMB_PROVIDER_ENABLE_MULTI_SOURCE_LOADING": "true",
        "EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING": "true", 
        "EMB_PROVIDER_PRELOAD_MODELS": "default",
        "EMB_PROVIDER_MODEL_MAPPING": "{\"default\": {\"name\": \"sentence-transformers/all-MiniLM-L12-v2\", \"path\": \"D:/项目/llm/models/sentence-transformers_all-MiniLM-L12-v2\", \"source\": \"transformers\"}, \"mini\": {\"name\": \"sentence-transformers/all-MiniLM-L6-v2\", \"path\": \"D:/项目/llm/models/sentence-transformers_all-MiniLM-L6-v2\", \"source\": \"transformers\"}}",
        "EMB_PROVIDER_MAX_BATCH_SIZE": "32",
        "EMB_PROVIDER_MAX_CONTEXT_LENGTH": "512",
        "EMB_PROVIDER_DEVICE": "cpu",
        "EMB_PROVIDER_HOST": "localhost",
        "EMB_PROVIDER_PORT": "9000",
        "EMB_PROVIDER_LOG_LEVEL": "INFO"
    }
    
    # 清理现有环境变量，避免冲突
    for key in list(os.environ.keys()):
        if key.startswith("EMB_PROVIDER_"):
            del os.environ[key]
    
    # 设置新的环境变量
    os.environ.update(test_config)
    
    # 初始化配置
    config = Config()
    
    print("1. 检查原始model_mapping配置值:")
    print(f"   config.model_mapping = {repr(config.model_mapping)}")
    
    print("\n2. 直接解析JSON字符串:")
    try:
        direct_parse = json.loads(config.model_mapping)
        print(f"   json.loads(config.model_mapping) = {direct_parse}")
    except Exception as e:
        print(f"   JSON解析失败: {e}")
    
    print("\n3. 检查get_model_mapping方法:")
    model_mapping_result = config.get_model_mapping()
    print(f"   config.get_model_mapping() = {model_mapping_result}")
    
    print("\n4. 检查get_model_info方法:")
    default_model_info = config.get_model_info('default')
    print(f"   config.get_model_info('default') = {default_model_info}")
    
    print("\n5. 检查别名是否在映射中:")
    print(f"   'default' in config.get_model_mapping() = {'default' in config.get_model_mapping()}")
    
    print("\n6. 检查默认模型名称:")
    print(f"   config.model_name = {config.model_name}")
    
    print("\n7. 检查预加载模型:")
    preload_models = config.get_preload_models()
    print(f"   config.get_preload_models() = {preload_models}")
    
    print("\n8. 检查is_model_preloaded方法:")
    is_default_preloaded = config.is_model_preloaded('default')
    print(f"   config.is_model_preloaded('default') = {is_default_preloaded}")
    
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    main()
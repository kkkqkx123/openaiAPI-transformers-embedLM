#!/usr/bin/env python3
"""调试脚本：检查config.model_name和get_model_info方法的行为"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emb_model_provider.core.config import config

def main():
    print("=== 调试 config.model_name 和 get_model_info 方法 ===")
    
    # 1. 检查 config.model_name 的值
    print(f"1. config.model_name 值: {repr(config.model_name)}")
    print(f"   类型: {type(config.model_name)}")
    
    # 2. 检查 get_model_info("default") 的结果
    default_config = config.get_model_info("default")
    print(f"2. get_model_info('default') 结果: {default_config}")
    
    # 3. 检查 get_model_info(config.model_name) 的结果
    model_name_config = config.get_model_info(config.model_name)
    print(f"3. get_model_info(config.model_name) 结果: {model_name_config}")
    
    # 4. 检查 model_name 是否在模型映射中
    model_mapping = config.get_model_mapping()
    print(f"4. model_name 是否在映射中: {config.model_name in model_mapping}")
    
    # 5. 检查 model_name 是否等于 "default"
    print(f"5. model_name == 'default': {config.model_name == 'default'}")
    
    # 6. 检查 model_name 是否等于映射中 default 模型的 name
    if "default" in model_mapping:
        default_model_name = model_mapping["default"]["name"]
        print(f"6. model_name == default模型name: {config.model_name == default_model_name}")
        print(f"   默认模型name: {default_model_name}")
    
    # 7. 测试 ModelManager 构造函数中的逻辑
    test_alias = "default"
    result_alias = test_alias or config.model_name
    print(f"7. 'default' or config.model_name 结果: {result_alias}")
    
    # 8. 检查 get_model_info 对结果的处理
    result_config = config.get_model_info(result_alias)
    print(f"8. get_model_info({repr(result_alias)}) 结果: {result_config}")
    
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    main()
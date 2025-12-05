#!/usr/bin/env python3
"""
检查模型配置状态的诊断脚本
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def check_config():
    try:
        from emb_model_provider.core.config import config
        print('=== 模型映射配置 ===')
        model_mapping = config.get_model_mapping()
        for model_name, model_info in model_mapping.items():
            print(f'{model_name}: {model_info.get("name", "N/A")}')
            print(f'  路径: {model_info.get("path", "N/A")}')
            print(f'  源: {model_info.get("source", "N/A")}')
            print()
        
        print('=== 预加载模型 ===')
        print(f'预加载列表: {config.get_preload_models()}')
        
        print('=== 默认模型信息 ===')
        print(f'默认模型名称: {config.model_name}')
        
        print('\n=== 检查模型路径 ===')
        default_model_info = config.get_model_info('default')
        if default_model_info:
            default_path = default_model_info.get('path', '')
            print(f'Default模型路径: {default_path}')
            print(f'路径存在: {os.path.exists(default_path) if default_path else "N/A"}')
        
    except Exception as e:
        print(f'配置检查失败: {e}')
        import traceback
        traceback.print_exc()

def test_model_manager():
    try:
        from emb_model_provider.core.model_manager import get_model_manager
        print('\n=== 测试模型管理器 ===')
        
        model_manager = get_model_manager('default')
        print(f'模型管理器: {type(model_manager)}')
        
        # 检查是否已加载
        print(f'模型是否已加载: {model_manager.is_model_loaded()}')
        
    except Exception as e:
        print(f'模型管理器测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_config()
    test_model_manager()
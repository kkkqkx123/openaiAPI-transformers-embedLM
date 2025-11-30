"""
测试预加载配置功能的脚本
"""
import os
import sys
from emb_model_provider.core.config import Config
from emb_model_provider.core.model_manager import preload_models, get_model_manager, unload_all_models


def test_preload_functionality():
    """测试预加载功能"""
    print("测试预加载配置功能...")
    
    # 创建测试配置
    os.environ['EMB_PROVIDER_PRELOAD_MODELS'] = "all-MiniLM-L12-v2,test-model"
    os.environ['EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING'] = "false"
    os.environ['EMB_PROVIDER_MODEL_MAPPING'] = '{"all-MiniLM-L12-v2": {"name": "sentence-transformers/all-MiniLM-L12-v2", "path": "", "source": "transformers"}, "test-model": {"name": "sentence-transformers/all-MiniLM-L6-v2", "path": "", "source": "transformers"}}'
    
    # 重新加载配置
    config = Config()
    
    print(f"预加载模型列表: {config.get_preload_models()}")
    print(f"all-MiniLM-L12-v2 是否在预加载列表: {config.is_model_preloaded('all-MiniLM-L12-v2')}")
    print(f"unknown-model 是否在预加载列表: {config.is_model_preloaded('unknown-model')}")
    
    # 测试预加载功能
    try:
        print("开始预加载模型...")
        preload_models()
        print("预加载完成")
    except Exception as e:
        print(f"预加载失败: {e}")
    
    # 测试获取预加载的模型
    try:
        manager = get_model_manager('all-MiniLM-L12-v2')
        print(f"成功获取预加载模型: all-MiniLM-L12-v2")
    except Exception as e:
        print(f"获取预加载模型失败: {e}")
    
    # 测试获取非预加载的模型（应该失败）
    try:
        manager = get_model_manager('unknown-model')
        print(f"成功获取非预加载模型: unknown-model (这不应该发生)")
    except Exception as e:
        print(f"正确地拒绝了非预加载模型: {e}")
    
    # 清理
    unload_all_models()
    print("测试完成")


def test_dynamic_loading_enabled():
    """测试启用动态加载的情况"""
    print("\n测试启用动态加载的情况...")
    
    # 启用动态加载
    os.environ['EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING'] = "true"
    os.environ['EMB_PROVIDER_PRELOAD_MODELS'] = "all-MiniLM-L12-v2"
    
    # 重新加载配置
    config = Config()
    
    print(f"动态加载是否启用: {config.enable_dynamic_model_loading}")
    print(f"all-MiniLM-L12-v2 是否在预加载列表: {config.is_model_preloaded('all-MiniLM-L12-v2')}")
    print(f"unknown-model 是否在预加载列表: {config.is_model_preloaded('unknown-model')}")
    
    # 在动态加载启用的情况下，获取非预加载模型应该成功（如果模型存在）
    try:
        # 注意：这里只是测试配置逻辑，实际模型可能不存在
        print("在动态加载启用的情况下，任何模型理论上都可以加载（如果存在）")
    except Exception as e:
        print(f"获取模型时出错: {e}")


if __name__ == "__main__":
    test_preload_functionality()
    test_dynamic_loading_enabled()
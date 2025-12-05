#!/usr/bin/env python3
"""
模型映射配置诊断脚本
用于精准定位模型映射JSON解析问题
"""

import json
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from emb_model_provider.core.config import Config


def test_json_parsing():
    """测试JSON解析功能"""
    print("=== JSON解析测试 ===")
    
    # 测试1: 从环境变量读取的JSON字符串
    test_json = '''{
    "mini": {
        "name": "sentence-transformers/all-MiniLM-L12-v2",
        "model_path": "D:\\models\\all-MiniLM-L12-v2",
        "source": "huggingface",
        "precision": "fp16"
    },
    "jina-code": {
        "name": "jinaai/jina-code-embeddings-0.5b",
        "model_path": "D:\\models\\jina-code-emb-0.5b",
        "source": "local",
        "precision": "fp16"
    }
}'''
    
    print("测试JSON字符串:")
    print(test_json)
    print()
    
    try:
        parsed = json.loads(test_json)
        print("✓ JSON解析成功")
        print(f"解析结果类型: {type(parsed)}")
        print(f"包含的模型: {list(parsed.keys())}")
        
        if "jina-code" in parsed:
            print(f"jina-code配置: {parsed['jina-code']}")
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON解析失败: {e}")
        print(f"错误位置: 行{e.lineno}, 列{e.colno}")
        return False
    
    return True


def test_env_file_parsing():
    """测试.env文件中的JSON解析"""
    print("\n=== .env文件解析测试 ===")
    
    env_file = project_root / ".env"
    if not env_file.exists():
        print("✗ .env文件不存在")
        return False
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找MODEL_MAPPING配置
        lines = content.split('\n')
        model_mapping_lines = []
        in_mapping = False
        
        for line in lines:
            if 'EMB_PROVIDER_MODEL_MAPPING' in line:
                in_mapping = True
                
            if in_mapping:
                model_mapping_lines.append(line)
                if line.strip() == "}'" and line.count("}'") == line.count("{") + 1:
                    break
        
        if model_mapping_lines:
            # 提取JSON内容
            first_line = model_mapping_lines[0]
            json_start = first_line.find("'") + 1
            
            json_content = first_line[json_start:]
            for line in model_mapping_lines[1:]:
                json_content += "\n" + line
            
            # 移除末尾的单引号
            if json_content.endswith("'"):
                json_content = json_content[:-1]
            
            print("从.env文件提取的JSON:")
            print(json_content)
            print()
            
            try:
                parsed = json.loads(json_content)
                print("✓ .env文件JSON解析成功")
                print(f"包含的模型: {list(parsed.keys())}")
                return True
            except json.JSONDecodeError as e:
                print(f"✗ .env文件JSON解析失败: {e}")
                print(f"错误位置: 行{e.lineno}, 列{e.colno}")
                return False
        else:
            print("✗ 未找到MODEL_MAPPING配置")
            return False
            
    except Exception as e:
        print(f"✗ 读取.env文件失败: {e}")
        return False


def test_config_class():
    """测试Config类的模型映射功能"""
    print("\n=== Config类模型映射测试 ===")
    
    try:
        # 创建配置实例
        config = Config()
        
        print(f"配置类创建成功")
        print(f"model_mapping字段值: {config.model_mapping[:100]}...")
        
        # 测试get_model_mapping方法
        mapping = config.get_model_mapping()
        print(f"get_model_mapping()返回: {type(mapping)}")
        print(f"映射内容: {mapping}")
        
        if mapping:
            print(f"✓ 模型映射包含 {len(mapping)} 个模型")
            for alias, info in mapping.items():
                print(f"  - {alias}: {info}")
        else:
            print("✗ 模型映射为空")
        
        # 测试特定模型
        jina_info = config.get_model_info("jina-code")
        print(f"jina-code模型信息: {jina_info}")
        
        # 测试预加载模型
        preload_models = config.get_preload_models()
        print(f"预加载模型: {preload_models}")
        
        return len(mapping) > 0
        
    except Exception as e:
        print(f"✗ Config类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_config_from_env():
    """测试从环境变量加载的真实配置"""
    print("\n=== 真实环境配置测试 ===")
    
    try:
        # 模拟环境变量
        os.environ['EMB_PROVIDER_MODEL_MAPPING'] = '''{
    "jina-code": {
        "name": "jinaai/jina-code-embeddings-0.5b",
        "model_path": "D:\\models\\jina-code-emb-0.5b",
        "source": "local",
        "precision": "fp16"
    }
}'''
        os.environ['EMB_PROVIDER_PRELOAD_MODELS'] = 'jina-code'
        
        config = Config.from_env()
        mapping = config.get_model_mapping()
        
        print(f"环境变量配置映射: {mapping}")
        
        success = "jina-code" in mapping
        if success:
            print("✓ 环境变量配置成功加载jina-code模型")
        else:
            print("✗ 环境变量配置未加载jina-code模型")
        
        return success
        
    except Exception as e:
        print(f"✗ 环境变量配置测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("模型映射配置诊断测试")
    print("=" * 50)
    
    results = []
    
    # 运行所有测试
    results.append(("JSON解析", test_json_parsing()))
    results.append((".env文件解析", test_env_file_parsing()))
    results.append(("Config类功能", test_config_class()))
    results.append(("环境变量配置", test_real_config_from_env()))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n诊断结论:")
    if all_passed:
        print("✓ 所有测试通过 - 模型映射配置功能正常")
    else:
        print("✗ 部分测试失败 - 需要修复配置问题")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
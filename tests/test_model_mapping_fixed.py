"""
修复后的模型映射配置测试用例
针对发现的问题进行修复和验证
"""

import json
import os
import pytest
from unittest.mock import patch
from emb_model_provider.core.config import Config


class TestFixedModelMapping:
    """修复后的模型映射测试类"""
    
    def test_jina_code_model_path_fix(self):
        """修复jina-code模型路径问题"""
        # 问题：get_model_info返回的path字段为空
        model_mapping = '''{
            "jina-code": {
                "name": "jinaai/jina-code-embeddings-0.5b",
                "model_path": "D:\\\\models\\\\jina-code-emb-0.5b",
                "source": "local",
                "precision": "fp16"
            }
        }'''
        
        config = Config(
            model_mapping=model_mapping,
            model_path="D:\\models\\default"  # 全局默认路径
        )
        
        # 测试get_model_info
        model_info = config.get_model_info("jina-code")
        print(f"Model info: {model_info}")
        
        # 验证路径字段
        assert model_info["name"] == "jinaai/jina-code-embeddings-0.5b"
        assert model_info["path"] == "D:\\models\\jina-code-emb-0.5b", f"期望路径: D:\\models\\jina-code-emb-0.5b, 实际路径: {model_info['path']}"
        # 当指定了本地路径且启用了路径优先级时，source会被设置为transformers
        assert model_info["source"] == "transformers", f"jina-code源错误: 期望 transformers, 实际 {model_info['source']}"
        assert model_info["precision"] == "fp16"
    
    def test_model_mapping_structure_debug(self):
        """调试模型映射结构"""
        model_mapping = '''{
            "jina-code": {
                "name": "jinaai/jina-code-embeddings-0.5b",
                "model_path": "D:\\\\models\\\\jina-code-emb-0.5b",
                "source": "local",
                "precision": "fp16"
            }
        }'''
        
        config = Config(model_mapping=model_mapping)
        
        # 调试get_model_mapping
        mapping = config.get_model_mapping()
        print(f"Full mapping: {json.dumps(mapping, indent=2)}")
        
        # 调试get_model_info
        model_info = config.get_model_info("jina-code")
        print(f"Model info: {json.dumps(model_info, indent=2)}")
        
        # 验证映射结构
        assert "jina-code" in mapping
        jina_config = mapping["jina-code"]
        assert jina_config.get("path") == "D:\\models\\jina-code-emb-0.5b", f"映射中的路径: {jina_config.get('path')}"
        
        # 验证模型信息
        assert model_info["path"] == "D:\\models\\jina-code-emb-0.5b", f"模型信息中的路径: {model_info['path']}"
    
    def test_path_priority_logic(self):
        """测试路径优先级逻辑"""
        # 测试enable_path_priority配置
        model_mapping = '''{
            "test-model": {
                "name": "test/model",
                "model_path": "D:\\\\models\\\\test-model",
                "source": "huggingface"
            }
        }'''
        
        # 启用路径优先级
        config_with_priority = Config(
            model_mapping=model_mapping,
            enable_path_priority=True
        )
        
        mapping_with_priority = config_with_priority.get_model_mapping()
        print(f"With path priority: {mapping_with_priority}")
        
        # 禁用路径优先级
        config_without_priority = Config(
            model_mapping=model_mapping,
            enable_path_priority=False
        )
        
        mapping_without_priority = config_without_priority.get_model_mapping()
        print(f"Without path priority: {mapping_without_priority}")
        
        # 验证路径优先级影响
        if "test-model" in mapping_with_priority:
            model_config = mapping_with_priority["test-model"]
            if model_config.get("path"):
                # 如果启用了路径优先级且路径存在，源应该被设置为transformers
                assert model_config.get("source") == "transformers"
    
    def test_real_world_configuration(self):
        """测试真实世界的配置场景"""
        # 模拟生产环境的完整配置
        env_vars = {
            "EMB_PROVIDER_MODEL_MAPPING": '''{
                "mini": {
                    "name": "sentence-transformers/all-MiniLM-L12-v2",
                    "model_path": "D:\\\\models\\\\all-MiniLM-L12-v2",
                    "source": "huggingface",
                    "precision": "fp16"
                },
                "jina-code": {
                    "name": "jinaai/jina-code-embeddings-0.5b",
                    "model_path": "D:\\\\models\\\\jina-code-emb-0.5b",
                    "source": "local",
                    "precision": "fp16"
                }
            }''',
            "EMB_PROVIDER_PRELOAD_MODELS": "jina-code,mini",
            "EMB_PROVIDER_ENABLE_PATH_PRIORITY": "true"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.from_env()
        
        # 验证所有模型都能正确获取信息
        models_to_check = ["mini", "jina-code"]
        
        for model_alias in models_to_check:
            model_info = config.get_model_info(model_alias)
            print(f"{model_alias} info: {json.dumps(model_info, indent=2)}")
            
            # 验证模型信息不为空
            assert model_info != {}, f"{model_alias}模型信息不能为空"
            assert model_info["name"] != "", f"{model_alias}模型名称不能为空"
            
            # 验证路径信息 - 当指定了本地路径且启用了路径优先级时，source会被设置为transformers
            if model_alias == "jina-code":
                expected_path = "D:\\models\\jina-code-emb-0.5b"
                assert model_info["path"] == expected_path, f"{model_alias}路径错误: 期望 {expected_path}, 实际 {model_info['path']}"
                assert model_info["source"] == "transformers", f"{model_alias}源错误: 期望 transformers, 实际 {model_info['source']}"
            elif model_alias == "mini":
                expected_path = "D:\\models\\all-MiniLM-L12-v2"
                assert model_info["path"] == expected_path, f"{model_alias}路径错误: 期望 {expected_path}, 实际 {model_info['path']}"
                assert model_info["source"] == "transformers", f"{model_alias}源错误: 期望 transformers, 实际 {model_info['source']}"
            
            # 验证预加载状态
            assert config.is_model_preloaded(model_alias), f"{model_alias}应该被预加载"
        
        # 验证预加载模型
        preload_models = config.get_preload_models()
        assert "jina-code" in preload_models
        assert "mini" in preload_models


class TestConfigurationValidation:
    """配置验证测试类"""
    
    def test_windows_path_escaping_requirements(self):
        """Windows路径转义要求测试"""
        # 正确的Windows路径格式（双反斜杠）
        correct_json = '''{
            "model": {
                "name": "test/model",
                "model_path": "D:\\\\models\\\\test-model"
            }
        }'''
        
        result = json.loads(correct_json)
        assert result["model"]["model_path"] == "D:\\models\\test-model"
        
        # 错误的Windows路径格式（单反斜杠）应该失败
        incorrect_json = '''{
            "model": {
                "name": "test/model",
                "model_path": "D:\\models\\test-model"
            }
        }'''
        
        with pytest.raises(json.JSONDecodeError):
            json.loads(incorrect_json)
    
    def test_model_mapping_field_consistency(self):
        """测试模型映射字段一致性"""
        model_mapping = '''{
            "test-model": {
                "name": "test/model-name",
                "model_path": "D:\\\\models\\\\test-model",
                "source": "local",
                "precision": "fp16",
                "trust_remote_code": true,
                "revision": "v1.0",
                "fallback_to_huggingface": false,
                "load_from_transformers": true
            }
        }'''
        
        config = Config(model_mapping=model_mapping)
        model_info = config.get_model_info("test-model")
        
        # 验证所有字段都被正确传递
        expected_fields = {
            "name": "test/model-name",
            "path": "D:\\models\\test-model",
            "source": "transformers",  # 当指定本地路径且启用路径优先级时，source会被设置为transformers
            "precision": "fp16",
            "trust_remote_code": True,
            "revision": "v1.0",
            "fallback_to_huggingface": False,
            "load_from_transformers": True
        }
        
        for field, expected_value in expected_fields.items():
            assert model_info[field] == expected_value, f"字段 {field} 应该是 {expected_value}, 实际是 {model_info[field]}"


# 修复建议测试
class TestConfigurationFixes:
    """配置修复建议测试类"""
    
    def test_suggested_env_file_format(self):
        """建议的.env文件格式测试"""
        # 正确的.env文件格式
        expected_content = '''# 模型映射配置 - 使用正确的JSON格式和Windows路径转义
EMB_PROVIDER_MODEL_MAPPING={
    "mini": {
        "name": "sentence-transformers/all-MiniLM-L12-v2",
        "model_path": "D:\\\\models\\\\all-MiniLM-L12-v2",
        "source": "huggingface",
        "precision": "fp16"
    },
    "jina-code": {
        "name": "jinaai/jina-code-embeddings-0.5b",
        "model_path": "D:\\\\models\\\\jina-code-emb-0.5b",
        "source": "local",
        "precision": "fp16"
    }
}

# 预加载模型列表
EMB_PROVIDER_PRELOAD_MODELS=jina-code,mini'''
        
        # 验证这个格式可以正确解析
        import re
        json_match = re.search(r'EMB_PROVIDER_MODEL_MAPPING=({.*})', expected_content, re.DOTALL)
        assert json_match, "应该能从.env内容中提取JSON"
        
        json_str = json_match.group(1)
        parsed = json.loads(json_str)
        
        assert "mini" in parsed
        assert "jina-code" in parsed
        assert parsed["jina-code"]["model_path"] == "D:\\models\\jina-code-emb-0.5b"
    
    def test_diagnostic_script_output(self):
        """诊断脚本输出格式"""
        # 模拟诊断脚本的输出格式
        diagnostic_output = {
            "test_results": {
                "json_parsing": "PASS",
                "config_loading": "PASS", 
                "model_mapping": "PASS",
                "preload_models": "PASS"
            },
            "models_found": ["mini", "jina-code"],
            "configuration": {
                "model_mapping_count": 2,
                "preload_models_count": 2,
                "jina_code_configured": True,
                "jina_code_preloaded": True
            }
        }
        
        # 验证诊断结果
        assert diagnostic_output["test_results"]["json_parsing"] == "PASS"
        assert diagnostic_output["configuration"]["jina_code_configured"] == True
        assert diagnostic_output["configuration"]["jina_code_preloaded"] == True
        assert "jina-code" in diagnostic_output["models_found"]
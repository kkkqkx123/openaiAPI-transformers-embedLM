"""
精准的模型映射JSON解析测试用例
用于验证和修复模型映射配置问题
"""

import json
import os
import pytest
from unittest.mock import patch
from emb_model_provider.core.config import Config


class TestModelMappingPrecision:
    """精准的模型映射JSON解析测试类"""
    
    def test_windows_path_json_parsing(self):
        """测试Windows路径格式的JSON解析"""
        # 测试1: 正确的Windows路径格式（双反斜杠）
        json_str = '''{
            "jina-code": {
                "name": "jinaai/jina-code-embeddings-0.5b",
                "model_path": "D:\\\\models\\\\jina-code-emb-0.5b",
                "source": "local",
                "precision": "fp16"
            }
        }'''
        
        result = json.loads(json_str)
        assert "jina-code" in result
        assert result["jina-code"]["model_path"] == "D:\\models\\jina-code-emb-0.5b"
    
    def test_windows_path_single_backslash_fails(self):
        """测试单反斜杠的Windows路径应该失败"""
        json_str = '''{
            "jina-code": {
                "name": "jinaai/jina-code-embeddings-0.5b",
                "model_path": "D:\\models\\jina-code-emb-0.5b",
                "source": "local",
                "precision": "fp16"
            }
        }'''
        
        with pytest.raises(json.JSONDecodeError):
            json.loads(json_str)
    
    def test_config_with_windows_paths(self):
        """测试Config类处理Windows路径"""
        model_mapping = '''{
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
        }'''
        
        config = Config(model_mapping=model_mapping)
        mapping = config.get_model_mapping()
        
        # 验证映射正确加载
        assert "mini" in mapping
        assert "jina-code" in mapping
        
        # 验证Windows路径正确解析
        assert mapping["mini"]["model_path"] == "D:\\models\\all-MiniLM-L12-v2"
        assert mapping["jina-code"]["model_path"] == "D:\\models\\jina-code-emb-0.5b"
    
    def test_config_from_env_with_json(self):
        """测试从环境变量加载JSON配置"""
        env_vars = {
            "EMB_PROVIDER_MODEL_MAPPING": '''{
                "test-model": {
                    "name": "test/model-name",
                    "model_path": "D:\\\\models\\\\test-model",
                    "source": "local",
                    "precision": "fp16"
                }
            }''',
            "EMB_PROVIDER_PRELOAD_MODELS": "test-model"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.from_env()
            mapping = config.get_model_mapping()
            preload_models = config.get_preload_models()
        
        # 验证模型映射
        assert "test-model" in mapping
        assert mapping["test-model"]["name"] == "test/model-name"
        assert mapping["test-model"]["model_path"] == "D:\\models\\test-model"
        
        # 验证预加载模型
        assert "test-model" in preload_models
    
    def test_jina_code_specific_config(self):
        """专门针对jina-code模型的配置测试"""
        model_mapping = '''{
            "jina-code": {
                "name": "jinaai/jina-code-embeddings-0.5b",
                "model_path": "D:\\\\models\\\\jina-code-emb-0.5b",
                "source": "local",
                "precision": "fp16",
                "trust_remote_code": false,
                "revision": "main",
                "fallback_to_huggingface": true,
                "load_from_transformers": false
            }
        }'''
        
        config = Config(
            model_mapping=model_mapping,
            preload_models="jina-code"
        )
        
        # 测试模型映射
        mapping = config.get_model_mapping()
        assert "jina-code" in mapping
        
        jina_config = mapping["jina-code"]
        assert jina_config["name"] == "jinaai/jina-code-embeddings-0.5b"
        assert jina_config["path"] == "D:\\models\\jina-code-emb-0.5b"
        # 当指定了本地路径且启用了路径优先级时，source会被设置为transformers
        assert jina_config["source"] == "transformers", f"期望 transformers, 实际 {jina_config.get('source')}"
        assert jina_config["precision"] == "fp16"
        assert jina_config["trust_remote_code"] is False
        assert jina_config["revision"] == "main"
        assert jina_config["fallback_to_huggingface"] is True
        assert jina_config["load_from_transformers"] is False
        
        # 测试模型信息获取
        model_info = config.get_model_info("jina-code")
        assert model_info["name"] == "jinaai/jina-code-embeddings-0.5b"
        assert model_info["path"] == "D:\\models\\jina-code-emb-0.5b"
        
        # 测试预加载状态
        assert config.is_model_preloaded("jina-code") == True
        assert "jina-code" in config.get_preload_models()
    
    def test_invalid_json_handling(self):
        """测试无效JSON的处理"""
        # 测试1: 语法错误的JSON
        invalid_json = '''{
            "model": {
                "name": "test/model",
                "model_path": "D:\\invalid"  # 错误的转义
            }
        }'''
        
        config = Config(model_mapping=invalid_json)
        mapping = config.get_model_mapping()
        
        # 应该返回空字典而不是抛出异常
        assert mapping == {}
    
    def test_empty_and_default_values(self):
        """测试空值和默认值处理"""
        # 测试空映射
        config = Config(model_mapping="")
        assert config.get_model_mapping() == {}
        
        # 测试默认映射字符串
        config = Config(model_mapping="{}")
        assert config.get_model_mapping() == {}
        
        # 测试不存在的模型
        config = Config(model_mapping='{"existing": "model/name"}')
        assert config.get_model_info("nonexistent") == {}
    
    def test_model_source_inheritance(self):
        """测试模型源继承"""
        model_mapping = '''{
            "model1": {
                "name": "test/model1"
            },
            "model2": {
                "name": "test/model2",
                "source": "modelscope"
            }
        }'''
        
        config = Config(
            model_mapping=model_mapping,
            model_source="huggingface"  # 全局默认源
        )
        
        mapping = config.get_model_mapping()
        
        # model1应该继承全局源
        assert mapping["model1"]["source"] == "huggingface"
        
        # model2应该使用自己的源
        assert mapping["model2"]["source"] == "modelscope"
    
    def test_precision_configuration(self):
        """测试精度配置"""
        model_mapping = '''{
            "fp16-model": {
                "name": "test/fp16-model",
                "precision": "fp16"
            },
            "auto-model": {
                "name": "test/auto-model"
            }
        }'''
        
        config = Config(
            model_mapping=model_mapping,
            model_precision="fp32"  # 全局默认精度
        )
        
        mapping = config.get_model_mapping()
        
        # 指定精度的模型
        assert mapping["fp16-model"]["precision"] == "fp16"
        
        # 未指定精度的模型应该使用全局精度
        assert mapping["auto-model"]["precision"] == "fp32"


class TestModelMappingIntegration:
    """模型映射集成测试"""
    
    def test_full_integration_scenario(self):
        """测试完整的集成场景"""
        # 模拟完整的生产环境配置
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
            "EMB_PROVIDER_ENABLE_DYNAMIC_MODEL_LOADING": "false",
            "EMB_PROVIDER_MODEL_SOURCE": "huggingface"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.from_env()
        
        # 验证完整配置
        mapping = config.get_model_mapping()
        assert len(mapping) == 2
        assert "mini" in mapping
        assert "jina-code" in mapping
        
        # 验证预加载模型
        preload_models = config.get_preload_models()
        assert "jina-code" in preload_models
        assert "mini" in preload_models
        
        # 验证动态加载设置
        assert config.enable_dynamic_model_loading == False
        
        # 验证模型信息完整性
        jina_info = config.get_model_info("jina-code")
        assert jina_info["name"] == "jinaai/jina-code-embeddings-0.5b"
        assert jina_info["path"] == "D:\\models\\jina-code-emb-0.5b"
        # 当指定了本地路径且启用了路径优先级时，source会被设置为transformers
        assert jina_info["source"] == "transformers", f"期望 transformers, 实际 {jina_info['source']}"
        assert jina_info["precision"] == "fp16"


# 验收标准测试
class TestAcceptanceCriteria:
    """验收标准测试类"""
    
    def test_jina_code_model_loading_criteria(self):
        """
        验收标准1: jina-code模型必须能被正确加载和识别
        """
        # 给定配置
        model_mapping = '''{
            "jina-code": {
                "name": "jinaai/jina-code-embeddings-0.5b",
                "model_path": "D:\\\\models\\\\jina-code-emb-0.5b",
                "source": "local",
                "precision": "fp16"
            }
        }'''
        
        # 当创建配置时
        config = Config(model_mapping=model_mapping, preload_models="jina-code")
        
        # 那么应该满足以下条件
        mapping = config.get_model_mapping()
        assert "jina-code" in mapping, "jina-code模型必须在映射中"
        
        model_info = config.get_model_info("jina-code")
        assert model_info != {}, "必须能获取jina-code模型信息"
        assert model_info["name"] == "jinaai/jina-code-embeddings-0.5b", "模型名称必须正确"
        assert model_info["path"] == "D:\\models\\jina-code-emb-0.5b", "模型路径必须正确"
        
        assert config.is_model_preloaded("jina-code"), "jina-code模型必须被预加载"
    
    def test_windows_path_handling_criteria(self):
        """
        验收标准2: Windows路径必须被正确处理
        """
        # 给定Windows路径配置
        model_mapping = '''{
            "test-model": {
                "name": "test/model",
                "model_path": "D:\\\\models\\\\test-model",
                "source": "local"
            }
        }'''
        
        # 当解析配置时
        config = Config(model_mapping=model_mapping)
        mapping = config.get_model_mapping()
        
        # 那么路径必须正确解析
        assert mapping["test-model"]["model_path"] == "D:\\models\\test-model", "Windows路径必须正确解析"
    
    def test_json_error_handling_criteria(self):
        """
        验收标准3: 无效的JSON必须被优雅处理
        """
        # 给定无效的JSON
        invalid_configs = [
            '''{"model": {"path": "D:\\invalid"}}''',  # 错误的转义
            '''{"model": {"path": "D:\models\test"}}''',  # 单斜杠
            '''{"model": {invalid json}}''',  # 语法错误
        ]
        
        for invalid_json in invalid_configs:
            # 当使用无效配置时
            config = Config(model_mapping=invalid_json)
            mapping = config.get_model_mapping()
            
            # 那么应该返回空映射而不是崩溃
            assert mapping == {}, f"无效JSON应该返回空映射: {invalid_json}"
    
    def test_preload_models_criteria(self):
        """
        验收标准4: 预加载模型配置必须正确工作
        """
        # 给定预加载配置
        model_mapping = '''{
            "model1": {"name": "test/model1"},
            "model2": {"name": "test/model2"}
        }'''
        
        config = Config(
            model_mapping=model_mapping,
            preload_models="model1,model2"
        )
        
        # 那么预加载模型列表必须正确
        preload_models = config.get_preload_models()
        assert "model1" in preload_models, "model1必须在预加载列表中"
        assert "model2" in preload_models, "model2必须在预加载列表中"
        assert len(preload_models) == 2, "预加载列表必须包含2个模型"
        
        # 并且预加载检查必须正确
        assert config.is_model_preloaded("model1") == True
        assert config.is_model_preloaded("model2") == True
        assert config.is_model_preloaded("model3") == False
"""
端到端测试，验证完整的 API 流程
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any
from fastapi.testclient import TestClient
from emb_model_provider.main import app
from emb_model_provider.core.config import config


@pytest.fixture
def client():
    """创建测试客户端"""
    # 在e2e测试中直接禁用动态批处理
    from emb_model_provider.core.config import config
    original_value = config.enable_dynamic_batching
    config.enable_dynamic_batching = False
    
    client = TestClient(app)
    yield client
    
    # 恢复原始配置
    config.enable_dynamic_batching = original_value


class TestE2EAPIFlow:
    """端到端 API 流程测试"""
    
    def test_complete_embedding_flow(self, client):
        """测试完整的嵌入流程 - 从请求到响应"""
        print("=== 开始测试 ===")
        
        # 1. 检查服务健康状态
        print("1. 检查健康状态...")
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        print("✓ 健康检查通过")
        
        # 2. 获取可用模型列表
        print("2. 获取模型列表...")
        models_response = client.get("/v1/models")
        assert models_response.status_code == 200
        models_data = models_response.json()
        assert "data" in models_data
        assert len(models_data["data"]) >= 1
        print(f"✓ 找到 {len(models_data['data'])} 个模型")
        
        # 获取模型ID
        model_id = models_data["data"][0]["id"]
        print(f"3. 使用模型: {model_id}")
        
        # 3. 发送单个文本嵌入请求
        print("4. 发送单个嵌入请求...")
        single_request = {
            "input": "This is a test sentence for embedding.",
            "model": model_id
        }
        
        embed_response = client.post("/v1/embeddings", json=single_request)
        assert embed_response.status_code == 200
        print("✓ 单个嵌入请求完成")
        
        embed_data = embed_response.json()
        assert "data" in embed_data
        assert "model" in embed_data
        assert "usage" in embed_data
        assert len(embed_data["data"]) == 1
        assert len(embed_data["data"][0]["embedding"]) == config.embedding_dimension
        assert embed_data["model"] == model_id
        assert embed_data["usage"]["prompt_tokens"] > 0
        assert embed_data["usage"]["total_tokens"] > 0
        
        # 4. 发送批量文本嵌入请求
        batch_request = {
            "input": [
                "First test sentence.",
                "Second test sentence.",
                "Third test sentence."
            ],
            "model": model_id
        }
        
        batch_response = client.post("/v1/embeddings", json=batch_request)
        assert batch_response.status_code == 200
        
        batch_data = batch_response.json()
        assert "data" in batch_data
        assert "model" in batch_data
        assert "usage" in batch_data
        assert len(batch_data["data"]) == 3
        assert all(len(item["embedding"]) == config.embedding_dimension for item in batch_data["data"])
        assert batch_data["model"] == model_id
        assert batch_data["usage"]["prompt_tokens"] > 0
        assert batch_data["usage"]["total_tokens"] > 0
        
        # 5. 验证嵌入向量的唯一性（不同文本应该产生不同的嵌入）
        embeddings = [item["embedding"] for item in batch_data["data"]]
        assert embeddings[0] != embeddings[1] != embeddings[2]
    
    def test_error_handling_flow(self, client):
        """测试错误处理流程"""
        # 1. 测试空输入错误
        empty_request = {
            "input": "",
            "model": config.model_name
        }
        
        response = client.post("/v1/embeddings", json=empty_request)
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["type"] == "invalid_request_error"
        assert "Input cannot be empty" in error_data["error"]["message"]
        
        # 2. 测试模型不存在错误
        invalid_model_request = {
            "input": "Test input",
            "model": "non-existent-model"
        }
        
        response = client.post("/v1/embeddings", json=invalid_model_request)
        assert response.status_code == 404
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["type"] == "model_not_found"
        
        # 3. 测试批处理大小超限错误
        large_input = [f"Test input {i}" for i in range(config.max_batch_size + 1)]
        oversized_request = {
            "input": large_input,
            "model": config.model_name
        }
        
        response = client.post("/v1/embeddings", json=oversized_request)
        assert response.status_code == 429
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["type"] == "batch_size_exceeded"
        
        # 4. 测试上下文长度超限错误
        long_text = "word " * (config.max_context_length + 10)
        context_request = {
            "input": long_text,
            "model": config.model_name
        }
        
        response = client.post("/v1/embeddings", json=context_request)
        assert response.status_code == 429
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["type"] == "context_length_exceeded"
    
    def test_request_response_format_compatibility(self, client):
        """测试请求和响应格式与 OpenAI API 的兼容性"""
        # 1. 测试基本请求格式
        basic_request = {
            "input": "Test compatibility with OpenAI format",
            "model": config.model_name
        }
        
        response = client.post("/v1/embeddings", json=basic_request)
        assert response.status_code == 200
        
        # 验证响应格式符合 OpenAI API 规范
        data = response.json()
        required_fields = ["object", "data", "model", "usage"]
        for field in required_fields:
            assert field in data
        
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1
        
        # 验证嵌入数据格式
        embedding_item = data["data"][0]
        assert "object" in embedding_item
        assert "embedding" in embedding_item
        assert "index" in embedding_item
        assert embedding_item["object"] == "embedding"
        assert isinstance(embedding_item["embedding"], list)
        assert all(isinstance(x, float) for x in embedding_item["embedding"])
        assert embedding_item["index"] == 0
        
        # 验证使用情况格式
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "total_tokens" in usage
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
        
        # 2. 测试带可选参数的请求格式
        optional_request = {
            "input": "Test with optional parameters",
            "model": config.model_name,
            "encoding_format": "float",
            "user": "test-user"
        }
        
        response = client.post("/v1/embeddings", json=optional_request)
        assert response.status_code == 200
        
        # 3. 测试批量请求格式
        batch_request = {
            "input": ["First text", "Second text"],
            "model": config.model_name
        }
        
        response = client.post("/v1/embeddings", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) == 2
        assert data["data"][0]["index"] == 0
        assert data["data"][1]["index"] == 1
    
    def test_consistency_across_requests(self, client):
        """测试跨请求的一致性"""
        test_text = "Consistency test sentence"
        
        # 发送相同文本的多个请求
        embeddings = []
        for i in range(3):
            request = {
                "input": test_text,
                "model": config.model_name
            }
            
            response = client.post("/v1/embeddings", json=request)
            assert response.status_code == 200
            
            embedding = response.json()["data"][0]["embedding"]
            embeddings.append(embedding)
        
        # 验证所有嵌入向量相同（浮点数精度允许微小差异）
        for i in range(1, len(embeddings)):
            for j in range(len(embeddings[0])):
                assert abs(embeddings[0][j] - embeddings[i][j]) < 1e-6
    
    def test_performance_characteristics(self, client):
        """测试性能特征"""
        test_texts = [
            "Short text.",
            "This is a medium length text that contains multiple sentences and should take slightly longer to process.",
            "This is a much longer text that contains many words and sentences. It is designed to test how the API handles longer inputs and whether the processing time scales appropriately with the input length. The text should be long enough to make a noticeable difference in processing time compared to shorter inputs."
        ]
        
        # 测试不同长度文本的处理时间
        processing_times = []
        
        for text in test_texts:
            request = {
                "input": text,
                "model": config.model_name
            }
            
            start_time = time.time()
            response = client.post("/v1/embeddings", json=request)
            end_time = time.time()
            
            assert response.status_code == 200
            processing_times.append(end_time - start_time)
        
        # 验证处理时间合理（应该都在几秒内完成）
        for i, processing_time in enumerate(processing_times):
            assert processing_time < 10.0, f"Text {i} took too long to process: {processing_time}s"
        
        # 验证批处理比单独处理更高效
        single_requests_time = 0
        for text in test_texts:
            request = {
                "input": text,
                "model": config.model_name
            }
            
            start_time = time.time()
            response = client.post("/v1/embeddings", json=request)
            end_time = time.time()
            
            assert response.status_code == 200
            single_requests_time += (end_time - start_time)
        
        # 批量请求
        batch_request = {
            "input": test_texts,
            "model": config.model_name
        }
        
        start_time = time.time()
        response = client.post("/v1/embeddings", json=batch_request)
        end_time = time.time()
        
        assert response.status_code == 200
        batch_request_time = end_time - start_time
        
        # 批处理应该比单独处理所有请求更快
        assert batch_request_time < single_requests_time, "Batch processing should be more efficient"


if __name__ == "__main__":
    pytest.main([__file__])
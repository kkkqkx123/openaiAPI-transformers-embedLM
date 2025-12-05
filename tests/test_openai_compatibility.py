"""
兼容性测试，验证与 OpenAI 客户端的兼容性
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from emb_model_provider.main import app
from emb_model_provider.core.config import config


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


class TestOpenAICompatibility:
    """OpenAI 客户端兼容性测试"""
    
    def test_openai_python_client_format(self, client):
        """测试与 OpenAI Python 客户端的请求格式兼容性"""
        # 模拟 OpenAI Python 客户端的请求格式
        openai_format_request = {
            "input": "Test compatibility with OpenAI Python client",
            "model": "default",
            "encoding_format": "float",
            "user": "test-user-123"
        }
        
        response = client.post("/v1/embeddings", json=openai_format_request)
        assert response.status_code == 200
        
        # 验证响应格式符合 OpenAI API 规范
        data = response.json()
        
        # 验证顶级字段
        assert "object" in data
        assert "data" in data
        assert "model" in data
        assert "usage" in data
        
        assert data["object"] == "list"
        assert data["model"] == "default"
        
        # 验证数据对象格式
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1
        
        embedding_object = data["data"][0]
        assert "object" in embedding_object
        assert "embedding" in embedding_object
        assert "index" in embedding_object
        
        assert embedding_object["object"] == "embedding"
        assert isinstance(embedding_object["embedding"], list)
        assert len(embedding_object["embedding"]) == config.embedding_dimension
        assert all(isinstance(x, (float, int)) for x in embedding_object["embedding"])
        assert embedding_object["index"] == 0
        
        # 验证使用情况对象格式
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "total_tokens" in usage
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
        assert usage["prompt_tokens"] > 0
        assert usage["total_tokens"] > 0
    
    def test_openai_batch_request_format(self, client):
        """测试与 OpenAI 批量请求格式的兼容性"""
        # 模拟 OpenAI 客户端的批量请求格式
        batch_request = {
            "input": [
                "First test sentence for batch compatibility",
                "Second test sentence for batch compatibility",
                "Third test sentence for batch compatibility"
            ],
            "model": config.model_name,
            "encoding_format": "float"
        }
        
        response = client.post("/v1/embeddings", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        
        # 验证批量响应格式
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 3
        
        # 验证每个嵌入对象的索引
        for i, embedding_object in enumerate(data["data"]):
            assert embedding_object["index"] == i
            assert embedding_object["object"] == "embedding"
            assert len(embedding_object["embedding"]) == config.embedding_dimension
        
        # 验证使用情况统计
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["total_tokens"] > 0
    
    def test_openai_error_response_format(self, client):
        """测试与 OpenAI 错误响应格式的兼容性"""
        # 测试无效请求的错误响应格式
        invalid_request = {
            "input": "",
            "model": config.model_name
        }
        
        response = client.post("/v1/embeddings", json=invalid_request)
        assert response.status_code == 400
        
        error_data = response.json()
        
        # 验证错误响应格式符合 OpenAI 规范
        assert "error" in error_data
        error = error_data["error"]
        
        assert "message" in error
        assert "type" in error
        assert "param" in error
        assert "code" in error or error["code"] is None
        
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "input"
        assert "Input cannot be empty" in error["message"]
    
    def test_openai_models_endpoint_format(self, client):
        """测试与 OpenAI models 端点响应格式的兼容性"""
        response = client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        
        # 验证 models 端点响应格式
        assert "object" in data
        assert "data" in data
        
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1
        
        # 验证模型对象格式
        model_object = data["data"][0]
        assert "id" in model_object
        assert "object" in model_object
        assert "created" in model_object
        assert "owned_by" in model_object
        
        assert model_object["object"] == "model"
        assert isinstance(model_object["created"], int)
        assert model_object["created"] > 0
    
    def test_openai_client_headers_compatibility(self, client):
        """测试与 OpenAI 客户端请求头的兼容性"""
        # 模拟 OpenAI 客户端发送的请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-test-key",  # 虽然我们不验证，但应该接受
            "User-Agent": "OpenAI/Python/v1.0.0"
        }
        
        request_data = {
            "input": "Test header compatibility",
            "model": config.model_name
        }
        
        response = client.post("/v1/embeddings", json=request_data, headers=headers)
        assert response.status_code == 200
        
        # 验证响应包含适当的头
        assert "content-type" in response.headers
        assert response.headers["content-type"] == "application/json"
    
    def test_mock_openai_client_response_format(self, client):
        """测试 API 响应格式与 OpenAI 客户端期望的一致性"""
        # 预期的 OpenAI API 响应格式
        expected_response_format = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3] + [0.0] * (config.embedding_dimension - 3),
                    "index": 0
                }
            ],
            "model": config.model_name,
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        
        # 这里我们测试我们的 API 是否能产生与 OpenAI 客户端期望的相同格式的响应
        request_data = {
            "input": "Test mock OpenAI client",
            "model": config.model_name
        }
        
        response = client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 200
        
        our_response = response.json()
        
        # 验证我们的响应结构与 OpenAI 客户端期望的一致
        assert our_response["object"] == expected_response_format["object"]
        assert len(our_response["data"]) == len(expected_response_format["data"])
        assert our_response["model"] == expected_response_format["model"]
        assert "usage" in our_response
        
        # 验证数据对象结构
        our_embedding = our_response["data"][0]
        expected_embedding = expected_response_format["data"][0]
        
        assert our_embedding["object"] == expected_embedding["object"]
        assert "embedding" in our_embedding
        assert "index" in our_embedding
        assert len(our_embedding["embedding"]) == config.embedding_dimension
    
    def test_openai_api_version_compatibility(self, client):
        """测试与 OpenAI API 版本的兼容性"""
        # 测试带有 API 版本头的请求
        headers = {
            "OpenAI-Organization": "org-test",
            "OpenAI-Project": "proj-test"
        }
        
        request_data = {
            "input": "Test API version compatibility",
            "model": config.model_name
        }
        
        response = client.post("/v1/embeddings", json=request_data, headers=headers)
        assert response.status_code == 200
        
        # 验证响应格式
        data = response.json()
        assert "object" in data
        assert "data" in data
        assert "model" in data
        assert "usage" in data
    
    def test_openai_encoding_formats(self, client):
        """测试不同的编码格式兼容性"""
        test_text = "Test encoding format compatibility"
        
        # 测试 float 格式（默认）
        float_request = {
            "input": test_text,
            "model": config.model_name,
            "encoding_format": "float"
        }
        
        response = client.post("/v1/embeddings", json=float_request)
        assert response.status_code == 200
        
        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert all(isinstance(x, (float, int)) for x in embedding)
        
        # 注意：我们的实现目前不支持 base64 格式，但应该优雅地处理
        # 这里我们测试如果客户端请求 base64 格式会发生什么
        base64_request = {
            "input": test_text,
            "model": config.model_name,
            "encoding_format": "base64"
        }
        
        # 我们的实现应该忽略不支持的编码格式并返回默认格式
        response = client.post("/v1/embeddings", json=base64_request)
        assert response.status_code == 200
        
        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert all(isinstance(x, (float, int)) for x in embedding)
    
    def test_openai_special_characters_handling(self, client):
        """测试特殊字符处理的兼容性"""
        # 测试包含特殊字符的文本
        special_texts = [
            "Text with emoji 🚀 and symbols #$%",
            "Text with newlines\nand\ttabs",
            "Text with quotes: 'single' and \"double\"",
            "Text with unicode: 中文, ñ, ü, ø",
            "Text with math: ∑∏∫∆∇∂",
            "Text with currency: $€£¥₹"
        ]
        
        for text in special_texts:
            request = {
                "input": text,
                "model": config.model_name
            }
            
            response = client.post("/v1/embeddings", json=request)
            assert response.status_code == 200
            
            data = response.json()
            assert len(data["data"][0]["embedding"]) == config.embedding_dimension
    
    def test_openai_large_request_handling(self, client):
        """测试大请求处理的兼容性"""
        # 创建一个较大的请求，但不超过限制
        large_text = "This is a test sentence. " * 50  # 约800个字符
        
        request = {
            "input": large_text,
            "model": config.model_name
        }
        
        response = client.post("/v1/embeddings", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"][0]["embedding"]) == config.embedding_dimension
        assert data["usage"]["prompt_tokens"] > 0
    
    def test_openai_response_time_consistency(self, client):
        """测试响应时间一致性"""
        test_text = "Test response time consistency"
        
        # 发送多个相同请求，验证响应时间的一致性
        response_times = []
        for _ in range(5):
            request = {
                "input": test_text,
                "model": config.model_name
            }
            
            import time
            start_time = time.time()
            response = client.post("/v1/embeddings", json=request)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        # 验证响应时间在合理范围内波动
        avg_time = sum(response_times) / len(response_times)
        max_deviation = max(abs(t - avg_time) for t in response_times)
        
        # 响应时间变化不应太大（这里设置为平均时间的50%）
        assert max_deviation < avg_time * 0.5, f"Response time variation too large: {max_deviation}s"


if __name__ == "__main__":
    pytest.main([__file__])
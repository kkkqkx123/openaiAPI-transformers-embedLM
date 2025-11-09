import pytest
import asyncio
from fastapi.testclient import TestClient
from emb_model_provider.main import app
from emb_model_provider.core.config import config
from emb_model_provider.api.embeddings import EmbeddingRequest
from emb_model_provider.api.models import ModelInfo


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


def test_embeddings_endpoint(client):
    """测试嵌入端点"""
    request_data = {
        "input": "Hello world",
        "model": config.model_name
    }
    
    response = client.post("/v1/embeddings", json=request_data)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "data" in data
    assert "model" in data
    assert "usage" in data
    assert len(data["data"]) == 1  # 单个输入应该返回一个嵌入
    assert isinstance(data["data"][0]["embedding"], list)
    assert len(data["data"][0]["embedding"]) == config.embedding_dimension  # 检查维度


def test_embeddings_endpoint_with_list_input(client):
    """测试嵌入端点，使用列表输入"""
    request_data = {
        "input": ["Hello world", "Test input"],
        "model": config.model_name
    }
    
    response = client.post("/v1/embeddings", json=request_data)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "data" in data
    assert "model" in data
    assert "usage" in data
    assert len(data["data"]) == 2  # 两个输入应该返回两个嵌入
    assert isinstance(data["data"][0]["embedding"], list)
    assert len(data["data"][0]["embedding"]) == config.embedding_dimension  # 检查维度


def test_embeddings_endpoint_invalid_input(client):
    """测试嵌入端点，使用无效输入"""
    request_data = {
        "input": "",  # 空输入
        "model": config.model_name
    }
    
    response = client.post("/v1/embeddings", json=request_data)
    
    assert response.status_code == 400 # Bad Request
    data = response.json()
    assert "error" in data


def test_models_endpoint(client):
    """测试模型端点"""
    response = client.get("/v1/models")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "data" in data
    assert len(data["data"]) >= 1  # 至少有一个模型
    
    model_info = data["data"][0]
    assert "id" in model_info
    assert "object" in model_info
    assert "created" in model_info
    assert "owned_by" in model_info
    assert model_info["object"] == "model"


def test_health_check_endpoint(client):
    """测试健康检查端点"""
    response = client.get("/health")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_embeddings_endpoint_batch_size_exceeded(client):
    """测试嵌入端点，超出批处理大小限制"""
    # 创建超过最大批处理大小的输入
    large_input = [f"Test input {i}" for i in range(config.max_batch_size + 1)]
    
    request_data = {
        "input": large_input,
        "model": config.model_name
    }
    
    response = client.post("/v1/embeddings", json=request_data)
    
    # 应该返回429错误，因为批处理大小超限
    assert response.status_code == 429
    data = response.json()
    assert "error" in data
    assert "batch_size_exceeded" in data["error"]["type"]


def test_embeddings_endpoint_context_length_exceeded(client):
    """测试嵌入端点，超出上下文长度限制"""
    # 创建一个非常长的输入，超过最大上下文长度
    long_text = "test " * (config.max_context_length + 10)
    
    request_data = {
        "input": long_text,
        "model": config.model_name
    }
    
    response = client.post("/v1/embeddings", json=request_data)
    
    # 应该返回429错误，因为上下文长度超限
    assert response.status_code == 429
    data = response.json()
    assert "error" in data
    assert "context_length_exceeded" in data["error"]["type"]
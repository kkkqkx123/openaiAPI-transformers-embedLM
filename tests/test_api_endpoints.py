import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from emb_model_provider.main import app
from emb_model_provider.core.config import config
from emb_model_provider.api.embeddings import EmbeddingRequest, EmbeddingData
from emb_model_provider.api.models import ModelInfo


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


def test_embeddings_endpoint(client):
    """测试嵌入端点"""
    # Mock both the embedding service and the batch processor
    mock_embedding = [0.1] * config.embedding_dimension  # Mock embedding vector
    mock_embedding_data = [EmbeddingData(embedding=mock_embedding, index=0)]

    with patch('emb_model_provider.api.embeddings.get_embedding_service') as mock_get_service, \
         patch('emb_model_provider.api.embeddings.get_realtime_batch_processor') as mock_get_batch_processor:

        # Create mock service
        mock_service = MagicMock()
        mock_service.process_embedding_request.return_value.data = mock_embedding_data
        mock_service.count_tokens.return_value = 2  # Mock token count
        mock_get_service.return_value = mock_service

        # Create mock batch processor with async submit_request method
        mock_batch_processor = MagicMock()
        mock_batch_processor.submit_request = AsyncMock(return_value=mock_embedding_data)
        mock_get_batch_processor.return_value = mock_batch_processor

        request_data = {
            "input": "Hello world",
            "model": "default"  # Using the 'default' alias defined in config
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
    # Mock both the embedding service and the batch processor
    mock_embedding1 = [0.1] * config.embedding_dimension  # Mock embedding vector
    mock_embedding2 = [0.2] * config.embedding_dimension  # Mock embedding vector
    mock_embedding_data = [
        EmbeddingData(embedding=mock_embedding1, index=0),
        EmbeddingData(embedding=mock_embedding2, index=1)
    ]

    with patch('emb_model_provider.api.embeddings.get_embedding_service') as mock_get_service, \
         patch('emb_model_provider.api.embeddings.get_realtime_batch_processor') as mock_get_batch_processor:
        # Create mock service
        mock_service = MagicMock()
        mock_service.process_embedding_request.return_value.data = mock_embedding_data
        mock_service.count_tokens.return_value = 4  # Mock token count
        mock_get_service.return_value = mock_service

        # Create mock batch processor with async submit_request method
        mock_batch_processor = MagicMock()
        mock_batch_processor.submit_request = AsyncMock(return_value=mock_embedding_data)
        mock_get_batch_processor.return_value = mock_batch_processor

        request_data = {
            "input": ["Hello world", "Test input"],
            "model": "default"  # Using the 'default' alias defined in config
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
    # Mock both the embedding service and the batch processor for invalid input
    from emb_model_provider.api.exceptions import EmbeddingAPIError

    with patch('emb_model_provider.api.embeddings.get_embedding_service') as mock_get_service, \
         patch('emb_model_provider.api.embeddings.get_realtime_batch_processor') as mock_get_batch_processor:

        # Create mock service that raises an EmbeddingAPIError for validation
        mock_service = MagicMock()
        # Mock the process_embedding_request method to raise an EmbeddingAPIError for empty input
        mock_service.process_embedding_request.side_effect = EmbeddingAPIError(
            message="Input cannot be empty.",
            error_type="invalid_request_error",
            param="input"
        )
        # Also mock validate_request to raise a validation error
        mock_service.validate_request.side_effect = EmbeddingAPIError(
            message="Input cannot be empty.",
            error_type="invalid_request_error",
            param="input"
        )
        mock_get_service.return_value = mock_service

        # Create mock batch processor that also raises an EmbeddingAPIError for invalid input
        mock_batch_processor = MagicMock()
        mock_batch_processor.submit_request = AsyncMock(side_effect=EmbeddingAPIError(
            message="Input cannot be empty.",
            error_type="invalid_request_error",
            param="input"
        ))
        mock_get_batch_processor.return_value = mock_batch_processor

        request_data = {
            "input": "",  # 空输入
            "model": "default"  # Using the 'default' alias defined in config
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
    # Mock both the embedding service and the batch processor for batch size exceeded
    with patch('emb_model_provider.api.embeddings.get_embedding_service') as mock_get_service, \
         patch('emb_model_provider.api.embeddings.get_realtime_batch_processor') as mock_get_batch_processor:
        # Create mock service that raises an error for batch size exceeded
        mock_service = MagicMock()
        # Mock the validate_request method to raise a BatchSizeExceededError
        from emb_model_provider.api.exceptions import BatchSizeExceededError
        mock_service.validate_request.side_effect = BatchSizeExceededError(
            max_size=config.max_batch_size,
            actual_size=config.max_batch_size + 1
        )
        mock_get_service.return_value = mock_service

        # Create mock batch processor that also raises an exception for batch size exceeded
        mock_batch_processor = MagicMock()
        mock_batch_processor.submit_request = AsyncMock(side_effect=BatchSizeExceededError(
            max_size=config.max_batch_size,
            actual_size=config.max_batch_size + 1
        ))
        mock_get_batch_processor.return_value = mock_batch_processor

        # Create input that exceeds batch size
        large_input = [f"Test input {i}" for i in range(config.max_batch_size + 1)]

        request_data = {
            "input": large_input,
            "model": "default"  # Using the 'default' alias defined in config
        }

        response = client.post("/v1/embeddings", json=request_data)

        # 应该返回429错误，因为批处理大小超限
        assert response.status_code == 429
        data = response.json()
        assert "error" in data
        assert "batch_size_exceeded" in data["error"]["type"]


def test_embeddings_endpoint_context_length_exceeded(client):
    """测试嵌入端点，超出上下文长度限制"""
    # Mock both the embedding service and the batch processor for context length exceeded
    with patch('emb_model_provider.api.embeddings.get_embedding_service') as mock_get_service, \
         patch('emb_model_provider.api.embeddings.get_realtime_batch_processor') as mock_get_batch_processor:
        # Create mock service that raises an error for context length exceeded
        mock_service = MagicMock()
        # Mock the validate_request method to raise a ContextLengthExceededError
        from emb_model_provider.api.exceptions import ContextLengthExceededError
        mock_service.validate_request.side_effect = ContextLengthExceededError(
            max_length=config.max_context_length,
            actual_length=config.max_context_length + 10
        )
        mock_get_service.return_value = mock_service

        # Create mock batch processor that also raises an exception for context length exceeded
        mock_batch_processor = MagicMock()
        mock_batch_processor.submit_request = AsyncMock(side_effect=ContextLengthExceededError(
            max_length=config.max_context_length,
            actual_length=config.max_context_length + 10
        ))
        mock_get_batch_processor.return_value = mock_batch_processor

        # 创建一个非常长的输入，超过最大上下文长度
        long_text = "test " * (config.max_context_length + 10)

        request_data = {
            "input": long_text,
            "model": "default"  # Using the 'default' alias defined in config
        }

        response = client.post("/v1/embeddings", json=request_data)

        # 应该返回429错误，因为上下文长度超限
        assert response.status_code == 429
        data = response.json()
        assert "error" in data
        assert "context_length_exceeded" in data["error"]["type"]
import pytest
from typing import List
from emb_model_provider.api.v1.embeddings import EmbeddingRequest, EmbeddingData, Usage, EmbeddingResponse
from emb_model_provider.api.v1.models import ModelInfo, ModelsResponse


class TestEmbeddingRequest:
    """测试 EmbeddingRequest 模型"""
    
    def test_embedding_request_with_string_input(self):
        """测试使用字符串输入的请求模型"""
        request = EmbeddingRequest(
            input="Hello world",
            model="all-MiniLM-L12-v2"
        )
        assert request.input == "Hello world"
        assert request.model == "all-MiniLM-L12-v2"
        assert request.encoding_format == "float"
        assert request.user is None
    
    def test_embedding_request_with_list_input(self):
        """测试使用列表输入的请求模型"""
        request = EmbeddingRequest(
            input=["Hello world", "Test input"],
            model="all-MiniLM-L12-v2",
            encoding_format="base64",
            user="test_user"
        )
        assert request.input == ["Hello world", "Test input"]
        assert request.model == "all-MiniLM-L12-v2"
        assert request.encoding_format == "base64"
        assert request.user == "test_user"
    
    def test_embedding_request_default_values(self):
        """测试请求模型的默认值"""
        request = EmbeddingRequest(
            input="Hello world",
            model="all-MiniLM-L12-v2"
        )
        assert request.encoding_format == "float"
        assert request.user is None


class TestEmbeddingData:
    """测试 EmbeddingData 模型"""
    
    def test_embedding_data(self):
        """测试嵌入数据模型"""
        embedding_data = EmbeddingData(
            embedding=[0.1, 0.2, 0.3],
            index=0
        )
        assert embedding_data.object == "embedding"
        assert embedding_data.embedding == [0.1, 0.2, 0.3]
        assert embedding_data.index == 0


class TestUsage:
    """测试 Usage 模型"""
    
    def test_usage(self):
        """测试使用情况模型"""
        usage = Usage(
            prompt_tokens=10,
            total_tokens=10
        )
        assert usage.prompt_tokens == 10
        assert usage.total_tokens == 10


class TestEmbeddingResponse:
    """测试 EmbeddingResponse 模型"""
    
    def test_embedding_response(self):
        """测试嵌入响应模型"""
        response = EmbeddingResponse(
            data=[
                EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0),
                EmbeddingData(embedding=[0.4, 0.5, 0.6], index=1)
            ],
            model="all-MiniLM-L12-v2",
            usage=Usage(prompt_tokens=10, total_tokens=10)
        )
        assert response.object == "list"
        assert len(response.data) == 2
        assert response.model == "all-MiniLM-L12-v2"
        assert response.usage.prompt_tokens == 10
        assert response.usage.total_tokens == 10


class TestModelInfo:
    """测试 ModelInfo 模型"""
    
    def test_model_info(self):
        """测试模型信息模型"""
        model_info = ModelInfo(
            id="all-MiniLM-L12-v2",
            owned_by="user"
        )
        assert model_info.id == "all-MiniLM-L12-v2"
        assert model_info.object == "model"
        assert model_info.owned_by == "user"
        # 检查 created 时间戳是否为整数（近似检查）
        assert isinstance(model_info.created, int)
        assert model_info.created > 0


class TestModelsResponse:
    """测试 ModelsResponse 模型"""
    
    def test_models_response(self):
        """测试模型响应模型"""
        response = ModelsResponse(
            data=[
                ModelInfo(id="all-MiniLM-L12-v2", owned_by="user"),
                ModelInfo(id="another-model", owned_by="user")
            ]
        )
        assert response.object == "list"
        assert len(response.data) == 2
        assert response.data[0].id == "all-MiniLM-L12-v2"
        assert response.data[1].id == "another-model"


if __name__ == "__main__":
    pytest.main([__file__])
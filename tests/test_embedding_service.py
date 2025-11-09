import pytest
import torch
from unittest.mock import Mock, patch
from emb_model_provider.services.embedding_service import EmbeddingService
from emb_model_provider.api.embeddings import EmbeddingRequest
from emb_model_provider.core.config import Config
from emb_model_provider.api.exceptions import EmbeddingAPIError, BatchSizeExceededError, ContextLengthExceededError


class TestEmbeddingService:
    """测试 EmbeddingService 类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = Config()
        # Mock模型和tokenizer
        with patch('emb_model_provider.core.model_manager.ModelManager') as mock_model_manager:
            mock_model_manager_instance = Mock()
            mock_model_manager_instance.tokenizer = Mock()
            mock_model_manager_instance.model = Mock()
            
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 模拟编码结果
            mock_tokenizer.return_value = {
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
            }
            mock_model_manager_instance.tokenizer = mock_tokenizer
            
            # Mock model output
            mock_model_output = Mock()
            mock_model_output.last_hidden_state = torch.randn(1, 5, 384)  # 模拟模型输出
            mock_model_manager_instance.model.return_value = mock_model_output
            
            mock_model_manager.return_value = mock_model_manager_instance
            self.service = EmbeddingService(self.config)
    
    def test_validate_request_with_empty_input(self):
        """测试空输入验证"""
        request = EmbeddingRequest(
            input="",
            model="all-MiniLM-L12-v2"
        )
        
        with pytest.raises(EmbeddingAPIError) as exc_info:
            self.service.validate_request(request)
        
        assert exc_info.value.message == "Input cannot be empty."
        assert exc_info.value.type == "invalid_request_error"
        assert exc_info.value.param == "input"
    
    def test_validate_request_with_batch_size_exceeded(self):
        """测试批处理大小超出限制的验证"""
        # 创建超过最大批处理大小的输入
        large_input = ["test"] * (self.config.max_batch_size + 1)
        request = EmbeddingRequest(
            input=large_input,
            model="all-MiniLM-L12-v2"
        )
        
        with pytest.raises(BatchSizeExceededError):
            self.service.validate_request(request)
    
    def test_validate_request_with_context_length_exceeded(self):
        """测试上下文长度超出限制的验证"""
        # Mock tokenizer以返回超过限制的token数
        long_text = "test " * (self.config.max_context_length + 10)
        request = EmbeddingRequest(
            input=long_text,
            model="all-MiniLM-L12-v2"
        )
        
        # 修改tokenizer的encode方法返回长序列
        self.service.tokenizer.encode = Mock(return_value=list(range(self.config.max_context_length + 10)))
        
        with pytest.raises(ContextLengthExceededError):
            self.service.validate_request(request)
    
    def test_count_tokens_single_string(self):
        """测试单个字符串的token计数"""
        # Mock tokenizer的encode方法
        self.service.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        
        tokens_count = self.service.count_tokens("Hello world")
        assert tokens_count == 5
        
    def test_count_tokens_list_of_strings(self):
        """测试字符串列表的token计数"""
        # Mock tokenizer的encode方法
        self.service.tokenizer.encode = Mock(return_value=[1, 2, 3])
        
        tokens_count = self.service.count_tokens(["Hello", "world"])
        # 每个字符串都编码为3个token，总共2个字符串
        assert tokens_count == 6
    
    def test_mean_pooling(self):
        """测试平均池化功能"""
        # 创建模拟的模型输出和attention mask
        model_output = Mock()
        model_output.last_hidden_state = torch.randn(2, 5, 10)  # 2个序列，每个5个token，每个token 10维
        attention_mask = torch.tensor([
            [1, 1, 0, 0],  # 第一个序列前3个token有效
            [1, 1, 1, 1, 1]   # 第二个序列所有token有效
        ])
        
        result = self.service._mean_pooling(model_output, attention_mask)
        
        # 验证输出形状
        assert result.shape == (2, 10)  # 2个序列，每个10维嵌入
    
    def test_process_embedding_request(self):
        """测试处理嵌入请求的完整流程"""
        # Mock必要的方法
        self.service.validate_request = Mock()
        self.service.generate_embeddings = Mock(return_value=[])
        self.service.create_embedding_response = Mock()
        
        request = EmbeddingRequest(
            input="Hello world",
            model="all-MiniLM-L12-v2"
        )
        
        # 调用方法
        self.service.process_embedding_request(request)
        
        # 验证方法被调用
        self.service.validate_request.assert_called_once_with(request)
        self.service.generate_embeddings.assert_called_once()
        self.service.create_embedding_response.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
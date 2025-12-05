import pytest
import torch
from unittest.mock import Mock, patch
from emb_model_provider.services.embedding_service import EmbeddingService
from emb_model_provider.api.embeddings import EmbeddingRequest
from emb_model_provider.core.config import Config
from emb_model_provider.api.exceptions import EmbeddingAPIError, BatchSizeExceededError, ContextLengthExceededError, ModelNotFoundError


class TestEmbeddingService:
    """测试 EmbeddingService 类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = Config()

        # Mock tokenizer manager
        with patch('emb_model_provider.services.embedding_service.initialize_tokenizer_manager') as mock_init_tokenizer_manager:
            # Create a mock tokenizer manager
            mock_tokenizer_manager = Mock()
            # Create a mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 模拟编码结果
            mock_tokenizer.return_value = {
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
            }
            # Make the tokenizer manager return the mock tokenizer
            mock_tokenizer_manager.get_tokenizer.return_value = mock_tokenizer
            mock_init_tokenizer_manager.return_value = mock_tokenizer_manager

            # Mock模型
            with patch('emb_model_provider.core.model_manager.get_model_manager') as mock_get_model_manager:
                mock_model_manager_instance = Mock()
                mock_model_manager_instance.model = Mock()
                mock_model_manager_instance.is_model_loaded.return_value = True

                # Mock model output
                mock_model_output = Mock()
                mock_model_output.last_hidden_state = torch.randn(1, 5, 384)  # 模拟模型输出
                mock_get_model_manager.return_value = mock_model_manager_instance

                self.service = EmbeddingService(self.config)
                # Store reference to mock tokenizer for tests
                self.mock_tokenizer = mock_tokenizer
    
    def test_validate_request_with_empty_input(self):
        """测试空输入验证"""
        request = EmbeddingRequest(
            input="",
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        with pytest.raises(EmbeddingAPIError) as exc_info:
            self.service.validate_request(request)
        
        assert exc_info.value.message == "Input cannot be empty."
        assert exc_info.value.type == "invalid_request_error"
        assert exc_info.value.param == "input"
    
    def test_validate_request_with_unknown_model(self):
        """测试未知模型验证"""
        # Mock config.get_model_info返回空dict（模型不存在）
        with patch.object(Config, 'get_model_info', return_value={}):
            request = EmbeddingRequest(
                input="test",
                model="unknown-model"
            )
            
            with pytest.raises(ModelNotFoundError) as exc_info:
                self.service.validate_request(request)
            
            assert exc_info.value.model_name == "unknown-model"
    
    def test_validate_request_with_batch_size_exceeded(self):
        """测试批处理大小超出限制的验证"""
        # 创建超过最大批处理大小的输入
        large_input = ["test"] * (self.config.max_batch_size + 1)
        request = EmbeddingRequest(
            input=large_input,
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        with pytest.raises(BatchSizeExceededError):
            self.service.validate_request(request)
    
    def test_validate_request_with_context_length_exceeded(self):
        """测试上下文长度超出限制的验证"""
        # Mock tokenizer以返回超过限制的token数
        long_text = "test " * (self.config.max_context_length + 10)
        request = EmbeddingRequest(
            input=long_text,
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # 修改tokenizer的encode方法返回长序列
        tokenizer = self.service.get_tokenizer(request.model)
        tokenizer.encode = Mock(return_value=list(range(self.config.max_context_length + 10)))

        with pytest.raises(ContextLengthExceededError):
            self.service.validate_request(request)
    
    def test_count_tokens_single_string(self):
        """测试单个字符串的token计数"""
        model_alias = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        # Mock tokenizer manager for the specific model
        mock_tokenizer_manager = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 模拟编码结果为5个token
        mock_tokenizer_manager.get_tokenizer.return_value = mock_tokenizer

        # Replace the tokenizer manager for this specific test
        self.service.tokenizer_managers[model_alias] = mock_tokenizer_manager

        tokens_count = self.service.count_tokens("Hello world", model_alias)
        assert tokens_count == 5
        
    def test_count_tokens_list_of_strings(self):
        """测试字符串列表的token计数"""
        model_alias = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        # Mock tokenizer manager for the specific model
        mock_tokenizer_manager = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]  # 模拟编码结果为3个token
        mock_tokenizer_manager.get_tokenizer.return_value = mock_tokenizer

        # Replace the tokenizer manager for this specific test
        self.service.tokenizer_managers[model_alias] = mock_tokenizer_manager

        tokens_count = self.service.count_tokens(["Hello", "world"], model_alias)
        # 每个字符串都编码为3个token，总共2个字符串
        assert tokens_count == 6
    
    def test_mean_pooling(self):
        """测试平均池化功能"""
        # 创建模拟的模型输出和attention mask
        model_output = Mock()
        model_output.last_hidden_state = torch.randn(2, 5, 10)  # 2个序列，每个5个token，每个token 10维
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],  # 第一个序列前3个token有效，总共5个token
            [1, 1, 1, 1, 1]   # 第二个序列所有token有效，总共5个token
        ])
        
        result = self.service._mean_pooling(model_output, attention_mask)
        
        # 验证输出形状
        assert result.shape == (2, 10)  # 2个序列，每个10维嵌入
    
    def test_process_embedding_request(self):
        """测试处理嵌入请求的完整流程"""
        # Mock必要的方法
        with patch.object(self.service, 'count_tokens', return_value=5):
            self.service.validate_request = Mock()
            self.service.generate_embeddings = Mock(return_value=[])
            self.service.create_embedding_response = Mock()

            request = EmbeddingRequest(
                input="Hello world",
                model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

            # 调用方法
            self.service.process_embedding_request(request)

            # 验证方法被调用
            self.service.validate_request.assert_called_once_with(request)
            # Note: Need to account for model parameter
            self.service.generate_embeddings.assert_called_once()
            self.service.create_embedding_response.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
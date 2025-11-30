"""
Unit tests for model loader implementations.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open
import pytest
import torch

from emb_model_provider.loaders.base_loader import BaseModelLoader
from emb_model_provider.loaders.huggingface_loader import HuggingFaceModelLoader
from emb_model_provider.loaders.modelscope_loader import ModelScopeModelLoader
from emb_model_provider.api.exceptions import ModelLoadError

# Mock for ModelScope imports
class MockPipeline:
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, input):
        return {'text_embedding': [[0.1] * 384] * len(input['source_sentence'])}

class MockTasks:
    sentence_embedding = "sentence-embedding"


class TestBaseModelLoader:
    """Test cases for BaseModelLoader abstract class."""
    
    def test_abstract_methods(self):
        """Test that BaseModelLoader has required abstract methods."""
        # Should raise TypeError when trying to instantiate abstract class
        with pytest.raises(TypeError):
            BaseModelLoader("test-model")  # type: ignore[abstract]
        
        # Check that required abstract methods exist
        assert hasattr(BaseModelLoader, 'load_model')
        assert hasattr(BaseModelLoader, 'is_model_available')
        assert hasattr(BaseModelLoader, 'get_model_info')


class TestHuggingFaceModelLoader:
    """Test cases for HuggingFaceModelLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_name = "sentence-transformers/all-MiniLM-L12-v2"
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model")
        
        # Create mock model files
        os.makedirs(self.model_path, exist_ok=True)
        
        # Create required model files
        with open(os.path.join(self.model_path, "config.json"), "w") as f:
            f.write('{"model_type": "bert"}')
        
        with open(os.path.join(self.model_path, "pytorch_model.bin"), "wb") as f:
            f.write(b"dummy model data")
        
        with open(os.path.join(self.model_path, "tokenizer.json"), "w") as f:
            f.write('{"vocab": {"test": 0}}')
        
        with open(os.path.join(self.model_path, "tokenizer_config.json"), "w") as f:
            f.write('{"name": "test-tokenizer"}')
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test HuggingFaceModelLoader initialization."""
        loader = HuggingFaceModelLoader(self.model_name)
        
        assert loader.model_name == self.model_name
        assert loader.model_path is None
        assert loader.model is None
        assert loader.tokenizer is None
        assert not loader._model_loaded
    
    def test_init_with_model_path(self):
        """Test initialization with custom model path."""
        loader = HuggingFaceModelLoader(self.model_name, model_path=self.model_path)
        
        assert loader.model_name == self.model_name
        assert loader.model_path == self.model_path
    
    @patch('transformers.AutoConfig')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoTokenizer')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoModel')
    def test_load_model_success(self, mock_auto_model, mock_auto_tokenizer, mock_auto_config):
        """Test successful model loading."""
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Mock the config to return None for torch_dtype to ensure it falls back to CPU default (float32)
        mock_config_instance = MagicMock()
        mock_config_instance.torch_dtype = None
        mock_auto_config.from_pretrained.return_value = mock_config_instance
        
        loader = HuggingFaceModelLoader(self.model_name, load_from_transformers=True)
        model, tokenizer = loader.load_model()
        
        assert loader._model == mock_auto_model.from_pretrained.return_value
        assert loader._tokenizer == mock_auto_tokenizer.from_pretrained.return_value
        assert loader._model_loaded is True
        assert model == mock_auto_model.from_pretrained.return_value
        assert tokenizer == mock_auto_tokenizer.from_pretrained.return_value
        assert loader._model_loaded is True
        
        # Verify model was loaded with correct parameters
        # For "all-MiniLM" models, it will use float16 based on heuristics
        mock_auto_model.from_pretrained.assert_called_once_with(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
            trust_remote_code=False
        )
    
    @patch('transformers.AutoConfig')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoTokenizer')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoModel')
    def test_load_model_already_loaded(self, mock_auto_model, mock_auto_tokenizer, mock_auto_config):
        """Test loading model when already loaded."""
        loader = HuggingFaceModelLoader(self.model_name)
        loader._model_loaded = True
        loader._model = MagicMock()
        loader._tokenizer = MagicMock()
        
        model, tokenizer = loader.load_model()
        
        # Should return existing model without loading
        assert model == loader._model
        assert tokenizer == loader._tokenizer
        mock_auto_model.from_pretrained.assert_not_called()
        mock_auto_tokenizer.from_pretrained.assert_not_called()
    
    @patch('transformers.AutoConfig')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoTokenizer')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoModel')
    def test_load_model_failure(self, mock_auto_model, mock_auto_tokenizer, mock_auto_config):
        """Test model loading failure."""
        mock_auto_model.from_pretrained.side_effect = Exception("Model not found")
        
        loader = HuggingFaceModelLoader(self.model_name, load_from_transformers=True)
        
        with pytest.raises(ModelLoadError, match=f"Failed to load model {self.model_name}"):
            loader.load_model()
    
    def test_is_model_available_true(self):
        """Test checking if model is available - positive case."""
        loader = HuggingFaceModelLoader(self.model_name, model_path=self.model_path)
        available = loader.is_model_available()
        
        assert available is True
    
    def test_is_model_available_false(self):
        """Test checking if model is available - negative case."""
        empty_path = os.path.join(self.temp_dir, "empty_model")
        loader = HuggingFaceModelLoader(self.model_name, model_path=empty_path)
        available = loader.is_model_available()
        
        assert available is False
    
    @patch('transformers.AutoConfig')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoTokenizer')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoModel')
    def test_get_model_info_loaded(self, mock_auto_model, mock_auto_tokenizer, mock_auto_config):
        """Test getting model info when model is loaded."""
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_model.config.model_type = "bert"
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 30000
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        loader = HuggingFaceModelLoader(self.model_name, load_from_transformers=True)
        loader.load_model()
        
        info = loader.get_model_info()
        
        assert info["model_name"] == self.model_name
        assert info["source"] == "transformers"
        assert info["model_type"] == "bert"
        assert info["vocab_size"] == 30000
        assert info["loaded"] is True
    
    def test_get_model_info_not_loaded(self):
        """Test getting model info when model is not loaded."""
        loader = HuggingFaceModelLoader(self.model_name)
        
        info = loader.get_model_info()
        
        assert info["model_name"] == self.model_name
        assert info["source"] == "huggingface"
        assert info["loaded"] is False
    
    @patch('transformers.AutoConfig')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoTokenizer')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoModel')
    def test_validate_model_success(self, mock_auto_model, mock_auto_tokenizer, mock_auto_config):
        """Test model validation success."""
        # Create mock model with proper output structure
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(last_hidden_state=torch.randn(1, 3, 384))
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        loader = HuggingFaceModelLoader(self.model_name, load_from_transformers=True)
        loader.load_model()
        
        valid = loader.validate_model()
        
        assert valid is True
    
    @patch('transformers.AutoConfig')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoTokenizer')
    @patch('emb_model_provider.loaders.huggingface_loader.AutoModel')
    def test_cleanup(self, mock_auto_model, mock_auto_tokenizer, mock_auto_config):
        """Test model cleanup."""
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        loader = HuggingFaceModelLoader(self.model_name, load_from_transformers=True)
        loader.load_model()
        
        # Verify model is loaded
        assert loader._model_loaded is True
        assert loader._model is not None
        assert loader._tokenizer is not None
        
        # Clean up
        loader.cleanup()
        
        # Verify resources are cleaned up
        assert loader._model_loaded is False
        assert loader._model is None
        assert loader._tokenizer is None


class TestModelScopeModelLoader:
    """Test cases for ModelScopeModelLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_name = "damo/nlp_gte_sentence-embedding_chinese-base"
    
    def test_initialization(self):
        """Test that ModelScopeModelLoader initializes correctly."""
        loader = ModelScopeModelLoader("test-model")
        assert loader.model_name == "test-model"
        assert loader.model_path is None
        assert loader._pipeline is None
        assert loader._model_loaded is False
    
    def test_load_model_success(self):
        """Test successful model loading."""
        # Test the exception handling when modelscope is not available
        loader = ModelScopeModelLoader("test-model")
        
        # We expect this to raise an error since modelscope is not installed in test environment
        with pytest.raises(ModelLoadError):
            loader.load_model()
    
    def test_load_model_already_loaded(self):
        """Test loading when model is already loaded."""
        loader = ModelScopeModelLoader("test-model")
        loader._pipeline = MockPipeline()  # type: ignore[assignment]
        loader._model_loaded = True
        
        pipeline, _ = loader.load_model()
        
        # Should return existing pipeline without re-loading
        assert pipeline is not None
    
    @patch('emb_model_provider.loaders.modelscope_loader.modelscope', create=True)
    def test_load_model_failure(self, mock_modelscope_module):
        """Test model loading failure."""
        # Mock the modelscope submodules
        mock_modelscope_module.pipelines.pipeline = MagicMock(side_effect=ImportError("ModelScope not installed"))
        
        loader = ModelScopeModelLoader("test-model")
        
        # Mock import failure
        with patch('emb_model_provider.loaders.modelscope_loader.modelscope.pipelines.pipeline', side_effect=ImportError("ModelScope not installed")):
            with pytest.raises(ModelLoadError) as exc_info:
                loader.load_model()
            
            assert "ModelScope not installed" in str(exc_info.value)
    
    def test_is_model_available(self):
        """Test checking if model is available."""
        loader = ModelScopeModelLoader("test-model")
        available = loader.is_model_available()
        
        # ModelScope loader always returns True as it relies on pipeline creation
        assert available is True
    
    def test_get_model_info_loaded(self):
        """Test getting model info when model is loaded."""
        loader = ModelScopeModelLoader("test-model")
        loader._pipeline = MockPipeline()  # type: ignore[assignment]
        loader._model_loaded = True
        
        info = loader.get_model_info()
        
        assert info["model_name"] == "test-model"
        assert info["loaded"] is True
        assert info["source"] == "modelscope"
        assert "pipeline_type" in info
    
    def test_get_model_info_not_loaded(self):
        """Test getting model info when model is not loaded."""
        loader = ModelScopeModelLoader("test-model")
        
        info = loader.get_model_info()
        
        assert info["model_name"] == "test-model"
        assert info["loaded"] is False
        assert info["source"] == "modelscope"
    
    def test_generate_embeddings(self):
        """Test generating embeddings."""
        loader = ModelScopeModelLoader("test-model")
        loader._pipeline = MockPipeline()  # type: ignore[assignment]
        loader._model_loaded = True
        
        embeddings = loader.generate_embeddings(["test input"])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
    
    def test_generate_embeddings_not_loaded(self):
        """Test generating embeddings when model is not loaded."""
        loader = ModelScopeModelLoader("test-model")
        
        with pytest.raises(ModelLoadError) as exc_info:
            loader.generate_embeddings(["test input"])
        
        assert "Model is not loaded" in str(exc_info.value)
    
    def test_cleanup(self):
        """Test that cleanup properly resets the loader."""
        loader = ModelScopeModelLoader("test-model")
        
        # Mock the pipeline
        loader._pipeline = MockPipeline()  # type: ignore[assignment]
        loader._model_loaded = True
        
        # Cleanup
        loader.cleanup()
        
        # Check that pipeline is reset
        assert loader._pipeline is None
        assert not loader._model_loaded
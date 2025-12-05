"""
Unit tests for the model manager module.
"""

import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
import torch

from emb_model_provider.core.model_manager import ModelManager


class TestModelManager:
    """Test cases for ModelManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model")
        self.model_name = "test-model"
        
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
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_init_with_defaults(self, mock_config):
        """Test ModelManager initialization with default values."""
        # Setup config mock for get_model_info
        mock_model_config = {
            "name": self.model_name,
            "path": self.model_path,
            "source": "transformers",
            "precision": "auto",
            "trust_remote_code": False,
            "revision": "main",
            "fallback_to_huggingface": True,
            "load_from_transformers": False
        }
        mock_config.get_model_info.return_value = mock_model_config
        mock_config.device = "auto"
        mock_config.model_name = self.model_name

        manager = ModelManager(model_alias=self.model_name)

        assert manager.model_path == self.model_path
        assert manager.model_name == self.model_name
        assert not manager._model_loaded
        assert manager._model is None
        assert manager._tokenizer is None
    
    def test_init_with_custom_values(self):
        """Test ModelManager initialization with custom values."""
        from unittest.mock import Mock
        custom_path = os.path.join(self.temp_dir, "custom_model")
        custom_name = "custom-model"
        
        custom_loader = Mock()
        manager = ModelManager(loader=custom_loader)
        
        assert manager._loader == custom_loader
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_device_property(self, mock_config):
        """Test device property is set correctly."""
        # Setup config mock for get_model_info
        mock_model_config = {
            "name": self.model_name,
            "path": self.model_path,
            "source": "transformers",
            "precision": "auto",
            "trust_remote_code": False,
            "revision": "main",
            "fallback_to_huggingface": True,
            "load_from_transformers": False
        }
        mock_config.get_model_info.return_value = mock_model_config
        mock_config.device = "cpu"
        mock_config.model_name = self.model_name

        manager = ModelManager(model_alias=self.model_name)

        assert manager.device == "cpu"
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_is_loaded_property(self, mock_config):
        """Test is_loaded property."""
        # Setup config mock for get_model_info
        mock_model_config = {
            "name": self.model_name,
            "path": self.model_path,
            "source": "transformers",
            "precision": "auto",
            "trust_remote_code": False,
            "revision": "main",
            "fallback_to_huggingface": True,
            "load_from_transformers": False
        }
        mock_config.get_model_info.return_value = mock_model_config
        mock_config.model_name = self.model_name

        manager = ModelManager(model_alias=self.model_name)

        assert manager.is_loaded is False

        manager._model_loaded = True
        assert manager.is_loaded is True
    
    @patch.object(ModelManager, '_create_loader')
    @patch('emb_model_provider.core.model_manager.config')
    def test_load_model_local_available(self, mock_config, mock_create_loader):
        """Test loading model when it's available locally."""
        # Setup config mock for get_model_info
        mock_model_config = {
            "name": self.model_name,
            "path": self.model_path,
            "source": "transformers",
            "precision": "auto",
            "trust_remote_code": False,
            "revision": "main",
            "fallback_to_huggingface": True,
            "load_from_transformers": False
        }
        mock_config.get_model_info.return_value = mock_model_config
        mock_config.device = "cpu"
        mock_config.model_name = self.model_name
        mock_config.enable_offline_mode = False
        mock_config.enable_path_priority = True

        # Create mock loader
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_create_loader.return_value = mock_loader

        manager = ModelManager(model_alias=self.model_name)
        manager.load_model()

        # Check that the loader was created and used to load the model
        mock_create_loader.assert_called_once()
        mock_loader.load_model.assert_called_once()
        assert manager._model_loaded is True
        assert manager._model is mock_model
        assert manager._tokenizer is mock_tokenizer
    
    @patch.object(ModelManager, '_create_loader')
    @patch('emb_model_provider.core.model_manager.config')
    def test_load_model_download_required(self, mock_config, mock_create_loader):
        """Test loading model when download is required."""
        # Setup config mock for get_model_info
        mock_model_config = {
            "name": self.model_name,
            "path": self.model_path,
            "source": "transformers",
            "precision": "auto",
            "trust_remote_code": False,
            "revision": "main",
            "fallback_to_huggingface": True,
            "load_from_transformers": False
        }
        mock_config.get_model_info.return_value = mock_model_config
        mock_config.device = "cpu"
        mock_config.model_name = self.model_name
        mock_config.enable_offline_mode = False
        mock_config.enable_path_priority = True

        # Create mock loader
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_create_loader.return_value = mock_loader

        manager = ModelManager(model_alias=self.model_name)
        manager.load_model()

        # Check that the loader was created and used to load the model
        mock_create_loader.assert_called_once()
        mock_loader.load_model.assert_called_once()
        assert manager._model_loaded is True
        assert manager._model is mock_model
        assert manager._tokenizer is mock_tokenizer
    
    @patch.object(ModelManager, '_create_loader')
    @patch('emb_model_provider.core.model_manager.config')
    def test_load_model_already_loaded(self, mock_config, mock_create_loader):
        """Test loading model when it's already loaded."""
        # Setup config mock for get_model_info
        mock_model_config = {
            "name": self.model_name,
            "path": self.model_path,
            "source": "transformers",
            "precision": "auto",
            "trust_remote_code": False,
            "revision": "main",
            "fallback_to_huggingface": True,
            "load_from_transformers": False
        }
        mock_config.get_model_info.return_value = mock_model_config
        mock_config.device = "cpu"
        mock_config.model_name = self.model_name
        mock_config.enable_offline_mode = False
        mock_config.enable_path_priority = True

        # Create mock loader
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_create_loader.return_value = mock_loader

        manager = ModelManager(model_alias=self.model_name)
        manager._model_loaded = True
        # Call load_model again - it should not try to load again
        manager.load_model()

        # Since model is already loaded, create_loader should not be called again
        mock_create_loader.assert_not_called()
        assert manager._model_loaded is True
    
    @patch.object(ModelManager, '_mean_pooling')
    @patch.object(ModelManager, '_tokenize_inputs')
    @patch('emb_model_provider.core.model_manager.config')
    def test_generate_embeddings_single_input(self, mock_config, mock_tokenize, mock_mean_pooling):
        """Test generating embeddings for a single input."""
        # Setup config mock for get_model_info
        mock_model_config = {
            "name": self.model_name,
            "path": self.model_path,
            "source": "transformers",
            "precision": "auto",
            "trust_remote_code": False,
            "revision": "main",
            "fallback_to_huggingface": True,
            "load_from_transformers": False
        }
        mock_config.get_model_info.return_value = mock_model_config
        mock_config.model_name = self.model_name
        mock_config.max_batch_size = 32
        mock_config.max_context_length = 512
        mock_config.embedding_dimension = 384
        mock_config.device = "cpu"  # Use CPU for testing

        # Set up mock model and tokenizer
        manager = ModelManager(model_alias=self.model_name)
        manager._model_loaded = True
        manager._model = MagicMock()
        manager._tokenizer = MagicMock()

        # Mock tokenization
        mock_tokenized = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_tokenize.return_value = mock_tokenized

        # Mock model output and pooling
        class MockOutput:
            def __init__(self):
                self.last_hidden_state = torch.tensor([[[0.1, 0.2, 0.3]]])

        mock_model_output = MockOutput()
        manager._model.return_value = mock_model_output
        mock_mean_pooling.return_value = torch.tensor([[0.1, 0.2, 0.3]])

        # Generate embeddings
        result = manager.generate_embeddings("test input")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 3
    
    @patch.object(ModelManager, '_mean_pooling')
    @patch.object(ModelManager, '_tokenize_inputs')
    @patch('emb_model_provider.core.model_manager.config')
    def test_generate_embeddings_multiple_inputs(self, mock_config, mock_tokenize, mock_mean_pooling):
        """Test generating embeddings for multiple inputs."""
        # Setup config mock for get_model_info
        mock_model_config = {
            "name": self.model_name,
            "path": self.model_path,
            "source": "transformers",
            "precision": "auto",
            "trust_remote_code": False,
            "revision": "main",
            "fallback_to_huggingface": True,
            "load_from_transformers": False
        }
        mock_config.get_model_info.return_value = mock_model_config
        mock_config.model_name = self.model_name
        mock_config.max_batch_size = 32
        mock_config.max_context_length = 512
        mock_config.embedding_dimension = 384
        mock_config.device = "cpu"  # Use CPU for testing

        # Set up mock model and tokenizer
        manager = ModelManager(model_alias=self.model_name)
        manager._model_loaded = True
        manager._model = MagicMock()
        manager._tokenizer = MagicMock()

        # Mock tokenization
        mock_tokenized = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        mock_tokenize.return_value = mock_tokenized

        # Mock model output and pooling
        class MockOutput:
            def __init__(self):
                self.last_hidden_state = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])

        mock_model_output = MockOutput()
        manager._model.return_value = mock_model_output
        mock_mean_pooling.return_value = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Generate embeddings
        result = manager.generate_embeddings(["input 1", "input 2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(emb, list) for emb in result)
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_generate_embeddings_model_not_loaded(self, mock_config):
        """Test generating embeddings when model is not loaded."""
        mock_config.model_path = self.model_path
        mock_config.model_name = self.model_name
        
        manager = ModelManager()
        
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            manager.generate_embeddings("test input")
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_generate_embeddings_empty_inputs(self, mock_config):
        """Test generating embeddings with empty inputs."""
        mock_config.model_path = self.model_path
        mock_config.model_name = self.model_name
        
        manager = ModelManager()
        manager._model_loaded = True
        
        with pytest.raises(ValueError, match="Inputs cannot be empty"):
            manager.generate_embeddings([])
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_generate_embeddings_batch_size_exceeded(self, mock_config):
        """Test generating embeddings with batch size exceeded."""
        mock_config.model_path = self.model_path
        mock_config.model_name = self.model_name
        mock_config.max_batch_size = 2
        
        manager = ModelManager()
        manager._model_loaded = True
        
        with pytest.raises(ValueError, match="Batch size 3 exceeds maximum 2"):
            manager.generate_embeddings(["input 1", "input 2", "input 3"])
    
    @patch.object(ModelManager, 'generate_embeddings')
    @patch('emb_model_provider.core.model_manager.config')
    def test_generate_batch_embeddings(self, mock_config, mock_generate_embeddings):
        """Test batch embedding generation."""
        mock_config.model_path = self.model_path
        mock_config.model_name = self.model_name
        mock_config.max_batch_size = 2
        
        # Mock generate_embeddings to return different results for each batch
        def mock_generate(inputs):
            return [[float(i)] for i in range(len(inputs))]
        
        mock_generate_embeddings.side_effect = mock_generate
        
        manager = ModelManager()
        manager._model_loaded = True
        
        # Generate batch embeddings
        inputs = ["input 1", "input 2", "input 3", "input 4", "input 5"]
        result = manager.generate_batch_embeddings(inputs)
        
        assert len(result) == 5
        assert result == [[0.0], [1.0], [0.0], [1.0], [0.0]]
        
        # Verify generate_embeddings was called correct number of times
        assert mock_generate_embeddings.call_count == 3
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_get_model_info_not_loaded(self, mock_config):
        """Test getting model info when model is not loaded."""
        mock_config.model_path = self.model_path
        mock_config.model_name = self.model_name
        mock_config.load_from_transformers = False  # Explicitly set this
        
        manager = ModelManager()
        
        info = manager.get_model_info()
        
        assert info["model_name"] == self.model_name
        assert info["model_path"] == self.model_path
        assert info["loaded"] is False
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_get_model_info_loaded(self, mock_config):
        """Test getting model info when model is loaded."""
        mock_config.model_path = self.model_path
        mock_config.model_name = self.model_name
        mock_config.max_context_length = 512
        mock_config.embedding_dimension = 384
        mock_config.load_from_transformers = False  # Explicitly set this
        
        manager = ModelManager()
        manager._model_loaded = True
        manager._model = MagicMock()
        manager._model.config.model_type = "bert"
        manager._tokenizer = MagicMock()
        manager._tokenizer.vocab_size = 30000
        
        info = manager.get_model_info()
        
        assert info["model_name"] == self.model_name
        assert info["model_path"] == self.model_path
        assert info["loaded"] is True
        assert info["max_context_length"] == 512
        assert info["embedding_dimension"] == 384
        assert info["vocab_size"] == 30000
        assert info["model_type"] == "bert"
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_save_embeddings_cache(self, mock_config):
        """Test saving embeddings cache."""
        mock_config.model_path = self.model_path
        mock_config.model_name = self.model_name
        mock_config.max_context_length = 512
        mock_config.embedding_dimension = 384
        
        manager = ModelManager()
        
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        inputs = ["input 1", "input 2"]
        cache_path = os.path.join(self.temp_dir, "cache.pkl")
        
        manager.save_embeddings_cache(embeddings, inputs, cache_path)
        
        # Verify cache file was created
        assert os.path.exists(cache_path)
        
        # Verify cache content
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        assert cache_data["inputs"] == inputs
        assert cache_data["embeddings"] == embeddings
        assert cache_data["model_name"] == self.model_name
        assert cache_data["config"]["max_context_length"] == 512
        assert cache_data["config"]["embedding_dimension"] == 384
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_load_embeddings_cache_success(self, mock_config):
        """Test loading embeddings cache successfully."""
        mock_config.model_path = self.model_path
        mock_config.model_name = self.model_name
        mock_config.max_context_length = 512
        mock_config.embedding_dimension = 384
        
        manager = ModelManager()
        
        # Create cache file
        cache_data = {
            "inputs": ["input 1", "input 2"],
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "model_name": self.model_name,
            "config": {
                "max_context_length": 512,
                "embedding_dimension": 384
            }
        }
        
        cache_path = os.path.join(self.temp_dir, "cache.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Load cache
        result = manager.load_embeddings_cache(cache_path)
        
        assert result == cache_data
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_load_embeddings_cache_incompatible(self, mock_config):
        """Test loading incompatible embeddings cache."""
        mock_config.model_path = self.model_path
        mock_config.model_name = "different-model"
        mock_config.max_context_length = 512
        mock_config.embedding_dimension = 384
        
        manager = ModelManager()
        
        # Create cache file with different model name
        cache_data = {
            "inputs": ["input 1", "input 2"],
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "model_name": "original-model",
            "config": {
                "max_context_length": 512,
                "embedding_dimension": 384
            }
        }
        
        cache_path = os.path.join(self.temp_dir, "cache.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Load cache should return None for incompatible cache
        result = manager.load_embeddings_cache(cache_path)
        
        assert result is None
    
    @patch('emb_model_provider.core.model_manager.config')
    def test_load_embeddings_cache_file_not_found(self, mock_config):
        """Test loading embeddings cache when file doesn't exist."""
        mock_config.model_path = self.model_path
        mock_config.model_name = self.model_name
        
        manager = ModelManager()
        
        # Try to load non-existent cache file
        cache_path = os.path.join(self.temp_dir, "nonexistent.pkl")
        result = manager.load_embeddings_cache(cache_path)
        
        assert result is None
"""
Unit tests for the configuration module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from emb_model_provider.core.config import Config


class TestConfig:
    """Test cases for the Config class."""

    def test_default_values(self):
        """Test that default values are correctly set."""
        config = Config()
        
        assert config.model_path == "D:\\models\\all-MiniLM-L12-v2"
        assert config.model_name == "all-MiniLM-L12-v2"
        assert config.max_batch_size == 32
        assert config.max_context_length == 512
        assert config.embedding_dimension == 384
        assert config.memory_limit == "2GB"
        assert config.device == "auto"
        assert config.host == "localhost"
        assert config.port == 9000
        assert config.log_level == "INFO"

    def test_from_env_with_defaults(self):
        """Test loading configuration from environment with default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env()
            
            assert config.model_path == "D:\\models\\all-MiniLM-L12-v2"
            assert config.model_name == "all-MiniLM-L12-v2"
            assert config.max_batch_size == 32
            assert config.max_context_length == 512
            assert config.embedding_dimension == 384
            assert config.memory_limit == "2GB"
            assert config.device == "auto"
            assert config.host == "localhost"
            assert config.port == 9000
            assert config.log_level == "INFO"

    def test_from_env_with_custom_values(self):
        """Test loading configuration from environment with custom values."""
        env_vars = {
            "EMB_PROVIDER_MODEL_PATH": "/custom/models/path",
            "EMB_PROVIDER_MODEL_NAME": "custom-model",
            "EMB_PROVIDER_MAX_BATCH_SIZE": "64",
            "EMB_PROVIDER_MAX_CONTEXT_LENGTH": "1024",
            "EMB_PROVIDER_EMBEDDING_DIMENSION": "768",
            "EMB_PROVIDER_MEMORY_LIMIT": "4GB",
            "EMB_PROVIDER_DEVICE": "cuda",
            "EMB_PROVIDER_HOST": "0.0.0.0",
            "EMB_PROVIDER_PORT": "8080",
            "EMB_PROVIDER_LOG_LEVEL": "DEBUG",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.from_env()
            
            assert config.model_path == "/custom/models/path"
            assert config.model_name == "custom-model"
            assert config.max_batch_size == 64
            assert config.max_context_length == 1024
            assert config.embedding_dimension == 768
            assert config.memory_limit == "4GB"
            assert config.device == "cuda"
            assert config.host == "0.0.0.0"
            assert config.port == 8080
            assert config.log_level == "DEBUG"

    def test_load_from_file(self):
        """Test loading configuration from a .env file."""
        env_content = """
EMB_PROVIDER_MODEL_PATH=/file/models/path
EMB_PROVIDER_MODEL_NAME=file-model
EMB_PROVIDER_MAX_BATCH_SIZE=16
EMB_PROVIDER_MAX_CONTEXT_LENGTH=256
EMB_PROVIDER_EMBEDDING_DIMENSION=512
EMB_PROVIDER_MEMORY_LIMIT=1GB
EMB_PROVIDER_DEVICE=cpu
EMB_PROVIDER_HOST=127.0.0.1
EMB_PROVIDER_PORT=8000
EMB_PROVIDER_LOG_LEVEL=WARNING
        """.strip()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            config = Config.load_from_file(env_file_path)
            
            assert config.model_path == "/file/models/path"
            assert config.model_name == "file-model"
            assert config.max_batch_size == 16
            assert config.max_context_length == 256
            assert config.embedding_dimension == 512
            assert config.memory_limit == "1GB"
            assert config.device == "cpu"
            assert config.host == "127.0.0.1"
            assert config.port == 8000
            assert config.log_level == "WARNING"
        finally:
            os.unlink(env_file_path)

    def test_load_from_nonexistent_file(self):
        """Test loading configuration from a non-existent .env file."""
        with pytest.raises(FileNotFoundError, match="Environment file not found"):
            Config.load_from_file("/nonexistent/path/.env")

    def test_get_model_config(self):
        """Test getting model configuration."""
        config = Config()
        model_config = config.get_model_config()
        
        expected_keys = ["model_path", "model_name", "max_context_length", "embedding_dimension", "device"]
        assert all(key in model_config for key in expected_keys)
        assert model_config["model_path"] == "D:\\models\\all-MiniLM-L12-v2"
        assert model_config["model_name"] == "all-MiniLM-L12-v2"
        assert model_config["max_context_length"] == 512
        assert model_config["embedding_dimension"] == 384
        assert model_config["device"] == "auto"

    def test_get_api_config(self):
        """Test getting API configuration."""
        config = Config()
        api_config = config.get_api_config()
        
        expected_keys = ["host", "port"]
        assert all(key in api_config for key in expected_keys)
        assert api_config["host"] == "localhost"
        assert api_config["port"] == 9000

    def test_get_processing_config(self):
        """Test getting processing configuration."""
        config = Config()
        processing_config = config.get_processing_config()
        
        expected_keys = ["max_batch_size", "max_context_length", "memory_limit"]
        assert all(key in processing_config for key in expected_keys)
        assert processing_config["max_batch_size"] == 32
        assert processing_config["max_context_length"] == 512
        assert processing_config["memory_limit"] == "2GB"

    def test_get_logging_config(self):
        """Test getting logging configuration."""
        config = Config()
        logging_config = config.get_logging_config()
        
        expected_keys = ["log_level"]
        assert all(key in logging_config for key in expected_keys)
        assert logging_config["log_level"] == "INFO"

    def test_validation_constraints(self):
        """Test that validation constraints are enforced."""
        # Test max_batch_size constraints
        with pytest.raises(ValueError):
            Config(max_batch_size=0)  # Too small
        
        with pytest.raises(ValueError):
            Config(max_batch_size=129)  # Too large
        
        # Test max_context_length constraints
        with pytest.raises(ValueError):
            Config(max_context_length=0)  # Too small
        
        with pytest.raises(ValueError):
            Config(max_context_length=2049)  # Too large
        
        # Test embedding_dimension constraints
        with pytest.raises(ValueError):
            Config(embedding_dimension=0)  # Too small
        
        # Test port constraints
        with pytest.raises(ValueError):
            Config(port=0)  # Too small
        
        with pytest.raises(ValueError):
            Config(port=65536)  # Too large
        
        # Test log_level pattern
        with pytest.raises(ValueError):
            Config(log_level="INVALID")  # Invalid pattern

    def test_valid_values(self):
        """Test that valid values are accepted."""
        # These should not raise any exceptions
        # Test individual valid values
        Config(max_batch_size=1)
        Config(max_batch_size=128)
        Config(max_context_length=1)
        Config(max_context_length=2048)
        Config(embedding_dimension=1)
        Config(port=1)
        Config(port=65535)
        Config(log_level="DEBUG")
        Config(log_level="INFO")
        Config(log_level="WARNING")
        Config(log_level="ERROR")
        
        # Test combination of valid values
        config = Config(
            max_batch_size=64,
            max_context_length=1024,
            embedding_dimension=512,
            port=8080,
            log_level="DEBUG",
        )
        
        assert config is not None
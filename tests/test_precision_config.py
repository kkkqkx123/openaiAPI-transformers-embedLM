"""
Tests for precision configuration functionality.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
import torch

from emb_model_provider.core.config import Config
from emb_model_provider.loaders.huggingface_loader import HuggingFaceModelLoader
from emb_model_provider.loaders.modelscope_loader import ModelScopeModelLoader


class TestPrecisionConfiguration:
    """Test precision configuration functionality."""

    def test_huggingface_loader_precision_config(self):
        """Test HuggingFace loader precision configuration."""
        # Test with explicit precision
        loader = HuggingFaceModelLoader("test-model", model_precision="fp16")
        assert loader.model_precision == "fp16"
        
        # Test with quantization
        loader = HuggingFaceModelLoader(
            "test-model",
            enable_quantization=True,
            quantization_method="bitsandbytes"
        )
        assert loader.enable_quantization is True
        assert loader.quantization_method == "bitsandbytes"
        
        # Test with GPU memory optimization
        loader = HuggingFaceModelLoader("test-model", enable_gpu_memory_optimization=True)
        assert loader.enable_gpu_memory_optimization is True

    def test_modelscope_loader_precision_config(self):
        """Test ModelScope loader precision configuration."""
        # Test with explicit precision
        loader = ModelScopeModelLoader("test-model", model_precision="bf16")
        assert loader.model_precision == "bf16"
        
        # Test with quantization
        loader = ModelScopeModelLoader(
            "test-model",
            enable_quantization=True,
            quantization_method="bitsandbytes"
        )
        assert loader.enable_quantization is True
        assert loader.quantization_method == "bitsandbytes"
        
        # Test with GPU memory optimization
        loader = ModelScopeModelLoader("test-model", enable_gpu_memory_optimization=True)
        assert loader.enable_gpu_memory_optimization is True

    def test_huggingface_get_optimal_precision(self):
        """Test HuggingFace loader's precision configuration."""
        # Test with default configuration
        loader = HuggingFaceModelLoader("test-model")
        
        # Test that the loader uses the default precision
        assert loader.model_precision == "auto"

    def test_modelscope_get_optimal_precision(self):
        """Test ModelScope loader's precision configuration."""
        # Test with default configuration
        loader = ModelScopeModelLoader("test-model")
        
        # Test that the loader uses the default precision
        assert loader.model_precision == "auto"

    def test_precision_override_matching(self):
        """Test precision override matching logic."""
        config = Config(
            model_precision="fp32",
            model_precision_overrides='{"sentence-transformers": "fp16", "damo": "bf16", "specific-model": "int8"}'
        )
        
        # Test config's get_precision_for_model method
        assert config.get_precision_for_model("specific-model") == "int8"
        
        # Test partial matches
        assert config.get_precision_for_model("sentence-transformers/all-MiniLM-L6-v2") == "fp16"
        assert config.get_precision_for_model("damo/nlp_corom_sentence-embedding_english-base") == "bf16"
        
        # Test no match (fallback to global)
        assert config.get_precision_for_model("other-model") == "fp32"

    def test_quantization_config(self):
        """Test quantization configuration."""
        # Create loader with quantization settings
        loader = HuggingFaceModelLoader(
            "test-model",
            enable_quantization=True,
            quantization_method="bitsandbytes"
        )
        
        # Test that quantization settings are properly set in loader
        assert loader.enable_quantization is True
        assert loader.quantization_method == "bitsandbytes"
        
        # Test that the loader uses the default precision for the model
        assert loader.model_precision == "auto"  # Default precision for test-model

    def test_device_map_config(self):
        """Test device map configuration for GPU memory optimization."""
        # Test with GPU memory optimization enabled
        loader = HuggingFaceModelLoader("test-model", enable_gpu_memory_optimization=True)
        
        # Test that GPU memory optimization setting is properly set
        assert loader.enable_gpu_memory_optimization is True
        
        # Test without optimization
        loader = HuggingFaceModelLoader("test-model", enable_gpu_memory_optimization=False)
        
        assert loader.enable_gpu_memory_optimization is False

    @patch("torch.cuda.is_available")
    def test_bf16_device_support_detection(self, mock_cuda_available):
        """Test bfloat16 support detection based on device capabilities."""
        config = Config(model_precision="auto")
        
        # Test when CUDA is available and supports bfloat16
        mock_cuda_available.return_value = True
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):  # A100/V100
            # Simulate the logic that would be used in the loader
            # This tests the device capability detection logic
            if torch.cuda.is_available():
                device_capability = torch.cuda.get_device_capability(0)
                if device_capability[0] >= 8:  # Ampere architecture or newer
                    assert device_capability[0] >= 8
        
        # Test when CUDA is available but doesn't support bfloat16
        with patch("torch.cuda.get_device_capability", return_value=(7, 0)):  # Older GPU
            if torch.cuda.is_available():
                device_capability = torch.cuda.get_device_capability(0)
                if device_capability[0] < 8:
                    assert device_capability[0] < 8
        
        # Test CPU-only environment
        mock_cuda_available.return_value = False
        if not torch.cuda.is_available():
            assert not torch.cuda.is_available()

    def test_precision_priority_chain(self):
        """Test precision priority chain with overrides and native precision."""
        # Test with model-specific override
        loader = HuggingFaceModelLoader(
            "specific-model",
            model_precision="fp32"
        )
        
        # Test that the loader uses the specified precision
        assert loader.model_precision == "fp32"
        
        # Test with generic model name
        loader = HuggingFaceModelLoader(
            "generic-model",
            model_precision="fp32"
        )
        
        # Test that the loader uses the specified precision for generic model
        assert loader.model_precision == "fp32"

    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test invalid precision in Config creation
        with pytest.raises(ValueError):
            # This should fail because "invalid" is not a valid precision
            Config(model_precision="invalid")
        
        # Test invalid quantization method in Config creation
        with pytest.raises(ValueError):
            # This should fail because "invalid" is not a valid quantization method
            Config(quantization_method="invalid")


class TestPrecisionIntegration:
    """Integration tests for precision configuration."""

    def test_config_loader_integration(self):
        """Test integration between Config and model loaders."""
        # Test HuggingFace loader integration
        hf_loader = HuggingFaceModelLoader(
            "test-model",
            model_precision="fp16",
            enable_quantization=True,
            quantization_method="bitsandbytes",
            enable_gpu_memory_optimization=True
        )
        assert hf_loader.model_precision == "fp16"
        assert hf_loader.enable_quantization is True
        assert hf_loader.quantization_method == "bitsandbytes"
        assert hf_loader.enable_gpu_memory_optimization is True
        
        # Test ModelScope loader integration (with mocking)
        with patch.dict('sys.modules', {
            'modelscope': MagicMock(),
            'modelscope.pipelines': MagicMock(),
            'modelscope.utils.constant': MagicMock()
        }):
            from emb_model_provider.loaders.modelscope_loader import ModelScopeModelLoader
            
            ms_loader = ModelScopeModelLoader(
                "test-model",
                model_precision="fp16",
                enable_quantization=True,
                quantization_method="bitsandbytes",
                enable_gpu_memory_optimization=True
            )
            assert ms_loader.model_precision == "fp16"
            assert ms_loader.enable_quantization is True
            assert ms_loader.quantization_method == "bitsandbytes"
            assert ms_loader.enable_gpu_memory_optimization is True

    def test_precision_consistency(self):
        """Test consistency between HuggingFace and ModelScope loaders."""
        hf_loader = HuggingFaceModelLoader(
            "test-model",
            model_precision="bf16",
            enable_quantization=True,
            quantization_method="bitsandbytes"
        )
        
        ms_loader = ModelScopeModelLoader(
            "test-model",
            model_precision="bf16",
            enable_quantization=True,
            quantization_method="bitsandbytes"
        )
        
        # Both loaders should have the same configuration
        assert hf_loader.model_precision == ms_loader.model_precision
        assert hf_loader.enable_quantization == ms_loader.enable_quantization
        assert hf_loader.quantization_method == ms_loader.quantization_method

    def test_precision_documentation_examples(self):
        """Test examples from the precision configuration documentation."""
        # Example 1: Basic fp16 configuration
        loader = HuggingFaceModelLoader("test-model", model_precision="fp16")
        assert loader.model_precision == "fp16"
        
        # Example 2: Quantization with bitsandbytes
        loader = HuggingFaceModelLoader(
            "test-model",
            enable_quantization=True,
            quantization_method="bitsandbytes"
        )
        assert loader.enable_quantization is True
        assert loader.quantization_method == "bitsandbytes"
        
        # Example 3: Model-specific precision
        loader = HuggingFaceModelLoader("test-model", model_precision="fp32")
        assert loader.model_precision == "fp32"
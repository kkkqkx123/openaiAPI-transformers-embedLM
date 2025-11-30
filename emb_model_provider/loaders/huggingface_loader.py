"""
HuggingFace Model Loader Implementation

This module provides the HuggingFaceModelLoader class that implements the BaseModelLoader
interface for loading models from Hugging Face Hub and transformers library.
"""

import json
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download, hf_hub_download

from .base_loader import BaseModelLoader
from ..core.config import config
from ..core.logging import get_logger, log_model_event
from ..api.exceptions import ModelLoadError


class HuggingFaceModelLoader(BaseModelLoader):
    """
    HuggingFace model loader for loading models from Hugging Face Hub.
    
    This loader supports:
    - Loading models directly from transformers
    - Downloading models from Hugging Face Hub
    - Loading models from local paths
    - Automatic device placement and optimization
    """
    
    def __init__(self, model_name: str, model_path: Optional[str] = None,
                 trust_remote_code: bool = False, **kwargs: Any):
        """
        Initialize the HuggingFace model loader.
        
        Args:
            model_name: Name of the model (e.g., 'all-MiniLM-L12-v2')
            model_path: Optional local path to the model
            **kwargs: Additional configuration options
        """
        super().__init__(model_name, model_path, trust_remote_code)
        
        # Initialize logger
        self.logger = get_logger("emb_model_provider.loaders.huggingface")
        
        # HuggingFace specific configuration
        self.transformers_model_name = kwargs.get('transformers_model_name', model_name)
        # Note: transformers_cache_dir is no longer used as HuggingFace manages its own cache
        self.transformers_trust_remote_code = kwargs.get('transformers_trust_remote_code', False)
        self.load_from_transformers = kwargs.get('load_from_transformers', False)
        
        # Precision configuration
        self.model_precision = kwargs.get('model_precision', config.get_precision_for_model(model_name))
        self.enable_quantization = kwargs.get('enable_quantization', config.enable_quantization)
        self.quantization_method = kwargs.get('quantization_method', config.quantization_method)
        self.enable_gpu_memory_optimization = kwargs.get('enable_gpu_memory_optimization', config.enable_gpu_memory_optimization)
        
        # Model components
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
    
    def load_model(self) -> tuple:
        """
        Load the model using HuggingFace transformers.
        
        This method will:
        1. If load_from_transformers is True, load directly from transformers
        2. Otherwise, try to load from local path first
        3. If not available locally, download from Hugging Face Hub and load
        
        Returns:
            tuple: (model, tokenizer) tuple
            
        Raises:
            ModelLoadError: If model loading fails
        """
        if self._model_loaded:
            self.logger.debug("Model already loaded")
            return self._model, self._tokenizer
        
        try:
            # Set device before loading model
            device = self.get_device()
            self.device = device  # type: ignore[assignment]  # Base class defines as Optional[str] but we know it's str
            
            # Check if we should load directly from transformers
            if self.load_from_transformers:
                self._load_transformers_model()
            else:
                # Try to load from local path first
                if self.is_model_available():
                    self._load_local_model()
                else:
                    # Download model from Hugging Face Hub
                    self._download_model()
                    # Load the downloaded model
                    self._load_local_model()
            
            return self._model, self._tokenizer
        
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}", self.model_name)
    
    def _load_transformers_model(self) -> None:
        """
        Load model and tokenizer directly from transformers.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            self.logger.info(f"Loading model {self.transformers_model_name} directly from transformers")
            log_model_event("load_start", self.transformers_model_name, {"source": "transformers"})
            
            # Prepare kwargs for model loading
            # Smart precision selection based on model requirements and device capabilities
            model_kwargs = {
                "torch_dtype": self._get_optimal_precision(),
                "low_cpu_mem_usage": False,
                "trust_remote_code": self.transformers_trust_remote_code
            }
            
            # Add quantization support if enabled
            if self.enable_quantization:
                if self.quantization_method == "int8":
                    model_kwargs["load_in_8bit"] = True
                    model_kwargs["torch_dtype"] = torch.float16  # Use fp16 for compute with int8 weights
                elif self.quantization_method == "int4":
                    model_kwargs["load_in_4bit"] = True
                    model_kwargs["torch_dtype"] = torch.float16  # Use fp16 for compute with int4 weights
                    
                # Add GPU memory optimization if enabled
                if self.enable_gpu_memory_optimization:
                    model_kwargs["device_map"] = "auto"
            
            # Add quantization support if enabled
            if self.enable_quantization:
                if self.quantization_method == "int8":
                    model_kwargs["load_in_8bit"] = True
                    model_kwargs["torch_dtype"] = torch.float16  # Use fp16 for compute with int8 weights
                elif self.quantization_method == "int4":
                    model_kwargs["load_in_4bit"] = True
                    model_kwargs["torch_dtype"] = torch.float16  # Use fp16 for compute with int4 weights
                    
                # Add GPU memory optimization if enabled
                if self.enable_gpu_memory_optimization:
                    model_kwargs["device_map"] = "auto"
            
            # Note: transformers_cache_dir is no longer used as HuggingFace manages its own cache
            
            # Load tokenizer
            tokenizer_kwargs = {"trust_remote_code": self.transformers_trust_remote_code}
                
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.transformers_model_name,
                **tokenizer_kwargs
            )
            
            # Load model
            self._model = AutoModel.from_pretrained(
                self.transformers_model_name,
                **model_kwargs
            )
            
            # Move model to the target device
            if self._model is not None:
                self._model.to(self.device)
                self._model.eval()
            
            self._model_loaded = True
            
            self.logger.info(f"Model loaded successfully from transformers: {self.transformers_model_name}")
            log_model_event("load_complete", self.transformers_model_name, {"source": "transformers", "device": self.device})
            
        except Exception as e:
            self.logger.error(f"Failed to load model from transformers: {e}")
            log_model_event("load_error", self.transformers_model_name, {"source": "transformers", "error": str(e)})
            raise ModelLoadError(f"Failed to load model from transformers: {e}", self.transformers_model_name)
    
    def _load_local_model(self) -> None:
        """
        Load model from local path.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            # When model_path is explicitly provided, use it directly
            # Note: cache_dir is no longer used as HuggingFace manages its own cache
            if self.model_path:
                model_path = self.model_path
            else:
                # Fallback to transformers cache if no model path provided
                # Use default transformers cache location
                model_path = os.path.join("models", self.model_name)
            self.logger.info(f"Loading model from local path: {model_path}")
            log_model_event("load_start", self.model_name, {"source": "local", "path": model_path})
            
            # Prepare kwargs for model loading
            # Smart precision selection based on model requirements and device capabilities
            model_kwargs = {
                "torch_dtype": self._get_optimal_precision(),
                "low_cpu_mem_usage": False,
                "trust_remote_code": self.transformers_trust_remote_code
            }
            
            # Load tokenizer
            tokenizer_kwargs = {"trust_remote_code": self.transformers_trust_remote_code}
                
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                **tokenizer_kwargs
            )
            
            # Load model
            self._model = AutoModel.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            # Move model to the target device
            if self._model is not None:
                self._model = self._model.to(self.device)
                self._model.eval()
            
            self._model_loaded = True
            
            self.logger.info(f"Model loaded successfully from local path: {model_path}")
            log_model_event("load_complete", self.model_name, {"source": "local", "path": model_path, "device": self.device})
            
        except Exception as e:
            self.logger.error(f"Failed to load model from local path: {e}")
            log_model_event("load_error", self.model_name, {"source": "local", "error": str(e)})
            raise ModelLoadError(f"Failed to load model from local path: {e}", self.model_name)
    
    def _get_optimal_precision(self) -> torch.dtype:
        """
        Determine the optimal precision for model loading based on actual model requirements.
        
        This method considers in order of priority:
        1. User configuration preferences (including model-specific overrides)
        2. Model's native precision requirements from config
        3. Device capabilities and performance characteristics
        4. Memory efficiency vs precision trade-offs
        5. Quantization support if enabled
        
        Returns:
            torch.dtype: Optimal precision dtype
        """
        # 1. Check user configuration first (highest priority)
        if self.model_precision != "auto":
            if self.model_precision == "fp16":
                return torch.float16
            elif self.model_precision == "fp32":
                return torch.float32
            elif self.model_precision == "bf16":
                if self.device and self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    self.logger.warning("bfloat16 not supported on this device, falling back to fp16")
                    return torch.float16
            elif self.model_precision in ["int8", "int4"]:
                # Quantization is handled separately in model loading kwargs
                # For compute dtype, use fp16 for better performance
                return torch.float16
        
        # 2. Try to detect model's native precision from config
        try:
            # Attempt to load model config to check native precision
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(self.model_name)
            
            # Check if model has specific dtype requirements
            if hasattr(model_config, 'torch_dtype'):
                native_dtype = model_config.torch_dtype
                if native_dtype is not None:
                    self.logger.info(f"Using model's native precision: {native_dtype}")
                    # Ensure we return a valid torch dtype
                    if isinstance(native_dtype, torch.dtype):
                        return native_dtype
                    # Fallback to default if native dtype is not a torch dtype
            
            # Check for quantization config that might indicate precision preferences
            if hasattr(model_config, 'quantization_config'):
                quant_config = model_config.quantization_config
                if hasattr(quant_config, 'bnb_4bit_compute_dtype'):
                    compute_dtype = quant_config.bnb_4bit_compute_dtype
                    if isinstance(compute_dtype, torch.dtype):
                        return compute_dtype
                    # Fallback to fp16 for quantization
                    return torch.float16
        
        except Exception as e:
            # If config loading fails, fall back to heuristic approach
            self.logger.debug(f"Could not load model config for precision detection: {e}")
        
        # 3. Device-based heuristics (fallback strategy)
        # For CPU devices, always use fp32 for better compatibility
        if self.device == "cpu":
            return torch.float32
        
        # 4. Model-specific heuristics for GPU devices
        # Check model name for known fp16-compatible models
        fp16_compatible_models = [
            "all-MiniLM", "paraphrase", "distilbert", "mpnet", "roberta", "sentence-transformers"
        ]
        
        model_lower = self.model_name.lower()
        if any(compatible in model_lower for compatible in fp16_compatible_models):
            # These embedding models are known to work well with fp16
            return torch.float16
        
        # 5. Default to fp32 for unknown models to ensure precision
        # Most embedding models benefit from higher precision
        return torch.float32
    
    def _download_model(self) -> None:
        """
        Download model from Hugging Face Hub.
        
        Raises:
            ModelLoadError: If download fails
        """
        try:
            self.logger.info(f"Downloading model {self.model_name} from Hugging Face Hub")
            log_model_event("download_start", self.model_name, {"source": "huggingface_hub"})
            
            # Determine the cache directory for download
            # Note: cache_dir is no longer used as HuggingFace manages its own cache
            # Priority: model_path (if it's a directory) > transformers_cache_dir > default
            if self.model_path and os.path.isdir(self.model_path):
                cache_dir = self.model_path
            else:
                # Use transformers cache as HuggingFace manages its own cache
                # Use default transformers cache location
                cache_dir = os.path.join("models", self.model_name)
            os.makedirs(cache_dir, exist_ok=True)
            
            # Use the model name directly as specified in the configuration
            snapshot_download(
                repo_id=self.model_name,
                cache_dir=cache_dir,
                local_dir=cache_dir,
                local_dir_use_symlinks=False
            )
            
            self.logger.info(f"Model downloaded successfully to: {cache_dir}")
            log_model_event("download_complete", self.model_name, {"source": "huggingface_hub", "path": cache_dir})
            
        except Exception as e:
            self.logger.error(f"Failed to download model from Hugging Face Hub: {e}")
            log_model_event("download_error", self.model_name, {"source": "huggingface_hub", "error": str(e)})
            raise ModelLoadError(f"Failed to download model from Hugging Face Hub: {e}", self.model_name)
    
    def is_model_available(self) -> bool:
        """
        Check if the model is available locally.
        
        Returns:
            True if model is available locally, False otherwise
        """
        if self.load_from_transformers:
            return True  # Always available when loading from transformers
        
        # Use default transformers cache location since transformers_cache_dir is no longer used
        model_path = self.model_path or os.path.join("models", self.model_name)
        
        if not os.path.exists(model_path):
            return False
        
        # Check for required files
        required_files = ["config.json", "pytorch_model.bin"]
        return all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            dict: Model information including name, path, device, etc.
        """
        if not self._model_loaded:
            return {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "device": self.device,
                "loaded": False,
                "load_from_transformers": self.load_from_transformers,
                "source": "transformers" if self.load_from_transformers else "huggingface" if not self.model_path else "local",
            }
        
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "loaded": True,
            "load_from_transformers": self.load_from_transformers,
            "max_context_length": config.max_context_length,
            "embedding_dimension": config.embedding_dimension,
            "vocab_size": self._tokenizer.vocab_size if self._tokenizer else 0,
            "model_type": self._model.config.model_type if self._model else "",
            # Note: cache_dir is no longer included as it's managed by the model libraries
            "source": "transformers" if self.load_from_transformers else "huggingface" if not self.model_path else "local",
        }
    
    def get_model_size(self) -> int:
        """
        Get the size of the model in bytes.
        
        Returns:
            int: Model size in bytes
        """
        if not self._model_loaded:
            return 0
        
        try:
            if self._model is None:
                return 0
                
            # Calculate model parameters size
            param_size = 0
            for param in self._model.parameters():
                param_size += param.nelement() * param.element_size()
            
            # Calculate buffer size
            buffer_size = 0
            for buffer in self._model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            return param_size + buffer_size
        except Exception as e:
            self.logger.warning(f"Failed to calculate model size: {e}")
            return 0
    
    def cleanup(self) -> None:
        """
        Clean up model resources.
        """
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        # Clear CUDA cache if using GPU
        if self.device and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        self._model_loaded = False
        self.logger.info(f"Model {self.model_name} cleaned up")
    
    def get_model(self) -> Optional[Any]:
        """Get the loaded model."""
        return self._model
    
    def get_tokenizer(self) -> Optional[Any]:
        """Get the loaded tokenizer."""
        return self._tokenizer
    
    def validate_model(self) -> bool:
        """
        Validate that the loaded model is functional.
        
        Returns:
            bool: True if the model is valid, False otherwise
        """
        if self._model is None or self._tokenizer is None:
            return False
            
        try:
            # Test with a simple input
            test_input = "Hello, world!"
            tokens = self._tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = self._model(**tokens)
                
            # Check if we get embeddings
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
                if embeddings.shape[0] > 0 and embeddings.shape[-1] > 0:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded
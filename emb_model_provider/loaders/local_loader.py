"""
Local Model Loader Implementation

This module provides the LocalModelLoader class that implements the BaseModelLoader
interface for loading models from local paths.
"""

import os
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModel, AutoTokenizer

from .base_loader import BaseModelLoader
from ..core.config import config
from ..core.logging import get_logger, log_model_event
from ..api.exceptions import ModelLoadError


class LocalModelLoader(BaseModelLoader):
    """
    Local model loader for loading models from local file paths.
    
    This loader specifically handles models that are already downloaded to a local path,
    providing a clear separation from remote model loading via HuggingFace or ModelScope.
    """
    
    def __init__(self, model_name: str, model_path: str,
                 trust_remote_code: bool = False, **kwargs: Any):
        """
        Initialize the local model loader.
        
        Args:
            model_name: Name of the model (e.g., 'all-MiniLM-L12-v2')
            model_path: Local path to the model directory
            cache_dir: Directory to cache downloaded models (not used in local loader)
            trust_remote_code: Whether to trust remote code in model files
            **kwargs: Additional configuration options
        """
        super().__init__(model_name, model_path, trust_remote_code, **kwargs)
        
        # Initialize logger
        self.logger = get_logger("emb_model_provider.loaders.local")
        
        # Local-specific configuration
        precision_config = config.get_precision_for_model(model_name)
        self.model_storage_precision = kwargs.get('model_storage_precision', precision_config['storage_precision'])
        self.model_compute_precision = kwargs.get('model_compute_precision', precision_config['compute_precision'])
        self.storage_config = kwargs.get('storage_config', getattr(config, 'storage_config', {}))
        self.compute_config = kwargs.get('compute_config', getattr(config, 'compute_config', {}))
        self.quantization_method = kwargs.get('quantization_method', config.quantization_method)
        self.enable_gpu_memory_optimization = kwargs.get('enable_gpu_memory_optimization', config.enable_gpu_memory_optimization)
        
        # Model components
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
    
    def load_model(self) -> tuple:
        """
        Load the model from local path.
        
        This method loads the model and tokenizer from the specified local path.
        
        Returns:
            tuple: (model, tokenizer) tuple
            
        Raises:
            ModelLoadError: If model loading fails
        """
        if self._model_loaded:
            self.logger.debug("Model already loaded")
            return self._model, self._tokenizer
        
        if self.model_path is None:
            raise ModelLoadError("model_path cannot be None for LocalModelLoader", self.model_name)
        
        try:
            self.logger.info(f"Loading model from local path: {self.model_path}")
            log_model_event("load_start", self.model_name, {"source": "local", "path": self.model_path})
            
            # Set device before loading model
            device = self.get_device()
            self.device = device  # type: ignore[assignment] # Base class defines as Optional[str] but we know it's str
            
            # Prepare kwargs for model loading
            model_kwargs = {
                "torch_dtype": self._get_optimal_precision(),
                "low_cpu_mem_usage": False,
                "trust_remote_code": self.trust_remote_code
            }
            
            # Handle storage precision (quantization)
            if self.model_storage_precision in ["int4", "int8", "fp4", "nf4"]:
                if self.model_storage_precision in ["int8", "int4"]:
                    if self.model_storage_precision == "int8":
                        model_kwargs["load_in_8bit"] = True
                    elif self.model_storage_precision == "int4":
                        model_kwargs["load_in_4bit"] = True

                    # Set the compute precision based on our compute precision config
                    compute_dtype = self._get_optimal_precision()
                    model_kwargs["torch_dtype"] = compute_dtype

                    # Add GPU memory optimization if enabled
                    if self.enable_gpu_memory_optimization:
                        model_kwargs["device_map"] = "auto"
                elif self.model_storage_precision in ["fp4", "nf4"]:
                    # Handle 4-bit float quantization with bitsandbytes
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type=self.model_storage_precision,
                        bnb_4bit_compute_dtype=self._get_optimal_precision()
                    )
                    model_kwargs["quantization_config"] = bnb_config
            elif self.model_storage_precision != "auto":
                # If storage precision is explicitly set to a floating point type
                if self.model_storage_precision == "fp16":
                    model_kwargs["torch_dtype"] = torch.float16
                elif self.model_storage_precision == "fp32":
                    model_kwargs["torch_dtype"] = torch.float32
                elif self.model_storage_precision == "bf16":
                    model_kwargs["torch_dtype"] = torch.bfloat16
                elif self.model_storage_precision == "fp8":
                    model_kwargs["torch_dtype"] = torch.float8_e4m3fn  # or torch.float8_e5m2
            
            # Load tokenizer
            tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                **tokenizer_kwargs
            )
            
            # Load model
            self._model = AutoModel.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # Move model to the target device
            if self._model is not None:
                self._model = self._model.to(self.device)
                self._model.eval()
            
            self._model_loaded = True
            
            self.logger.info(f"Model loaded successfully from local path: {self.model_path}")
            log_model_event("load_complete", self.model_name, {"source": "local", "path": self.model_path, "device": self.device})
            
            return self._model, self._tokenizer
            
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

        Returns:
            torch.dtype: Optimal compute precision dtype
        """
        # 1. Check user compute configuration first (highest priority)
        if self.model_compute_precision != "auto":
            if self.model_compute_precision == "fp16":
                return torch.float16
            elif self.model_compute_precision == "fp32":
                return torch.float32
            elif self.model_compute_precision == "bf16":
                if self.device and self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    self.logger.warning("bfloat16 not supported on this device, falling back to fp16")
                    return torch.float16
            elif self.model_compute_precision == "tf32":
                # TF32 is only available on Ampere and later GPUs
                if self.device and self.device.startswith("cuda"):
                    major, _ = torch.cuda.get_device_capability()
                    if major >= 8:
                        return torch.float32  # TF32 is used by enabling it globally, not as a dtype
                    else:
                        self.logger.warning("TF32 is not available on this GPU, falling back to fp16")
                        return torch.float16
                else:
                    self.logger.warning("TF32 is only available on CUDA, falling back to fp16")
                    return torch.float16
            else:
                # For other compute precisions, default to fp16
                return torch.float16
        
        # 2. Try to detect model's native precision from config
        try:
            # Attempt to load model config to check native precision
            from transformers import AutoConfig
            if self.model_path is None:
                raise ValueError("model_path cannot be None")
            model_config = AutoConfig.from_pretrained(self.model_path)
            
            # Check if model has specific dtype requirements
            if hasattr(model_config, 'torch_dtype'):
                native_dtype = model_config.torch_dtype
                if native_dtype is not None:
                    self.logger.info(f"Using model's native precision: {native_dtype}")
                    # Ensure we return a valid torch dtype
                    if isinstance(native_dtype, torch.dtype):
                        return native_dtype
                    # Fallback to default if native dtype is not a torch dtype
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
    
    def is_model_available(self) -> bool:
        """
        Check if the model is available locally.
        
        Returns:
            True if model is available locally, False otherwise
        """
        if not self.model_path or not os.path.exists(self.model_path):
            return False
        
        # Check for required files
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"]
        return all(os.path.exists(os.path.join(self.model_path, f)) for f in required_files)
    
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
                "loader": "local",
                "source": "local",
            }
        
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "loaded": True,
            "loader": "local",
            "max_context_length": config.max_context_length,
            "embedding_dimension": config.embedding_dimension,
            "vocab_size": self._tokenizer.vocab_size if self._tokenizer else 0,
            "model_type": self._model.config.model_type if self._model else "",
            "source": "local",
        }
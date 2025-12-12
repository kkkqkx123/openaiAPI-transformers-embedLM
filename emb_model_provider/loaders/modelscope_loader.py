"""
ModelScope Model Loader Implementation

This module provides the ModelScopeModelLoader class that implements the BaseModelLoader
interface for loading models from ModelScope hub.
"""

import os
from typing import Optional, Dict, Any, List, cast
from pathlib import Path
import torch

from .base_loader import BaseModelLoader
from ..core.config import Config, config
from ..core.logging import get_logger, log_model_event
from ..api.exceptions import ModelLoadError

# Import modelscope for type checking is done inside functions to support offline testing


class ModelScopeModelLoader(BaseModelLoader):
    """
    ModelScope model loader for loading models from ModelScope Hub.
    
    This loader supports:
    - Loading models from ModelScope Hub
    - Automatic model mapping from HuggingFace to ModelScope equivalents
    - Pipeline-based inference for sentence embeddings
    - Fallback to HuggingFace when ModelScope fails (if configured)
    """
    
    def __init__(self, model_name: str, model_path: Optional[str] = None,
                 trust_remote_code: bool = False,
                 config_instance: Optional[Config] = None, **kwargs: Any):
        """
        Initialize the ModelScope model loader.
        
        Args:
            model_name: Name of the model (e.g., 'all-MiniLM-L12-v2')
            model_path: Optional local path to the model
            config_instance: Optional config instance to use (defaults to global config)
            **kwargs: Additional configuration options
        """
        super().__init__(model_name, model_path, trust_remote_code, **kwargs)
        
        # Use provided config instance or fall back to global config
        self.config = config_instance or config
        
        # Initialize logger
        self.logger = get_logger("emb_model_provider.loaders.modelscope")
        
        # ModelScope specific configuration
        self.modelscope_model_name = kwargs.get('modelscope_model_name', self.config.modelscope_model_name or model_name)
        # Note: modelscope_cache_dir is no longer used as ModelScope manages its own cache
        self.modelscope_trust_remote_code = kwargs.get('modelscope_trust_remote_code', self.config.modelscope_trust_remote_code)
        self.modelscope_revision = kwargs.get('modelscope_revision', self.config.modelscope_revision)

        # Precision configuration from kwargs or config
        precision_config = self.config.get_precision_for_model(model_name)
        self.model_storage_precision = kwargs.get('model_storage_precision', precision_config['storage_precision'])
        self.model_compute_precision = kwargs.get('model_compute_precision', precision_config['compute_precision'])
        self.storage_config = kwargs.get('storage_config', getattr(self.config, 'storage_config', {}))
        self.compute_config = kwargs.get('compute_config', getattr(self.config, 'compute_config', {}))
        self.quantization_method = kwargs.get('quantization_method', getattr(self.config, 'quantization_method', 'int8'))
        self.enable_gpu_memory_optimization = kwargs.get('enable_gpu_memory_optimization',
                                                        getattr(self.config, 'enable_gpu_memory_optimization', False))
        
        # Model components
        self._pipeline = None
        self._model_loaded = False
    
    def load_model(self) -> tuple:
        """
        Load the model using ModelScope pipeline.
        
        Returns:
            tuple: (pipeline, None) tuple for ModelScope compatibility
            
        Raises:
            ModelLoadError: If model loading fails
        """
        if self._model_loaded:
            self.logger.debug("Model already loaded")
            return self._pipeline, None
        
        try:
            # Try to import ModelScope
            try:
                from modelscope.pipelines import pipeline  # type: ignore[import-untyped]
                from modelscope.utils.constant import Tasks  # type: ignore[import-untyped]
            except ImportError as e:
                self.logger.error(f"ModelScope not installed: {e}")
                raise ModelLoadError(
                    "ModelScope not installed. Install with: pip install modelscope",
                    self.model_name
                )
            
            # Use the model name directly since ModelScopeModelLoader is only used when source is 'modelscope'
            modelscope_id = self.model_name
            
            self.logger.info(f"Loading ModelScope model: {modelscope_id}")
            log_model_event("load_start", self.model_name, {"source": "modelscope", "model_id": modelscope_id})
            
            # Set device before loading model
            self.device = self.get_device()  # type: ignore[assignment]  # Base class defines as Optional[str] but we know it's str
            
            # Determine device for ModelScope
            device = "gpu" if self.device and self.device.startswith("cuda") else "cpu"
            
            # Prepare cache directory
            # Use default ModelScope cache directory since we're not managing it explicitly
            cache_dir = os.path.expanduser("~/.cache/modelscope/hub/")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Smart precision selection based on model requirements and device capabilities
            precision_config = self._get_optimal_precision()
            
            # Load the pipeline with optimal precision
            pipeline_kwargs = {
                "task": Tasks.sentence_embedding,
                "model": modelscope_id,
                "device": device,
                "model_revision": self.modelscope_revision,
                "trust_remote_code": self.modelscope_trust_remote_code,
                "cache_dir": cache_dir
            }
            
            # First handle storage precision (quantization)
            if self.model_storage_precision in ["int4", "int8", "fp4", "nf4"]:
                if self.model_storage_precision in ["int8", "int4"]:
                    if self.model_storage_precision == "int8":
                        pipeline_kwargs["load_in_8bit"] = True
                    elif self.model_storage_precision == "int4":
                        pipeline_kwargs["load_in_4bit"] = True

                    # Set the compute precision based on our compute precision config
                    compute_dtype = self._get_optimal_precision()
                    if compute_dtype:
                        pipeline_kwargs["torch_dtype"] = compute_dtype

                    # Add GPU memory optimization if enabled
                    if self.enable_gpu_memory_optimization:
                        pipeline_kwargs["device_map"] = "auto"
                elif self.model_storage_precision in ["fp4", "nf4"]:
                    # Handle 4-bit float quantization with bitsandbytes
                    from transformers import BitsAndBytesConfig
                    compute_dtype = self._get_optimal_precision()
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type=self.model_storage_precision,
                        bnb_4bit_compute_dtype=compute_dtype or torch.float16
                    )
                    pipeline_kwargs["quantization_config"] = bnb_config
            elif self.model_storage_precision != "auto":
                # If storage precision is explicitly set to a floating point type
                if self.model_storage_precision == "fp16":
                    pipeline_kwargs["torch_dtype"] = torch.float16
                elif self.model_storage_precision == "fp32":
                    pipeline_kwargs["torch_dtype"] = torch.float32
                elif self.model_storage_precision == "bf16":
                    pipeline_kwargs["torch_dtype"] = torch.bfloat16
                elif self.model_storage_precision == "fp8":
                    pipeline_kwargs["torch_dtype"] = torch.float8_e4m3fn  # or torch.float8_e5m2
            else:
                # Use compute precision only if no storage precision specified
                compute_dtype = self._get_optimal_precision()
                if compute_dtype:
                    pipeline_kwargs["torch_dtype"] = compute_dtype

            # Add GPU memory optimization if enabled
            if self.enable_gpu_memory_optimization:
                pipeline_kwargs["device_map"] = "auto"
            
            self._pipeline = pipeline(**pipeline_kwargs)
            
            self._model_loaded = True
            
            self.logger.info(f"ModelScope model loaded successfully: {modelscope_id}")
            log_model_event("load_complete", self.model_name, {"source": "modelscope", "model_id": modelscope_id, "device": device})
            
            return self._pipeline, None
            
        except Exception as e:
            self.logger.error(f"Failed to load model from ModelScope: {e}")
            log_model_event("load_error", self.model_name, {"source": "modelscope", "error": str(e)})
            raise ModelLoadError(f"Failed to load ModelScope model: {e}", self.model_name)
    
    def _get_optimal_precision(self) -> Optional[torch.dtype]:
        """
        Determine the optimal precision for model loading based on actual model requirements.

        This method considers in order of priority:
        1. User configuration preferences (including model-specific overrides)
        2. Model's native precision requirements from config
        3. Device capabilities and performance characteristics
        4. Memory efficiency vs precision trade-offs

        Returns:
            Optional[torch.dtype]: Optimal compute precision dtype, or None if auto-selection should be used
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
            from modelscope import AutoConfig
            model_config = AutoConfig.from_pretrained(self.modelscope_model_name or self.model_name)
            
            # Check if model has specific dtype requirements
            if hasattr(model_config, 'torch_dtype'):
                native_dtype = model_config.torch_dtype
                if native_dtype is not None:
                    self.logger.info(f"Using model's native precision: {native_dtype}")
                    return cast(torch.dtype, native_dtype)  # type: ignore[no-any-return]
            
            # Check for quantization config that might indicate precision preferences
            if hasattr(model_config, 'quantization_config'):
                quant_config = model_config.quantization_config
                if hasattr(quant_config, 'bnb_4bit_compute_dtype'):
                    return cast(torch.dtype, quant_config.bnb_4bit_compute_dtype)  # type: ignore[no-any-return]
        
        except ImportError:
            # If modelscope is not available, skip config loading
            self.logger.debug("ModelScope not available, skipping config loading for precision detection")
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
       # Check if model_path is provided and exists
       if self.model_path and os.path.exists(self.model_path):
           # Check for required model files
           required_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"]
           return all(os.path.exists(os.path.join(self.model_path, f)) for f in required_files)
       
       # For ModelScope models, we can't reliably check availability in offline mode
       # So we return True to allow the model loading to proceed and fail gracefully if needed
       return True
    
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
                "loader": "modelscope",
                "modelscope_model_name": self.modelscope_model_name,
                "source": "modelscope" if not self.model_path else "local",
            }
        
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "loaded": True,
            "loader": "modelscope",
            "modelscope_model_name": self.modelscope_model_name,
            "max_context_length": config.max_context_length,
            "embedding_dimension": config.embedding_dimension,
            # Note: cache_dir is no longer included as it's managed by the model libraries
            "source": "modelscope" if not self.model_path else "local",
            "pipeline_type": "sentence_embedding",
        }
    
    def generate_embeddings(self, inputs: List[str]) -> List[List[float]]:
        """
        Generate embeddings using ModelScope pipeline.
        
        Args:
            inputs: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            ModelLoadError: If model is not loaded
            ValueError: If inputs are invalid
        """
        if not self._model_loaded or self._pipeline is None:
            raise ModelLoadError("Model is not loaded. Call load_model() first.", self.model_name)
        
        if not inputs:
            raise ValueError("Inputs cannot be empty")
        
        try:
            # Handle newlines for consistent input
            texts = [text.replace("\n", " ") for text in inputs]
            inputs_dict = {"source_sentence": texts}
            
            # Generate embeddings using pipeline
            result = self._pipeline(input=inputs_dict)
            
            # Handle different result formats
            if isinstance(result, dict) and 'text_embedding' in result:
                embeddings = result['text_embedding']
                # Convert to list if it's a tensor
                if hasattr(embeddings, 'tolist'):
                    embeddings = embeddings.tolist()
            elif isinstance(result, list):
                # Direct list of embeddings
                embeddings = result
            else:
                raise ValueError(f"Unexpected result format from ModelScope pipeline: {type(result)}")
            
            self.logger.debug(f"Generated embeddings for {len(inputs)} inputs using ModelScope")

            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings with ModelScope: {e}")
            raise ModelLoadError(f"Failed to generate embeddings: {e}", self.model_name)
    
    def generate_batch_embeddings(self, inputs: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for a large list of inputs using batch processing.
        
        Args:
            inputs: List of texts to embed
            batch_size: Optional batch size (uses config default if not provided)
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not self._model_loaded:
            raise ModelLoadError("Model is not loaded. Call load_model() first.", self.model_name)
        
        if not inputs:
            raise ValueError("Inputs cannot be empty")
        
        # Use provided batch size or default from config
        batch_size = batch_size or config.max_batch_size
        
        # Process inputs in batches
        all_embeddings = []
        
        self.logger.info(f"Processing {len(inputs)} inputs in batches of {batch_size}")
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_embeddings = self.generate_embeddings(batch_inputs)
            all_embeddings.extend(batch_embeddings)
        
        self.logger.info(f"Generated embeddings for {len(all_embeddings)} inputs using ModelScope")
        
        return all_embeddings
    
    def get_model_size(self) -> int:
        """
        Get the size of the model in bytes.
        
        Returns:
            int: Model size in bytes (0 if not loaded)
        """
        # ModelScope doesn't provide direct access to model parameters
        # This is a placeholder - in practice, we'd need to estimate based on model type
        return 0
    
    def cleanup(self) -> None:
        """
        Clean up model resources.
        """
        if self._pipeline is not None:
            # ModelScope pipeline cleanup - just delete the reference
            # The actual cleanup will be handled by Python's garbage collector
            del self._pipeline
            self._pipeline = None
        
        self._model_loaded = False
        self.logger.info(f"ModelScope model {self.model_name} cleaned up")
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded
    
    @property
    def pipeline(self) -> Optional[Any]:
        """Get the loaded ModelScope pipeline."""
        return self._pipeline
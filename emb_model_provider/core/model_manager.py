from typing import Dict, List, Optional, Any, Union, Tuple, cast
import pickle
import threading
import torch
from tqdm import tqdm

# Local imports
from .config import Config
from .logging import get_logger, log_model_event
from ..loaders import BaseModelLoader, HuggingFaceModelLoader, ModelScopeModelLoader

# Get config instance
config = Config()

# Get logger
logger = get_logger(__name__)

# Global model manager instances
_model_managers: Dict[str, ModelManager] = {}
_model_manager_lock = threading.Lock()

# Default model manager for backward compatibility
_default_model_manager: Optional[ModelManager] = None


def get_model_manager(model_alias: Optional[str] = None) -> ModelManager:
    """Get a model manager instance for the specified model alias."""
    global _model_managers, _default_model_manager
    
    # Use default model if no alias specified
    if model_alias is None:
        if _default_model_manager is None:
            _default_model_manager = ModelManager()
        return _default_model_manager
    
    with _model_manager_lock:
        if model_alias not in _model_managers:
            # Check if model can be loaded (preloaded or dynamic loading enabled)
            if not config.enable_dynamic_model_loading and not config.is_model_preloaded(model_alias):
                raise RuntimeError(f"Model '{model_alias}' is not preloaded and dynamic loading is disabled")
            
            # Create new model manager for this alias
            _model_managers[model_alias] = ModelManager(model_alias=model_alias)
        
        return _model_managers[model_alias]

def preload_models() -> None:
    """Preload all models specified in the preload configuration."""
    global _model_managers
    
    preload_models = config.get_preload_models()
    if not preload_models:
        logger.info("No models specified for preloading")
        return
    
    logger.info(f"Preloading {len(preload_models)} models: {', '.join(preload_models)}")
    
    with _model_manager_lock:
        for model_alias in preload_models:
            if model_alias not in _model_managers:
                try:
                    manager = ModelManager(model_alias=model_alias)
                    manager.load_model()
                    _model_managers[model_alias] = manager
                    logger.info(f"Successfully preloaded model '{model_alias}'")
                except Exception as e:
                    logger.error(f"Failed to preload model '{model_alias}': {e}")
                    raise RuntimeError(f"Failed to preload model '{model_alias}': {e}")

def unload_model(model_alias: str) -> None:
    """Unload a specific model and free its resources."""
    global _model_managers
    
    with _model_manager_lock:
        if model_alias in _model_managers:
            _model_managers[model_alias].unload_model()
            del _model_managers[model_alias]
            logger.info(f"Unloaded model '{model_alias}'")

def unload_all_models() -> None:
    """Unload all models and free all resources."""
    global _model_managers, _default_model_manager
    
    with _model_manager_lock:
        for model_alias, manager in list(_model_managers.items()):
            manager.unload_model()
            del _model_managers[model_alias]
        
        if _default_model_manager:
            _default_model_manager.unload_model()
            _default_model_manager = None
        
        logger.info("All models unloaded")

def get_loaded_models() -> List[str]:
    """Get list of currently loaded model aliases."""
    global _model_managers
    
    with _model_manager_lock:
        loaded_models = []
        for model_alias, manager in _model_managers.items():
            if manager.is_model_loaded():
                loaded_models.append(model_alias)
        
        return loaded_models


class ModelManager:
    """
    Model manager that uses loader pattern to handle different model sources.
    Supports multiple models with different loaders and dynamic loading.
    """
    
    def __init__(self, model_alias: Optional[str] = None, loader: Optional[BaseModelLoader] = None):
        """
        Initialize the model manager for a specific model.
        
        Args:
            model_alias: Model alias to manage. If None, uses default model.
            loader: Optional loader instance. If not provided, will create based on config.
        """
        self.model_alias = model_alias or config.model_name
        
        # Get model configuration
        model_config = config.get_model_info(self.model_alias)
        if not model_config:
            raise ValueError(f"Model configuration not found for alias: {self.model_alias}")
        
        self.model_name = model_config["name"]
        self.model_path = model_config["path"]
        self.model_source = model_config["source"]
        self.model_precision = model_config["precision"]
        # Note: cache_dir is no longer used as HuggingFace and ModelScope manage their own caches
        self.trust_remote_code = model_config["trust_remote_code"]
        self.revision = model_config["revision"]
        self.fallback_to_huggingface = model_config["fallback_to_huggingface"]
        self.load_from_transformers = model_config["load_from_transformers"]
        
        # Global configuration
        self.enable_offline_mode = config.enable_offline_mode
        self.enable_path_priority = config.enable_path_priority
        # Note: fallback_to_cache_dir configuration has been removed as it's no longer needed
        
        # Note: cache_dir and cache_manager are no longer used as HuggingFace and ModelScope manage their own caches
        
        # Global configuration
        self.device = config.device
        self.max_batch_size = config.max_batch_size
        self.max_context_length = config.max_context_length
        self.embedding_dimension = config.embedding_dimension
        
        # Model and tokenizer references
        self._model: Optional[Any] = None
        
        # Model and tokenizer references
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._model_loaded = False
        self._loader: Optional[BaseModelLoader] = loader
        
        logger.info(f"ModelManager initialized for model '{self.model_alias}' with source={self.model_source}, device={self.device}")
    
    def _create_loader(self) -> BaseModelLoader:
        """
        Create appropriate loader based on configuration.
        
        Returns:
            BaseModelLoader: Configured loader instance
        """
        import os
        from ..loaders.local_loader import LocalModelLoader
        
        # Check if we should use local loader based on model path
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Using local model path for model '{self.model_alias}' from {self.model_path}")
            return LocalModelLoader(
                model_name=self.model_name,
                model_path=self.model_path,
                trust_remote_code=self.trust_remote_code,
                device=self.device
            )
        
        # Create loader based on model source
        if self.model_source == "transformers":
            logger.info(f"Creating HuggingFaceModelLoader for model '{self.model_alias}'")
            return HuggingFaceModelLoader(
                model_name=self.model_name,
                model_path=self.model_path,
                trust_remote_code=self.trust_remote_code,
                device=self.device,
                load_from_transformers=self.load_from_transformers
            )
        elif self.model_source == "modelscope":
            logger.info(f"Creating ModelScopeModelLoader for model '{self.model_alias}'")
            return ModelScopeModelLoader(
                model_name=self.model_name,
                model_path=self.model_path,
                device=self.device,
                config_instance=config
            )
        else:
            raise ValueError(f"Unsupported model source for model '{self.model_alias}': {self.model_source}")
    
    def load_model(self) -> None:
        """Load the model and tokenizer using the configured loader."""
        if self._model_loaded:
            return
            
        if self._loader is None:
            self._loader = self._create_loader()
            
        try:
            logger.info(f"Loading model '{self.model_alias}': {self.model_name}")
            log_model_event("load_start", self.model_name, {"loader": type(self._loader).__name__})
            
            self._model, self._tokenizer = self._loader.load_model()
            self._model_loaded = True
            
            logger.info(f"Model '{self.model_alias}' loaded successfully: {self.model_name}")
            log_model_event("load_complete", self.model_name, {"loader": type(self._loader).__name__})
            
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_alias}' ({self.model_name}): {e}")
            log_model_event("load_error", self.model_name, {"error": str(e)})
            raise RuntimeError(f"Failed to load model: {e}")
    
    def is_model_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model_loaded
    
    def unload_model(self) -> None:
        """Unload the model and free memory."""
        if self._model_loaded:
            logger.info(f"Unloading model '{self.model_alias}'")
            self._model = None
            self._tokenizer = None
            self._model_loaded = False
            # Force garbage collection
            import gc
            gc.collect()
            if self.device == "cuda":
                import torch
                torch.cuda.empty_cache()
    
    def _tokenize_inputs(self, inputs: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            inputs: Single text or list of texts
            
        Returns:
            Dict containing tokenized inputs
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded")
        
        # Convert single input to list
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Tokenize inputs
        tokenizer = self._tokenizer
        encoded = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_context_length,
            return_tensors="pt"
        )
        
        return cast(Dict[str, torch.Tensor], encoded)
    
    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on model outputs.
        
        Args:
            model_output: Model output tensor
            attention_mask: Attention mask tensor
            
        Returns:
            Pooled embeddings tensor
        """
        # Get token embeddings
        token_embeddings = model_output[0]  # First element contains token embeddings
        
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings for each token (excluding padding)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Count non-padding tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Calculate mean pooling
        return sum_embeddings / sum_mask
    
    def generate_embeddings(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for input texts.
        
        Args:
            inputs: Single text or list of texts
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not self._model_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        if not inputs:
            raise ValueError("Inputs cannot be empty")
        
        # Convert single input to list
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Check batch size limit
        if len(inputs) > self.max_batch_size:
            logger.warning(f"Input batch size {len(inputs)} exceeds maximum {self.max_batch_size}")
            raise ValueError(f"Batch size {len(inputs)} exceeds maximum {self.max_batch_size}")
        
        try:
            # Tokenize inputs
            encoded = self._tokenize_inputs(inputs)
            
            # Move to device
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            
            # Generate embeddings
            if self._model is None:
                raise RuntimeError("Model is not loaded")
            
            with torch.no_grad():
                model_output = self._model(**encoded)
            
            # Perform mean pooling
            embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
            
            # Convert to list
            embeddings_list = embeddings.cpu().numpy().tolist()
            
            logger.debug(f"Generated embeddings for {len(inputs)} inputs")
            
            return cast(List[List[float]], embeddings_list)
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    def generate_batch_embeddings(self, inputs: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for a large list of inputs using batch processing.
        
        Args:
            inputs: List of texts to embed
            batch_size: Optional batch size (overrides config)
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not self._model_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        if not inputs:
            raise ValueError("Inputs cannot be empty")
        
        # Use provided batch size or default from config
        batch_size = batch_size or config.max_batch_size
        
        # Process inputs in batches
        all_embeddings = []
        
        logger.info(f"Processing {len(inputs)} inputs in batches of {batch_size}")
        
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating embeddings"):
            batch_inputs = inputs[i:i + batch_size]
            batch_embeddings = self.generate_embeddings(batch_inputs)
            all_embeddings.extend(batch_embeddings)
        
        logger.info(f"Generated embeddings for {len(all_embeddings)} inputs")
        
        return all_embeddings
    
    @property
    def model(self) -> Optional[Any]:
        """Get the loaded model."""
        return self._model
    
    @property
    def tokenizer(self) -> Optional[Any]:
        """Get the loaded tokenizer."""
        return self._tokenizer
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        if not self._model_loaded:
            base_info = {
                "device": self.device,
                "loaded": False,
                "max_context_length": config.max_context_length,
                "embedding_dimension": config.embedding_dimension,
            }
            
            if config.load_from_transformers:
                base_info.update({
                    "model_name": config.model_name,
                    "model_path": None,
                    "load_from_transformers": True,
                    "model_source": "transformers",
                })
            else:
                base_info.update({
                    "model_name": config.model_name,
                    "model_path": config.model_path,
                    "load_from_transformers": False,
                    "model_source": config.model_source,
                })
            
            return base_info
        
        # Model is loaded, get detailed info
        base_info = {
            "device": self.device,
            "loaded": True,
            "max_context_length": config.max_context_length,
            "embedding_dimension": config.embedding_dimension,
            "vocab_size": self._tokenizer.vocab_size if self._tokenizer else 0,
            "model_type": self._model.config.model_type if self._model else "",
        }
        
        if config.load_from_transformers:
            base_info.update({
                "model_name": config.model_name,
                "model_path": None,
                "load_from_transformers": True,
                "model_source": "transformers",
                # Note: cache_dir is no longer included as it's managed by the model libraries
            })
        else:
            base_info.update({
                "model_name": config.model_name,
                "model_path": config.model_path,
                "load_from_transformers": False,
                "model_source": config.model_source,
            })
        
        # Add loader-specific information if available
        if self._loader:
            loader_info = self._loader.get_model_info()
            base_info.update(loader_info)
        
        return base_info
    
    def save_embeddings_cache(self, embeddings: List[List[float]], inputs: List[str], cache_path: str) -> None:
        """
        Save embeddings to cache file.
        
        Args:
            embeddings: List of embedding vectors
            inputs: List of input texts
            cache_path: Path to save the cache file
        """
        try:
            cache_data = {
                "inputs": inputs,
                "embeddings": embeddings,
                "model_name": config.model_name,
                "config": {
                    "max_context_length": config.max_context_length,
                    "embedding_dimension": config.embedding_dimension
                }
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Saved embeddings cache to {cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")
            raise RuntimeError(f"Failed to save embeddings cache: {e}")
    
    def load_embeddings_cache(self, cache_path: str) -> Optional[dict]:
        """
        Load embeddings from cache file.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            Optional[dict]: Cached data if valid, None otherwise
        """
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache data
            if (
                cache_data.get("model_name") != config.model_name or
                cache_data.get("config", {}).get("max_context_length") != config.max_context_length or
                cache_data.get("config", {}).get("embedding_dimension") != config.embedding_dimension
            ):
                logger.warning(f"Cache data is incompatible with current model configuration")
                return None
            
            logger.info(f"Loaded embeddings cache from {cache_path}")
            return cast(Dict[str, Any], cache_data)
            
        except Exception as e:
            logger.error(f"Failed to load embeddings cache: {e}")
            return None
    
    def cleanup(self) -> None:
        """
        Clean up resources and unload the model.
        """
        if self._loader:
            self._loader.cleanup()
            self._loader = None
        
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Note: cache manager functionality has been removed as HuggingFace and ModelScope manage their own caches
        
        logger.info("ModelManager cleanup completed")
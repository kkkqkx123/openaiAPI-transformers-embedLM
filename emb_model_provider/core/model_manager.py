from typing import Dict, List, Optional, Any, Union, Tuple, cast
import pickle
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


class ModelManager:
    """
    Model manager that uses loader pattern to handle different model sources.
    Delegates model loading to appropriate loaders while handling embedding generation.
    """
    
    def __init__(self, loader: Optional[BaseModelLoader] = None):
        """
        Initialize the model manager.
        
        Args:
            loader: Optional loader instance. If not provided, will create based on config.
        """
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.device = config.device
        self.max_batch_size = config.max_batch_size
        self.max_context_length = config.max_context_length
        self.embedding_dimension = config.embedding_dimension
        
        # Transformers-specific configuration
        self.load_from_transformers = config.load_from_transformers
        self.transformers_model_name = config.transformers_model_name
        self.transformers_cache_dir = config.transformers_cache_dir
        self.transformers_trust_remote_code = config.transformers_trust_remote_code
        
        # Model source configuration
        self.model_source = config.model_source
        
        # Model and tokenizer references
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._model_loaded = False
        self._loader: Optional[BaseModelLoader] = loader
        
        logger.info(f"ModelManager initialized with model_source={self.model_source}, device={self.device}")
    
    def _create_loader(self) -> BaseModelLoader:
        """
        Create appropriate loader based on configuration.
        
        Returns:
            BaseModelLoader: Configured loader instance
        """
        if self.load_from_transformers:
            logger.info("Creating HuggingFaceModelLoader for transformers loading")
            return HuggingFaceModelLoader(
                model_name=self.transformers_model_name,
                model_path=None,
                device=self.device,
                cache_dir=self.transformers_cache_dir,
                trust_remote_code=self.transformers_trust_remote_code
            )
        
        if self.model_source == "modelscope":
            logger.info("Creating ModelScopeModelLoader for ModelScope")
            return ModelScopeModelLoader(
                model_name=self.model_name,
                model_path=self.model_path,
                device=self.device
            )
        
        # Default to Hugging Face
        logger.info("Creating HuggingFaceModelLoader as default")
        return HuggingFaceModelLoader(
            model_name=self.model_name,
            model_path=self.model_path,
            device=self.device
        )
    
    def load_model(self) -> None:
        """
        Load model using the loader pattern.
        
        Creates a loader if not provided, then uses it to load the model and tokenizer.
        """
        try:
            # Create loader if not provided
            if not self._loader:
                self._loader = self._create_loader()
            
            logger.info(f"Loading model using loader: {type(self._loader).__name__}")
            log_model_event("load_start", self.model_name, {"loader": type(self._loader).__name__})
            
            # Use loader to load model and tokenizer
            self._model, self._tokenizer = self._loader.load_model()
            self._model_loaded = True
            
            logger.info(f"Model loaded successfully using {type(self._loader).__name__}")
            log_model_event("load_complete", self.model_name, {"loader": type(self._loader).__name__})
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            log_model_event("load_error", self.model_name, {"error": str(e)})
            raise RuntimeError(f"Failed to load model: {e}")
    
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
                "cache_dir": self.transformers_cache_dir,
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
        
        logger.info("ModelManager cleanup completed")
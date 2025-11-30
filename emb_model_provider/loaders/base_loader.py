"""
Abstract base class for model loaders.

This module defines the interface that all model loaders must implement,
providing a consistent way to load models from different sources.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import logging
from ..api.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class BaseModelLoader(ABC):
    """
    Abstract base class for loading embedding models from various sources.
    
    This class defines the interface that all model loaders must implement,
    ensuring consistent behavior across different model sources like HuggingFace
    and ModelScope.
    """
    
    def __init__(self, model_name: str, model_path: Optional[str] = None, 
                 cache_dir: Optional[str] = None, trust_remote_code: bool = False, **kwargs: Any) -> None:
        """
        Initialize the model loader.
        
        Args:
            model_name: Name or path of the model
            model_path: Optional local path to the model
            cache_dir: Directory to cache downloaded models
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional loader-specific parameters
        """
        self.model_name = model_name
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.kwargs = kwargs
        
        # These will be set when the model is loaded
        self.model = None
        self.tokenizer = None
        self.device = None
        
    @abstractmethod
    def load_model(self) -> tuple:
        """
        Load the model and tokenizer.
        
        Returns:
            tuple: (model, tokenizer) tuple
            
        Raises:
            ModelLoadError: If the model fails to load
        """
        pass
    
    @abstractmethod
    def is_model_available(self) -> bool:
        """
        Check if the model is available for loading.
        
        Returns:
            bool: True if the model can be loaded, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            dict: Model information including name, source, version, etc.
        """
        pass
    
    def get_device(self, device: Optional[str] = None) -> str:
        """
        Determine the appropriate device for model loading.
        
        Args:
            device: Preferred device ('cpu', 'cuda', 'auto', etc.)
            
        Returns:
            str: The device to use for loading the model
        """
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def validate_model(self) -> bool:
        """
        Validate that the loaded model is functional.
        
        Returns:
            bool: True if the model is valid, False otherwise
        """
        if self.model is None or self.tokenizer is None:
            return False
            
        try:
            # Test with a simple input
            test_input = "Hello, world!"
            tokens = self.tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = self.model(**tokens)
                
            # Check if we get embeddings
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
                if embeddings.shape[0] > 0 and embeddings.shape[-1] > 0:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def get_model_size(self) -> Optional[int]:
        """
        Get the size of the loaded model in parameters.
        
        Returns:
            Optional[int]: Number of parameters in the model, or None if model not loaded
        """
        if self.model is None:
            return None
            
        try:
            if hasattr(self.model, 'parameters'):
                return sum(p.numel() for p in self.model.parameters())  # type: ignore[union-attr]
            return None
        except Exception:
            return None
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the loader.
        
        This method should be called when the loader is no longer needed.
        """
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.device = None
        
    def __enter__(self) -> "BaseModelLoader":
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - clean up resources."""
        self.cleanup()
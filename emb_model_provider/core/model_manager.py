"""
Model manager module for embedding model provider.

This module provides functionality to load and manage embedding models,
including local loading and downloading from Hugging Face Hub.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Union

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from .config import config
from .logging import get_logger, log_model_event

logger = get_logger(__name__)


class ModelManager:
    """
    Manager for loading and managing embedding models.
    
    This class handles loading models from local paths or downloading
    them from Hugging Face Hub, and provides methods for batch inference.
    """
    
    def __init__(self, model_path: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            model_path: Path to the model directory (overrides config)
            model_name: Name of the model (overrides config)
        """
        self.model_path = model_path or config.model_path
        self.model_name = model_name or config.model_name
        self.device = self._get_device()
        
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        
        logger.info(f"Initializing ModelManager for {self.model_name}")
    
    def _get_device(self) -> str:
        """
        Determine the device to use for model inference.
        
        Returns:
            str: Device name ('cuda', 'cpu', or 'mps')
        """
        if config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return config.device
    
    def _is_model_available_locally(self) -> bool:
        """
        Check if the model is available locally.
        
        Returns:
            bool: True if model is available locally, False otherwise
        """
        model_path = Path(self.model_path)
        
        # Check for model files
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        # Check for alternative model file names
        alternative_files = [
            "model.safetensors",
            "tokenizer.json"
        ]
        
        # Check if required files exist
        has_required = all((model_path / file).exists() for file in required_files)
        
        # Check for alternative files
        has_alternative = (
            (model_path / "config.json").exists() and
            ((model_path / "pytorch_model.bin").exists() or (model_path / "model.safetensors").exists()) and
            (model_path / "tokenizer.json").exists()
        )
        
        return has_required or has_alternative
    
    def _load_local_model(self) -> None:
        """
        Load model and tokenizer from local path.
        
        Raises:
            FileNotFoundError: If model files are not found locally
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading model from local path: {self.model_path}")
            log_model_event("load_start", self.model_name, {"source": "local", "path": self.model_path})
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            # Load model
            self._model = AutoModel.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            # Move model to device
            self._model = self._model.to(self.device)
            self._model.eval()
            
            self._model_loaded = True
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            log_model_event("load_complete", self.model_name, {"source": "local", "device": self.device})
            
        except Exception as e:
            logger.error(f"Failed to load model from local path: {e}")
            log_model_event("load_error", self.model_name, {"source": "local", "error": str(e)})
            raise RuntimeError(f"Failed to load model from local path: {e}")
    
    def _download_model(self) -> None:
        """
        Download model from Hugging Face Hub.
        
        Raises:
            RuntimeError: If model download fails
        """
        try:
            logger.info(f"Downloading model {self.model_name} from Hugging Face Hub")
            log_model_event("download_start", self.model_name, {"source": "huggingface_hub"})
            
            # Create model directory if it doesn't exist
            model_dir = Path(self.model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Download model from Hugging Face Hub
            snapshot_download(
                repo_id=self.model_name,
                local_dir=self.model_path,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Model downloaded successfully to {self.model_path}")
            log_model_event("download_complete", self.model_name, {"source": "huggingface_hub", "path": self.model_path})
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            log_model_event("download_error", self.model_name, {"source": "huggingface_hub", "error": str(e)})
            raise RuntimeError(f"Failed to download model: {e}")
    
    def load_model(self) -> None:
        """
        Load the model, either from local path or by downloading from Hugging Face Hub.
        
        This method will first try to load the model from the local path.
        If the model is not available locally, it will download it from Hugging Face Hub.
        """
        if self._model_loaded:
            logger.debug("Model already loaded")
            return
        
        # Try to load from local path first
        if self._is_model_available_locally():
            self._load_local_model()
        else:
            # Download model from Hugging Face Hub
            self._download_model()
            # Load the downloaded model
            self._load_local_model()
    
    def _tokenize_inputs(self, inputs: List[str]) -> dict:
        """
        Tokenize input texts.
        
        Args:
            inputs: List of input texts to tokenize
            
        Returns:
            dict: Tokenized inputs
        """
        # Tokenize inputs
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded")
        
        tokenized = self._tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=config.max_context_length,
            return_tensors="pt"
        )
        
        # Move to device
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        
        return tokenized
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on model output.
        
        Args:
            model_output: Output from the model
            attention_mask: Attention mask from tokenization
            
        Returns:
            torch.Tensor: Pooled embeddings
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Perform mean pooling
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def generate_embeddings(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for input texts.
        
        Args:
            inputs: Single text or list of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If inputs are invalid
        """
        if not self._model_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        # Convert single input to list
        if isinstance(inputs, str):
            inputs = [inputs]
        
        if not inputs:
            raise ValueError("Inputs cannot be empty")
        
        # Check batch size limit
        if len(inputs) > config.max_batch_size:
            raise ValueError(f"Batch size {len(inputs)} exceeds maximum {config.max_batch_size}")
        
        try:
            # Tokenize inputs
            tokenized = self._tokenize_inputs(inputs)
            
            # Generate embeddings
            with torch.no_grad():
                if self._model is None:
                    raise RuntimeError("Model is not loaded")
                model_output = self._model(**tokenized)
                embeddings = self._mean_pooling(model_output, tokenized["attention_mask"])
            
            # Convert to CPU and then to list
            embeddings = embeddings.cpu().numpy()
            embeddings_list = embeddings.tolist()
            
            logger.debug(f"Generated embeddings for {len(inputs)} inputs")
            
            return embeddings_list
            
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
    def model(self):
        """Get the loaded model."""
        return self._model
    
    @property
    def tokenizer(self):
        """Get the loaded tokenizer."""
        return self._tokenizer
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        if not self._model_loaded:
            return {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "device": self.device,
                "loaded": False
            }
        
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "loaded": True,
            "max_context_length": config.max_context_length,
            "embedding_dimension": config.embedding_dimension,
            "vocab_size": self._tokenizer.vocab_size if self._tokenizer else 0,
            "model_type": self._model.config.model_type if self._model else "",
        }
    
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
                "model_name": self.model_name,
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
                cache_data.get("model_name") != self.model_name or
                cache_data.get("config", {}).get("max_context_length") != config.max_context_length or
                cache_data.get("config", {}).get("embedding_dimension") != config.embedding_dimension
            ):
                logger.warning(f"Cache data is incompatible with current model configuration")
                return None
            
            logger.info(f"Loaded embeddings cache from {cache_path}")
            return cache_data
            
        except Exception as e:
            logger.error(f"Failed to load embeddings cache: {e}")
            return None
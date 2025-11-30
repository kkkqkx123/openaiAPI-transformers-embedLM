# ModelScope Implementation Guide

This guide provides step-by-step instructions for implementing ModelScope support in the embedding model provider service.

## Prerequisites

- Python 3.8+
- Existing embedding model provider codebase
- ModelScope library: `pip install modelscope`

## Step 1: Update Configuration

### Add to `core/config.py`

```python
from typing import Optional, Dict
from pydantic import Field

# Add to Config class
model_source: str = Field(default="huggingface", env="EMB_PROVIDER_MODEL_SOURCE")
modelscope_model_id: Optional[str] = Field(default=None, env="EMB_PROVIDER_MODELSCOPE_MODEL_ID")
modelscope_cache_dir: str = Field(default="~/.cache/modelscope/hub/", env="MODELSCOPE_CACHE_DIR")
modelscope_model_mapping: Dict[str, str] = Field(default_factory=lambda: {
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "damo/nlp_gte_sentence-embedding_chinese-base": "damo/nlp_gte_sentence-embedding_chinese-base",
})
```

## Step 2: Create Base Loader Interface

### Create `core/base_model_loader.py`

```python
from abc import ABC, abstractmethod
from typing import List, Any

class BaseModelLoader(ABC):
    """Abstract base class for model loaders."""
    
    @abstractmethod
    def load_model(self, model_name: str) -> Any:
        """Load the model and return model components."""
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for given texts."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Get model information."""
        pass
```

## Step 3: Implement HuggingFace Loader

### Create `core/huggingface_loader.py`

```python
import torch
import logging
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
from core.base_model_loader import BaseModelLoader
from core.config import Config

logger = logging.getLogger(__name__)

class HuggingFaceModelLoader(BaseModelLoader):
    """Hugging Face Transformers model loader."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(self, model_name: str):
        """Load model using Hugging Face transformers."""
        try:
            logger.info(f"Loading Hugging Face model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Move to appropriate device
            device = self.config.device
            if device != "cpu" and torch.cuda.is_available():
                self.model = self.model.to(device)
            
            logger.info(f"Hugging Face model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {e}")
            raise RuntimeError(f"Failed to load Hugging Face model: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Hugging Face model."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        # Tokenize inputs
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.config.context_length
        )
        
        # Move to device
        device = self.config.device
        if device != "cpu" and torch.cuda.is_available():
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        return embeddings.tolist()
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "loader": "huggingface",
            "model_name": self.config.model_name if hasattr(self.config, 'model_name') else "unknown",
            "device": self.config.device
        }
```

## Step 4: Implement ModelScope Loader

### Create `core/modelscope_loader.py`

```python
import logging
from typing import List
from core.base_model_loader import BaseModelLoader
from core.config import Config

logger = logging.getLogger(__name__)

class ModelScopeModelLoader(BaseModelLoader):
    """ModelScope model loader."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embed_pipeline = None
    
    def load_model(self, model_name: str):
        """Load ModelScope model using pipeline interface."""
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            
            # Map model name to ModelScope ID if needed
            modelscope_id = self._map_model_name_to_modelscope(model_name)
            
            logger.info(f"Loading ModelScope model: {modelscope_id}")
            
            # Determine device
            device = "gpu" if self.config.device != "cpu" else "cpu"
            
            self.embed_pipeline = pipeline(
                Tasks.sentence_embedding,
                model=modelscope_id,
                device=device
            )
            
            logger.info(f"ModelScope model loaded successfully: {modelscope_id}")
            
        except ImportError as e:
            logger.error(f"ModelScope not installed: {e}")
            raise RuntimeError("ModelScope not installed. Install with: pip install modelscope")
        except Exception as e:
            logger.error(f"Failed to load ModelScope model: {e}")
            raise RuntimeError(f"Failed to load ModelScope model: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using ModelScope pipeline."""
        if not self.embed_pipeline:
            raise RuntimeError("Model not loaded")
        
        # Handle newlines for consistent input
        texts = [text.replace("\n", " ") for text in texts]
        
        # Generate embeddings
        inputs = {"source_sentence": texts}
        result = self.embed_pipeline(input=inputs)
        
        return result['text_embedding'].tolist()
    
    def _map_model_name_to_modelscope(self, model_name: str) -> str:
        """Map common model names to ModelScope equivalents."""
        # Use config mapping if available
        if hasattr(self.config, 'modelscope_model_mapping') and model_name in self.config.modelscope_model_mapping:
            return self.config.modelscope_model_mapping[model_name]
        
        # Default mappings
        default_mappings = {
            "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
            "thenlper/gte-small": "thenlper/gte-small",
            "thenlper/gte-base": "thenlper/gte-base",
            # Chinese models
            "shibing624/text2vec-base-chinese": "damo/nlp_gte_sentence-embedding_chinese-base",
            "GanymedeNil/text2vec-large-chinese": "damo/nlp_gte_sentence-embedding_chinese-base",
        }
        
        return default_mappings.get(model_name, model_name)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "loader": "modelscope",
            "model_name": self.config.model_name if hasattr(self.config, 'model_name') else "unknown",
            "device": self.config.device
        }
```

## Step 5: Update ModelManager

### Update `core/model_manager.py`

```python
import logging
from typing import List, Optional, Dict, Any
from core.config import Config
from core.base_model_loader import BaseModelLoader
from core.huggingface_loader import HuggingFaceModelLoader
from core.modelscope_loader import ModelScopeModelLoader

logger = logging.getLogger(__name__)

class ModelManager:
    """Enhanced ModelManager with multiple loader support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.loader: Optional[BaseModelLoader] = None
        self.model_name: Optional[str] = None
        self._initialize_loader()
    
    def _initialize_loader(self):
        """Initialize the appropriate model loader based on configuration."""
        source = self.config.model_source.lower()
        
        logger.info(f"Initializing model loader with source: {source}")
        
        if source == "huggingface":
            self.loader = HuggingFaceModelLoader(self.config)
        elif source == "modelscope":
            self.loader = ModelScopeModelLoader(self.config)
        elif source == "auto":
            # Try ModelScope first, fallback to Hugging Face
            self._initialize_auto_loader()
        else:
            raise ValueError(f"Unknown model source: {source}")
    
    def _initialize_auto_loader(self):
        """Initialize loader with auto fallback logic."""
        # Try sources in order of preference
        for loader_class, source_name in [
            (ModelScopeModelLoader, "ModelScope"),
            (HuggingFaceModelLoader, "Hugging Face")
        ]:
            try:
                self.loader = loader_class(self.config)
                logger.info(f"Using {source_name} model loader")
                break
            except (ImportError, RuntimeError) as e:
                logger.warning(f"{source_name} failed: {e}")
                continue
        else:
            raise RuntimeError("No model loader available")
    
    def load_model(self, model_name: str):
        """Load model using the configured loader."""
        self.model_name = model_name
        logger.info(f"Loading model: {model_name}")
        return self.loader.load_model(model_name)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the configured loader."""
        if not self.loader:
            raise RuntimeError("No model loader initialized")
        
        return self.loader.generate_embeddings(texts)
    
    def generate_batch_embeddings(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Generate embeddings in batches."""
        if batch_size is None:
            batch_size = self.config.max_batch_size
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.generate_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def get_model_info(self) -> dict:
        """Get model information."""
        if not self.loader:
            return {"loader": "none", "model_name": "none"}
        
        info = self.loader.get_model_info()
        info["model_name"] = self.model_name or info.get("model_name", "unknown")
        return info
```

## Step 6: Update Requirements

### Update `requirements.txt`

Add ModelScope as an optional dependency:

```
# Core requirements (existing)
torch>=1.9.0
transformers>=4.20.0
pydantic>=1.8.0

# Optional ModelScope support
modelscope>=1.4.0  # Optional, only needed for ModelScope models
```

## Step 7: Environment Configuration

### Example Environment Variables

```bash
# Use ModelScope
export EMB_PROVIDER_MODEL_SOURCE=modelscope
export EMB_PROVIDER_MODEL_NAME=damo/nlp_gte_sentence-embedding_chinese-base

# Use Hugging Face (default)
export EMB_PROVIDER_MODEL_SOURCE=huggingface
export EMB_PROVIDER_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Auto mode with fallback
export EMB_PROVIDER_MODEL_SOURCE=auto
export EMB_PROVIDER_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Optional ModelScope settings
export MODELSCOPE_CACHE_DIR=/custom/cache/path
export EMB_PROVIDER_MODELSCOPE_MODEL_ID=custom-model-id
```

## Step 8: Testing

### Create Test File `tests/test_model_loaders.py`

```python
import pytest
from core.config import Config
from core.huggingface_loader import HuggingFaceModelLoader
from core.modelscope_loader import ModelScopeModelLoader

class TestModelLoaders:
    
    def test_huggingface_loader(self):
        """Test Hugging Face loader."""
        config = Config(model_source="huggingface")
        loader = HuggingFaceModelLoader(config)
        
        # Test model loading
        loader.load_model("sentence-transformers/all-MiniLM-L6-v2")
        assert loader.model is not None
        assert loader.tokenizer is not None
        
        # Test embedding generation
        embeddings = loader.generate_embeddings(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384  # MiniLM-L6-v2 dimension
    
    def test_modelscope_loader(self):
        """Test ModelScope loader."""
        config = Config(model_source="modelscope")
        loader = ModelScopeModelLoader(config)
        
        # Test model loading
        loader.load_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        assert loader.embed_pipeline is not None
        
        # Test embedding generation
        embeddings = loader.generate_embeddings(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0
    
    def test_model_mapping(self):
        """Test model name mapping."""
        config = Config()
        loader = ModelScopeModelLoader(config)
        
        # Test default mappings
        mapped = loader._map_model_name_to_modelscope("sentence-transformers/all-MiniLM-L6-v2")
        assert mapped == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

## Step 9: Validation

### Run Integration Test

```python
from core.config import Config
from core.model_manager import ModelManager

# Test different configurations
configs = [
    Config(model_source="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2"),
    Config(model_source="modelscope", model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    Config(model_source="auto", model_name="sentence-transformers/all-MiniLM-L6-v2"),
]

for config in configs:
    print(f"Testing with {config.model_source}")
    manager = ModelManager(config)
    manager.load_model(config.model_name)
    
    embeddings = manager.generate_embeddings(["Hello world", "Test text"])
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Model info: {manager.get_model_info()}")
    print("-" * 50)
```

## Troubleshooting

### Common Issues

1. **ModelScope Import Error**:
   ```
   ImportError: No module named 'modelscope'
   ```
   Solution: Install ModelScope: `pip install modelscope`

2. **Model Not Found**:
   ```
   RuntimeError: Model 'model_id' not found on ModelScope
   ```
   Solution: Check model ID and use proper ModelScope model names

3. **Device Issues**:
   ```
   RuntimeError: CUDA out of memory
   ```
   Solution: Reduce batch size or use CPU: `export EMB_PROVIDER_DEVICE=cpu`

4. **Fallback Not Working**:
   ```
   RuntimeError: No model loader available
   ```
   Solution: Check that at least one model source is properly installed

## Performance Tips

1. **Pre-download Models**: Use ModelScope's `snapshot_download()` to cache models
2. **Batch Processing**: Use appropriate batch sizes for your hardware
3. **Device Selection**: Use GPU for better performance when available
4. **Memory Management**: Clear cache periodically for long-running services

## Next Steps

1. **Add Monitoring**: Track model loading times and inference performance
2. **Model Registry**: Create centralized model metadata management
3. **Custom Loaders**: Support for additional model sources
4. **Optimization**: Implement model quantization and optimization techniques
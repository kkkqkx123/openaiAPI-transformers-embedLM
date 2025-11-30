# ModelScope Integration Design Document

## Overview

This document outlines the design for adding ModelScope model support to the existing embedding model provider service that currently only supports Hugging Face Transformers models. The integration will be controlled through environment variables to allow users to specify their preferred model loading method.

## Current Architecture Analysis

### Existing Hugging Face Implementation

The current system uses the following components:

1. **ModelManager** (`core/model_manager.py`): Central class handling model loading and inference
2. **Config** (`core/config.py`): Configuration management using Pydantic Settings
3. **Model Loading Flow**:
   - Check local model files
   - Try loading from Hugging Face transformers
   - Download from Hugging Face Hub if needed
   - Generate embeddings using AutoModel and AutoTokenizer

### Key Methods in Current Implementation

- `_load_transformers_model()`: Loads model using Hugging Face AutoModel
- `load_model()`: Main model loading logic with fallback chain
- `generate_embeddings()`: Text to embedding conversion
- `generate_batch_embeddings()`: Batch processing for efficiency

## Proposed Architecture

### Design Goals

1. **Backward Compatibility**: Maintain existing Hugging Face functionality
2. **Environment Variable Control**: Use `EMB_PROVIDER_MODEL_SOURCE` to specify loading method
3. **Unified Interface**: Same API regardless of model source
4. **Error Handling**: Graceful fallback between model sources
5. **Performance**: Maintain similar inference speeds

### Environment Variables

| Variable | Values | Default | Description |
|----------|---------|---------|-------------|
| `EMB_PROVIDER_MODEL_SOURCE` | `huggingface`, `modelscope`, `auto` | `huggingface` | Model loading source |
| `EMB_PROVIDER_MODELSCOPE_MODEL_ID` | ModelScope model ID | - | Specific ModelScope model identifier |
| `MODELSCOPE_CACHE_DIR` | Directory path | `~/.cache/modelscope/hub/` | ModelScope cache location |

### Implementation Strategy

#### 1. Configuration Updates

Update `Config` class to include ModelScope-specific settings:

```python
# In core/config.py
class Config(BaseSettings):
    # Existing Hugging Face settings...
    
    # ModelScope settings
    model_source: str = Field(default="huggingface", env="EMB_PROVIDER_MODEL_SOURCE")
    modelscope_model_id: Optional[str] = Field(default=None, env="EMB_PROVIDER_MODELSCOPE_MODEL_ID")
    modelscope_cache_dir: str = Field(default="~/.cache/modelscope/hub/", env="MODELSCOPE_CACHE_DIR")
    
    # Model mapping for ModelScope equivalents
    modelscope_model_mapping: Dict[str, str] = Field(default_factory=dict)
```

#### 2. ModelManager Refactoring

Create a modular architecture with separate loaders:

```python
# In core/model_manager.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

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

class HuggingFaceModelLoader(BaseModelLoader):
    """Hugging Face Transformers model loader."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(self, model_name: str):
        # Existing Hugging Face loading logic
        pass
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Existing embedding generation logic
        pass

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
            
            self.embed_pipeline = pipeline(
                Tasks.sentence_embedding,
                model=modelscope_id,
                device=self.config.device
            )
            
            logger.info(f"ModelScope model loaded: {modelscope_id}")
            
        except ImportError:
            raise RuntimeError("ModelScope not installed. Install with: pip install modelscope")
        except Exception as e:
            raise RuntimeError(f"Failed to load ModelScope model: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using ModelScope pipeline."""
        if not self.embed_pipeline:
            raise RuntimeError("Model not loaded")
        
        # Handle newlines for consistent input
        texts = [text.replace("\n", " ") for text in texts]
        inputs = {"source_sentence": texts}
        
        result = self.embed_pipeline(input=inputs)
        return result['text_embedding'].tolist()
    
    def _map_model_name_to_modelscope(self, model_name: str) -> str:
        """Map common model names to ModelScope equivalents."""
        # Default mappings
        default_mappings = {
            "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
            # Add more mappings as needed
        }
        
        # Use config mapping if available
        if model_name in self.config.modelscope_model_mapping:
            return self.config.modelscope_model_mapping[model_name]
        
        # Use default mapping
        return default_mappings.get(model_name, model_name)

class ModelManager:
    """Enhanced ModelManager with multiple loader support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.loader: Optional[BaseModelLoader] = None
        self._initialize_loader()
    
    def _initialize_loader(self):
        """Initialize the appropriate model loader based on configuration."""
        source = self.config.model_source.lower()
        
        if source == "huggingface":
            self.loader = HuggingFaceModelLoader(self.config)
        elif source == "modelscope":
            self.loader = ModelScopeModelLoader(self.config)
        elif source == "auto":
            # Try ModelScope first, fallback to Hugging Face
            try:
                self.loader = ModelScopeModelLoader(self.config)
            except RuntimeError:
                logger.warning("ModelScope failed, falling back to Hugging Face")
                self.loader = HuggingFaceModelLoader(self.config)
        else:
            raise ValueError(f"Unknown model source: {source}")
    
    def load_model(self, model_name: str):
        """Load model using the configured loader."""
        return self.loader.load_model(model_name)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the configured loader."""
        return self.loader.generate_embeddings(texts)
```

#### 3. Error Handling and Fallback

Implement robust error handling with fallback mechanisms:

```python
class ModelManager:
    def _initialize_loader(self):
        """Initialize loader with fallback logic."""
        source = self.config.model_source.lower()
        
        if source == "auto":
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
        else:
            # Use specific loader
            loader_map = {
                "huggingface": HuggingFaceModelLoader,
                "modelscope": ModelScopeModelLoader
            }
            
            if source not in loader_map:
                raise ValueError(f"Unknown model source: {source}")
            
            self.loader = loader_map[source](self.config)
```

#### 4. Model Mapping Configuration

Provide a way to map Hugging Face model names to ModelScope equivalents:

```python
# In config.py or separate mapping file
DEFAULT_MODELSCOPE_MAPPING = {
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "thenlper/gte-small": "thenlper/gte-small",
    "thenlper/gte-base": "thenlper/gte-base",
    # Chinese models
    "shibing624/text2vec-base-chinese": "damo/nlp_gte_sentence-embedding_chinese-base",
    "GanymedeNil/text2vec-large-chinese": "damo/nlp_gte_sentence-embedding_chinese-base",
}
```

## Implementation Steps

### Phase 1: Configuration Updates

1. Add ModelScope configuration options to `Config` class
2. Update environment variable parsing
3. Add model mapping configuration

### Phase 2: ModelLoader Implementation

1. Create abstract `BaseModelLoader` class
2. Implement `HuggingFaceModelLoader` (refactor existing code)
3. Implement `ModelScopeModelLoader` (new implementation)
4. Add comprehensive error handling

### Phase 3: ModelManager Refactoring

1. Update `ModelManager` to use loader pattern
2. Implement loader initialization logic
3. Add fallback mechanisms
4. Update existing methods to delegate to loader

### Phase 4: Testing and Validation

1. Unit tests for each loader
2. Integration tests for fallback scenarios
3. Performance benchmarks
4. Error handling validation

## Usage Examples

### Basic Usage with Environment Variables

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
```

### Programmatic Usage

```python
from emb_model_provider.core.config import Config
from emb_model_provider.core.model_manager import ModelManager

# Configure for ModelScope
config = Config(
    model_source="modelscope",
    model_name="damo/nlp_gte_sentence-embedding_chinese-base"
)

# Initialize manager
manager = ModelManager(config)
manager.load_model(config.model_name)

# Generate embeddings
embeddings = manager.generate_embeddings(["Hello world", "Test text"])
```

## Performance Considerations

### ModelScope vs Hugging Face Comparison

| Aspect | Hugging Face | ModelScope | Notes |
|--------|--------------|------------|--------|
| Model Loading | ~10-30s | ~15-45s | ModelScope may be slower for first load |
| Inference Speed | Baseline | Similar | Comparable performance for same model architecture |
| Memory Usage | Baseline | Similar | Same model architecture uses similar memory |
| Cache Management | ~/.cache/huggingface/ | ~/.cache/modelscope/hub/ | Separate cache directories |

### Optimization Strategies

1. **Pre-download Models**: Use `snapshot_download()` to cache models
2. **Batch Processing**: Both loaders support efficient batching
3. **Device Management**: Proper GPU/CPU allocation
4. **Memory Cleanup**: Regular cache clearing for long-running services

## Error Handling

### Common Error Scenarios

1. **ModelScope Not Installed**:
   ```
   RuntimeError: ModelScope not installed. Install with: pip install modelscope
   ```

2. **Model Not Found**:
   ```
   RuntimeError: Model 'model_id' not found on ModelScope
   ```

3. **Network Issues**:
   ```
   RuntimeError: Failed to download model from ModelScope
   ```

4. **Fallback Scenarios**:
   - ModelScope fails → Try Hugging Face (in auto mode)
   - Specific model fails → Try mapped equivalent
   - All sources fail → Raise comprehensive error

## Testing Strategy

### Unit Tests

```python
# Test ModelScope loader
def test_modelscope_loader():
    config = Config(model_source="modelscope")
    loader = ModelScopeModelLoader(config)
    
    # Test model loading
    loader.load_model("damo/nlp_gte_sentence-embedding_chinese-base")
    assert loader.embed_pipeline is not None
    
    # Test embedding generation
    embeddings = loader.generate_embeddings(["test text"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0
```

### Integration Tests

```python
# Test fallback mechanism
def test_auto_fallback():
    config = Config(model_source="auto")
    manager = ModelManager(config)
    
    # Should succeed even if ModelScope fails
    manager.load_model("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = manager.generate_embeddings(["test"])
    assert embeddings is not None
```

## Migration Guide

### For Existing Users

No changes required - default behavior remains Hugging Face:

```bash
# Existing usage continues to work
export EMB_PROVIDER_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
# Model will be loaded from Hugging Face
```

### For ModelScope Users

Add environment variable to switch source:

```bash
# Switch to ModelScope
export EMB_PROVIDER_MODEL_SOURCE=modelscope
export EMB_PROVIDER_MODEL_NAME=damo/nlp_gte_sentence-embedding_chinese-base
```

### For Mixed Usage

Use auto mode for maximum compatibility:

```bash
# Try ModelScope first, fallback to Hugging Face
export EMB_PROVIDER_MODEL_SOURCE=auto
export EMB_PROVIDER_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

## Future Enhancements

1. **Model Registry**: Centralized model metadata and availability
2. **Performance Monitoring**: Track loading times and inference speeds
3. **Custom Model Support**: Allow user-defined model loaders
4. **Multi-Model Support**: Load multiple models simultaneously
5. **Model Versioning**: Support specific model versions

## Conclusion

This design provides a flexible, backward-compatible way to add ModelScope support to the existing embedding model provider. The loader pattern allows for easy extension to additional model sources in the future, while environment variables provide simple configuration control for users.
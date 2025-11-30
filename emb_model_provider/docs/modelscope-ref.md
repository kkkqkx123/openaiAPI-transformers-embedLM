# ModelScope Integration Reference

This document provides reference information for integrating ModelScope models into the embedding model provider service.

## Overview

ModelScope is a comprehensive model hub and library that provides access to a wide range of pre-trained models, including embedding models. This reference covers how to load and use embedding models from ModelScope.

## Installation

```bash
pip install modelscope
```

## Basic Model Loading

### Using ModelScope Pipeline for Embeddings

ModelScope provides a unified pipeline interface for loading and using models:

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Load embedding model using pipeline
embed_pipeline = pipeline(
    Tasks.sentence_embedding, 
    model="damo/nlp_gte_sentence-embedding_chinese-base"
)

# Generate embeddings
texts = ["Hello world", "This is a test"]
inputs = {"source_sentence": texts}
result = embed_pipeline(input=inputs)
embeddings = result['text_embedding']
```

### Common Embedding Models on ModelScope

Here are some popular embedding models available on ModelScope:

| Model ID | Description | Language | Dimension |
|----------|-------------|----------|-----------|
| `damo/nlp_gte_sentence-embedding_chinese-base` | Chinese sentence embeddings | Chinese | 768 |
| `damo/nlp_gte_sentence-embedding_chinese-small` | Lightweight Chinese embeddings | Chinese | 384 |
| `damo/nlp_corom_sentence-embedding_english-base` | English sentence embeddings | English | 768 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multilingual embeddings | Multi | 384 |

### Model Download and Caching

ModelScope automatically handles model downloading and caching:

```python
from modelscope import snapshot_download

# Download model to local cache
model_dir = snapshot_download("damo/nlp_gte_sentence-embedding_chinese-base")
print(f"Model downloaded to: {model_dir}")

# The model will be cached in ~/.cache/modelscope/hub/
```

## Advanced Usage

### Custom Embedding Class for Integration

For integration with embedding services, you can create a custom embedding class:

```python
from typing import List, Any
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class ModelScopeEmbedding:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.embed = pipeline(Tasks.sentence_embedding, model=model_id)
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        # Handle newlines for consistent input
        texts = [text.replace("\n", " ") for text in texts]
        inputs = {"source_sentence": texts}
        result = self.embed(input=inputs)
        return result['text_embedding'].tolist()
    
    def encode_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        text = text.replace("\n", " ")
        inputs = {"source_sentence": [text]}
        result = self.embed(input=inputs)
        return result['text_embedding'][0].tolist()
```

### Batch Processing

ModelScope supports efficient batch processing:

```python
# Process large batches efficiently
large_text_list = ["text1", "text2", ..., "text1000"]
batch_size = 32

all_embeddings = []
for i in range(0, len(large_text_list), batch_size):
    batch = large_text_list[i:i + batch_size]
    inputs = {"source_sentence": batch}
    result = embed_pipeline(input=inputs)
    all_embeddings.extend(result['text_embedding'].tolist())
```

### Device Management

Control device placement for optimal performance:

```python
import torch
from modelscope.pipelines import pipeline

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model to specific device
embed_pipeline = pipeline(
    Tasks.sentence_embedding,
    model="damo/nlp_gte_sentence-embedding_chinese-base",
    device=device
)
```

## Integration with Transformers

ModelScope models can also be loaded using the transformers library:

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load ModelScope model using transformers
model_name = "damo/nlp_gte_sentence-embedding_chinese-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Generate embeddings (requires custom pooling logic)
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).tolist()
```

## Error Handling

Common issues and solutions:

```python
try:
    embed_pipeline = pipeline(Tasks.sentence_embedding, model="model_id")
except ImportError as e:
    print(f"ModelScope not installed: {e}")
    # Fallback to transformers or other implementation
except Exception as e:
    print(f"Failed to load model: {e}")
    # Handle model-specific errors
```

## Performance Optimization

### Memory Management

```python
# Clear GPU memory after use
import torch
import gc

# After generating embeddings
torch.cuda.empty_cache()
gc.collect()
```

### Model Quantization

For memory-constrained environments:

```python
# Load model with quantization (if supported)
embed_pipeline = pipeline(
    Tasks.sentence_embedding,
    model="model_id",
    torch_dtype=torch.float16  # Use half precision
)
```

## Environment Variables

ModelScope respects several environment variables:

- `MODELSCOPE_CACHE_DIR`: Custom cache directory for models
- `MODELSCOPE_DOWNLOAD_TIMEOUT`: Download timeout in seconds
- `MODELSCOPE_MAX_WORKERS`: Maximum number of download workers

Example:
```bash
export MODELSCOPE_CACHE_DIR=/path/to/custom/cache
export MODELSCOPE_DOWNLOAD_TIMEOUT=300
```

## References

- [ModelScope Documentation](https://modelscope.cn/docs)
- [ModelScope GitHub](https://github.com/modelscope/modelscope)
- [Available Models](https://modelscope.cn/models)
# Embedding Model Provider API - Project Context

## Overview

The Embedding Model Provider API is a FastAPI-based service that provides OpenAI-compatible embeddings API functionality. It uses the `all-MiniLM-L12-v2` model to generate high-quality text embeddings and is built with PyTorch and Transformers.

## Project Structure

```
emb_model_provider/
├── __init__.py
├── main.py              # FastAPI application entry point
├── api/                 # API routes
│   ├── __init__.py
│   ├── embeddings.py    # Embeddings endpoint
│   ├── models.py        # Models endpoint
│   ├── exceptions.py    # Exception definitions
│   └── middleware.py    # Middleware
├── core/                # Core business logic
│   ├── __init__.py
│   ├── config.py        # Configuration management
│   ├── logging.py       # Logging configuration
│   └── model_manager.py # Model management
└── services/            # Service layer
    ├── __init__.py
    └── embedding_service.py # Embedding service
```

## Architecture

The application follows a layered architecture:

1. **API Layer** (`api/`): Handles HTTP requests and responses, validates input, and returns proper responses
2. **Service Layer** (`services/`): Contains business logic for processing requests
3. **Core Layer** (`core/`): Manages configuration, model loading, and logging

## Core Components

### Configuration (config.py)
- Uses Pydantic Settings to manage configuration via environment variables
- Key settings include model path, batch size, context length, device selection, and API host/port
- Automatically loads from `.env` file if it exists

### Model Manager (model_manager.py)
- Downloads models from Hugging Face Hub if not available locally
- Supports local model loading
- Handles different device types (CPU, CUDA, MPS)
- Provides batch embedding generation
- Includes caching mechanism for embeddings

### Embedding Service (embedding_service.py)
- Validates requests against configuration limits
- Processes embedding requests with proper error handling
- Performs tokenization, embedding generation, and mean pooling
- Calculates token usage for the API response

### API Endpoints
- `POST /v1/embeddings`: Creates embeddings for text inputs
- `GET /v1/models`: Lists available models
- `GET /health`: Health check endpoint
- `GET /`: Root endpoint with API information

### Logging and Monitoring
- **JSON Logging**: Structured JSON logging with timestamps, levels, and request IDs
- **Request ID Tracking**: Middleware that generates unique request IDs for tracing across requests
- **Model Event Logging**: Specialized logging for model events like loading, downloading, and errors
- **Performance Monitoring**: Timing of request processing and batch operations

### Middleware
- **CORS Middleware**: Configured to allow cross-origin requests
- **Request ID Middleware**: Adds unique request IDs to each request for tracking
- **Exception Handling**: Global exception handling with proper error responses in OpenAI-compatible format

## Setup and Running

### Requirements
- Python 3.10+
- uv package manager (recommended) or pip
- At least 2GB available memory

### Installation
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Running the service
```bash
# Using the built-in run function
uv run python -m emb_model_provider.main

# Using uvicorn directly
uvicorn emb_model_provider.main:app --host localhost --port 9000
```

### Environment Variables
Key environment variables follow the `EMB_PROVIDER_*` prefix:
- `EMB_PROVIDER_MODEL_PATH`: Path to the model directory
- `EMB_PROVIDER_MODEL_NAME`: Name of the model (default: all-MiniLM-L12-v2)
- `EMB_PROVIDER_MAX_BATCH_SIZE`: Maximum batch size (default: 32)
- `EMB_PROVIDER_MAX_CONTEXT_LENGTH`: Max context length (default: 512)
- `EMB_PROVIDER_DEVICE`: Device to run on (auto, cpu, cuda)
- `EMB_PROVIDER_HOST`: Host to bind (default: localhost)
- `EMB_PROVIDER_PORT`: Port to bind (default: 9000)
- `EMB_PROVIDER_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## API Usage

The service is fully compatible with OpenAI's embeddings API format:

### Creating embeddings
```python
import requests

response = requests.post(
    "http://localhost:9000/v1/embeddings",
    json={
        "input": "Hello, world!",
        "model": "all-MiniLM-L12-v2"
    }
)
```

### Using with OpenAI Python client
```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",  # No real API key needed
    base_url="http://localhost:9000/v1"
)

response = client.embeddings.create(
    model="all-MiniLM-L12-v2",
    input="Hello, world!"
)
```

## Docker Deployment

The project includes a Dockerfile for containerized deployment:

```bash
# Build the image
docker build -t emb-model-provider .

# Run the container
docker run -p 9000:9000 -v /path/to/models:/models emb-model-provider
```

## Testing

The project includes comprehensive tests that can be run with:
```bash
uv run pytest
uv run pytest --cov=emb_model_provider  # With coverage
```

The test suite includes:
- Unit tests for individual components
- API endpoint tests
- End-to-end integration tests
- OpenAI compatibility tests
- Performance tests
- Model manager tests

## Key Features

1. **OpenAI Compatibility**: Full compatibility with OpenAI's embeddings API format
2. **High Performance**: Based on FastAPI and PyTorch for efficient inference
3. **Auto Model Management**: Supports local model loading and automatic download from Hugging Face Hub
4. **Flexible Configuration**: Environment variable and file-based configuration
5. **Comprehensive Error Handling**: Follows OpenAI's API error format
6. **Structured Logging**: JSON-formatted structured logging
7. **Batch Processing**: Support for batch embedding requests
8. **Multiple Device Support**: Automatic selection between CPU, CUDA, or MPS
9. **Request Tracking**: Unique request IDs for monitoring and debugging
10. **Performance Monitoring**: Timing and metrics for API operations

## Dependencies

Key dependencies include:
- `torch` (>=2.9.0): PyTorch for model inference
- `transformers`: Hugging Face Transformers library
- `fastapi`: Web framework
- `pydantic`: Data validation
- `uvicorn`: ASGI server
- `huggingface_hub`: For model downloads
- `accelerate`: For optimized model inference

## Development

For development, the project includes tools for:
- Type checking with `mypy`
- Testing with `pytest`
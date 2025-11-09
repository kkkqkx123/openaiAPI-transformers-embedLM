"""
API module for embedding model provider.
"""
from . import embeddings, models
from .middleware import exception_handlers

__all__ = ['embeddings', 'models', 'exception_handlers']
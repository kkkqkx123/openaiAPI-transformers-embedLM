"""
Configuration module for embedding model provider.

This module defines the configuration model and provides functionality
to load configuration from environment variables and .env files.
"""

import os
from typing import Optional
from pathlib import Path

from pydantic import Field, SettingsConfigDict
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    Configuration model for the embedding model provider.
    
    This class defines all configuration parameters with their default values
    and environment variable mappings.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="EMB_PROVIDER_",
        case_sensitive=False,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    # Model configuration
    model_path: str = Field(
        default="D:\\models\\all-MiniLM-L12-v2",
        description="Path to the model directory"
    )
    model_name: str = Field(
        default="all-MiniLM-L12-v2",
        description="Name of the model"
    )
    
    # Processing configuration
    max_batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Maximum batch size for processing"
    )
    max_context_length: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Maximum context length in tokens"
    )
    embedding_dimension: int = Field(
        default=384,
        ge=1,
        description="Dimension of the embedding vectors"
    )
    
    # Resource configuration
    memory_limit: str = Field(
        default="2GB",
        description="Memory limit for the service"
    )
    device: str = Field(
        default="auto",
        description="Device to run the model on (auto, cpu, cuda)"
    )
    
    # API configuration
    host: str = Field(
        default="localhost",
        description="Host to bind the API server"
    )
    port: int = Field(
        default=9000,
        ge=1,
        le=65535,
        description="Port to bind the API server"
    )
    
    # Logging configuration
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR)$",
        description="Logging level"
    )
        
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """
        Load configuration from environment variables and optionally from a .env file.
        
        Args:
            env_file: Optional path to a .env file to load configuration from
            
        Returns:
            Config: Configuration instance with values from environment variables
        """
        if env_file:
            return cls(_env_file=env_file)
        return cls()
    
    @classmethod
    def load_from_file(cls, env_file: str) -> "Config":
        """
        Load configuration from a specific .env file.
        
        Args:
            env_file: Path to the .env file
            
        Returns:
            Config: Configuration instance with values from the specified file
        """
        env_path = Path(env_file)
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        
        return cls(_env_file=str(env_path.absolute()))
    
    def get_model_config(self) -> dict:
        """
        Get model-related configuration.
        
        Returns:
            dict: Model configuration parameters
        """
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "max_context_length": self.max_context_length,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
        }
    
    def get_api_config(self) -> dict:
        """
        Get API-related configuration.
        
        Returns:
            dict: API configuration parameters
        """
        return {
            "host": self.host,
            "port": self.port,
        }
    
    def get_processing_config(self) -> dict:
        """
        Get processing-related configuration.
        
        Returns:
            dict: Processing configuration parameters
        """
        return {
            "max_batch_size": self.max_batch_size,
            "max_context_length": self.max_context_length,
            "memory_limit": self.memory_limit,
        }
    
    def get_logging_config(self) -> dict:
        """
        Get logging-related configuration.
        
        Returns:
            dict: Logging configuration parameters
        """
        return {
            "log_level": self.log_level,
        }


# Global configuration instance
config = Config.from_env()
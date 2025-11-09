"""
Configuration module for embedding model provider.

This module defines the configuration model and provides functionality
to load configuration from environment variables and .env files.
"""

import os
from typing import Optional
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import torch


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
        le=512,  # 提高最大限制以支持高端GPU
        description="Maximum batch size for processing"
    )
    
    # Dynamic batch processing configuration
    enable_dynamic_batching: bool = Field(
        default=True,
        description="Enable dynamic batch processing for better throughput"
    )
    
    max_wait_time_ms: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum wait time in milliseconds for dynamic batching"
    )
    
    min_batch_size: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Minimum batch size for dynamic batching"
    )
    
    # Memory optimization configuration
    enable_length_grouping: bool = Field(
        default=True,
        description="Enable length-based grouping to reduce padding overhead"
    )
    
    length_group_tolerance: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Tolerance for length grouping (20% means groups can have 20% length difference)"
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


    def get_optimal_batch_size(self) -> int:
        """
        根据GPU内存自动计算最优批处理大小
        
        Returns:
            int: 最优批处理大小
        """
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            
            # 根据GPU内存大小估算最优批处理大小
            # 基础估算：每个样本大约需要 100-200MB 显存（包括模型和中间结果）
            if gpu_memory_gb >= 24:  # RTX 4090, A100
                optimal_size = 256
            elif gpu_memory_gb >= 16:  # RTX 3090, V100
                optimal_size = 128
            elif gpu_memory_gb >= 12:  # RTX 3060, 3080
                optimal_size = 64
            elif gpu_memory_gb >= 8:   # RTX 2070, 2080
                optimal_size = 32
            else:  # 低端GPU或集成GPU
                optimal_size = 16
            
            # 确保不超过配置的最大值
            return min(optimal_size, self.max_batch_size)
        else:
            # CPU模式下使用较小的批处理大小
            return min(16, self.max_batch_size)
    
    def optimize_for_hardware(self) -> None:
        """
        根据硬件特性优化配置
        """
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            
            # 根据GPU架构调整配置
            if gpu_props.major >= 8:  # Ampere架构及以上
                # 启用更激进的批处理设置
                if self.max_batch_size < 64:
                    self.max_batch_size = 64
            
            # 根据显存大小调整动态批处理参数
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            if gpu_memory_gb >= 16:
                self.max_wait_time_ms = 150  # 高端GPU可以等待更长时间以获得更大批次
                self.min_batch_size = 4
            else:
                self.max_wait_time_ms = 50   # 低端GPU快速处理小批次
                self.min_batch_size = 1


# Global configuration instance
config = Config.from_env()

# 自动优化配置
config.optimize_for_hardware()
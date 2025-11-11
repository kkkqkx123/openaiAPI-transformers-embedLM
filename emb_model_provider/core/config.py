"""
Configuration module for embedding model provider.

This module defines the configuration model and provides functionality
to load configuration from environment variables and .env files.
"""

import os
from typing import Optional, List
from pathlib import Path
import json
import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import torch

logger = logging.getLogger(__name__)


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
        default="D:\\models\\multilingual-MiniLM-L12-v2",
        description="Path to the model directory"
    )
    model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Name of the model"
    )

    # Model aliases configuration
    model_aliases: str = Field(
        default="",
        description="Comma-separated list of model aliases in format alias1:actual_model_name1,alias2:actual_model_name2"
    )
    
    # Multi-model mapping configuration (JSON format)
    model_mapping: str = Field(
        default="{}",
        description="JSON string mapping model aliases to actual model names and paths"
    )

    # Transformers model loading configuration
    load_from_transformers: bool = Field(
        default=False,
        description="Whether to load model directly from transformers without local caching"
    )
    transformers_model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Model name to load directly from transformers (used when load_from_transformers=True)"
    )
    transformers_cache_dir: Optional[str] = Field(
        default=None,
        description="Custom cache directory for transformers models (None uses default cache)"
    )
    transformers_trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code when loading from transformers"
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

    hard_timeout_additional_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Additional timeout in seconds after max_wait_time to force processing of small batches"
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

    # File logging configuration
    log_to_file: bool = Field(
        default=True,
        description="Enable logging to files"
    )
    log_dir: str = Field(
        default="logs",
        description="Directory to store log files"
    )
    log_file_max_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum size of a single log file in MB"
    )
    log_retention_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Default number of days to retain log files"
    )
    log_cleanup_interval_hours: int = Field(
        default=1,
        ge=1,
        le=24,
        description="Interval in hours for log cleanup checks"
    )
    log_max_dir_size_mb: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum total size of log directory in MB before cleanup"
    )
    log_cleanup_target_size_mb: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Target size in MB after cleanup (should be less than log_max_dir_size_mb)"
    )
    log_cleanup_retention_days: str = Field(
        default="7,3,1",
        description="Retention days to try during cleanup (comma-separated)"
    )

    def get_log_cleanup_retention_days(self) -> List[int]:
        """
        Parse the log cleanup retention days from string to list.

        Returns:
            List[int]: List of retention days
        """
        try:
            return [int(x.strip()) for x in self.log_cleanup_retention_days.split(',')]
        except (ValueError, AttributeError):
            return [7, 3, 1]  # Default fallback

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """
        Load configuration from environment variables and optionally from a .env file.

        Args:
            env_file: Optional path to a .env file to load configuration from

        Returns:
            Config: Configuration instance with values from environment variables
        """
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
            
        # 从指定的.env文件加载，不修改全局环境变量
        from dotenv import dotenv_values
        env_vars = dotenv_values(str(env_path.absolute()))
        
        # 过滤掉None值并转换类型
        filtered_env_vars = {}
        for key, value in env_vars.items():
            if value is not None:
                # 移除前缀
                if key.startswith("EMB_PROVIDER_"):
                    config_key = key[len("EMB_PROVIDER_"):].lower()
                    
                    # 类型转换
                    if config_key in ["max_batch_size", "max_context_length", "embedding_dimension",
                                     "port", "log_file_max_size", "log_retention_days",
                                     "log_cleanup_interval_hours", "log_max_dir_size_mb",
                                     "log_cleanup_target_size_mb", "min_batch_size",
                                     "max_wait_time_ms"]:
                        try:
                            filtered_env_vars[config_key] = int(value)
                        except ValueError:
                            filtered_env_vars[config_key] = value
                    elif config_key in ["load_from_transformers", "enable_dynamic_batching",
                                       "enable_length_grouping", "log_to_file",
                                       "transformers_trust_remote_code"]:
                        filtered_env_vars[config_key] = value.lower() in ['true', '1', 'yes', 'on']
                    elif config_key in ["length_group_tolerance", "hard_timeout_additional_seconds"]:
                        try:
                            filtered_env_vars[config_key] = float(value)
                        except ValueError:
                            filtered_env_vars[config_key] = value
                    else:
                        filtered_env_vars[config_key] = value
        
        # 创建一个新的Config实例，传入环境变量
        return cls(**filtered_env_vars)

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
            "load_from_transformers": self.load_from_transformers,
            "transformers_model_name": self.transformers_model_name,
            "transformers_cache_dir": self.transformers_cache_dir,
            "transformers_trust_remote_code": self.transformers_trust_remote_code,
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
            "log_to_file": self.log_to_file,
            "log_dir": self.log_dir,
            "log_file_max_size": self.log_file_max_size,
            "log_retention_days": self.log_retention_days,
            "log_cleanup_interval_hours": self.log_cleanup_interval_hours,
            "log_max_dir_size_mb": self.log_max_dir_size_mb,
            "log_cleanup_target_size_mb": self.log_cleanup_target_size_mb,
            "log_cleanup_retention_days": self.get_log_cleanup_retention_days(),
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

    def get_model_aliases(self) -> dict:
        """
        Parse the model aliases string into a dictionary mapping.

        Returns:
            dict: Dictionary mapping aliases to actual model names
        """
        if not self.model_aliases:
            return {}

        aliases = {}
        try:
            for alias_mapping in self.model_aliases.split(','):
                if ':' in alias_mapping:
                    alias, actual_name = alias_mapping.strip().split(':', 1)
                    aliases[alias.strip()] = actual_name.strip()
        except Exception:
            # If there's an error parsing the aliases, return empty dict
            return {}

        return aliases
    
    def get_model_mapping(self) -> dict:
        """
        Parse the model mapping JSON string into a dictionary.
        
        Returns:
            dict: Dictionary mapping aliases to actual model names and paths
        """
        if not self.model_mapping or self.model_mapping == "{}":
            return {}
            
        try:
            return json.loads(self.model_mapping)
        except json.JSONDecodeError:
            logger.warning("Failed to parse model mapping JSON")
            return {}
    
    def get_model_info(self, alias: str) -> dict:
        """
        Get model information (name and path) for a given alias.
        
        Args:
            alias: Model alias
            
        Returns:
            dict: Dictionary containing model name and path, or empty dict if not found
        """
        model_mapping = self.get_model_mapping()
        if alias in model_mapping:
            model_info = model_mapping[alias]
            # If model_info is a string, it's just the model name
            if isinstance(model_info, str):
                return {
                    "name": model_info,
                    "path": self.model_path  # Use default path
                }
            # If model_info is a dict, it contains both name and path
            elif isinstance(model_info, dict):
                return {
                    "name": model_info.get("name", alias),
                    "path": model_info.get("path", self.model_path)
                }
        # If alias not found, check if it's the default model
        elif alias == self.model_name:
            return {
                "name": self.model_name,
                "path": self.model_path
            }
        return {}


# Global configuration instance
config = Config.from_env()

# 自动优化配置
config.optimize_for_hardware()
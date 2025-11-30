"""
Thread-safe tokenizer manager for embedding model provider.

This module provides functionality to manage tokenizer instances in a thread-safe manner,
solving the "Already borrowed" error that occurs with fast tokenizers in concurrent scenarios.
"""

import os
import threading
import copy
from typing import Optional, Dict, Any, cast, Generator
from contextlib import contextmanager
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from .logging import get_logger

logger = get_logger(__name__)

# 设置环境变量以禁用tokenizer并行性警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ThreadSafeTokenizerManager:
    """
    线程安全的tokenizer管理器
    
    解决"Already borrowed"错误的几种策略：
    1. 线程本地存储：每个线程维护独立的tokenizer实例
    2. 深拷贝：在需要时创建tokenizer的深拷贝
    3. 池化管理：维护tokenizer池，按需分配
    """
    
    def __init__(self, model_path: str, use_thread_local: bool = True, pool_size: int = 4):
        """
        初始化tokenizer管理器
        
        Args:
            model_path: 模型路径
            use_thread_local: 是否使用线程本地存储
            pool_size: tokenizer池大小
        """
        self.model_path = model_path
        self.use_thread_local = use_thread_local
        self.pool_size = pool_size
        
        # 主tokenizer实例（用于创建副本）
        self._master_tokenizer: Optional[PreTrainedTokenizer] = None
        
        # 线程本地存储
        self._thread_local = threading.local()
        
        # tokenizer池
        self._tokenizer_pool: Dict[int, PreTrainedTokenizer] = {}
        self._pool_lock = threading.Lock()
        self._available_thread_ids: set[int] = set()
        
        # 加载主tokenizer
        self._load_master_tokenizer()
        
        logger.info(f"ThreadSafeTokenizerManager initialized with strategy: {'thread_local' if use_thread_local else 'pool'}")
    
    def _load_master_tokenizer(self) -> None:
        """加载主tokenizer实例"""
        try:
            logger.info(f"Loading master tokenizer from {self.model_path}")
            self._master_tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                use_fast=True  # 使用fast tokenizer但通过管理器保证线程安全
            )
            logger.info("Master tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load master tokenizer: {e}")
            raise
    
    def _create_tokenizer_copy(self) -> PreTrainedTokenizer:
        """
        创建tokenizer的深拷贝
        
        Returns:
            PreTrainedTokenizer: tokenizer副本
        """
        if self._master_tokenizer is None:
            raise RuntimeError("Master tokenizer not loaded")
        
        try:
            # 创建tokenizer的深拷贝
            tokenizer_copy: PreTrainedTokenizer = copy.deepcopy(self._master_tokenizer)
            return tokenizer_copy
        except Exception as e:
            logger.error(f"Failed to create tokenizer copy: {e}")
            # 如果深拷贝失败，尝试重新加载
            tokenizer: PreTrainedTokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                use_fast=True
            ))
            return tokenizer
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """
        获取线程安全的tokenizer实例
        
        Returns:
            PreTrainedTokenizer: tokenizer实例
        """
        if self.use_thread_local:
            return self._get_thread_local_tokenizer()
        else:
            return self._get_pooled_tokenizer()
    
    def _get_thread_local_tokenizer(self) -> PreTrainedTokenizer:
        """获取线程本地的tokenizer实例"""
        if not hasattr(self._thread_local, 'tokenizer'):
            self._thread_local.tokenizer = self._create_tokenizer_copy()
            logger.debug(f"Created thread-local tokenizer for thread {threading.get_ident()}")
        
        return cast(PreTrainedTokenizer, self._thread_local.tokenizer)
    
    def _get_pooled_tokenizer(self) -> PreTrainedTokenizer:
        """从池中获取tokenizer实例"""
        thread_id: int = threading.get_ident()
        
        with self._pool_lock:
            # 检查当前线程是否已经有分配的tokenizer
            if thread_id in self._tokenizer_pool:
                return self._tokenizer_pool[thread_id]
            
            # 尝试从可用池中获取
            if self._available_thread_ids:
                pool_id: int = self._available_thread_ids.pop()
                tokenizer: PreTrainedTokenizer = self._tokenizer_pool[pool_id]
                del self._tokenizer_pool[pool_id]
                self._tokenizer_pool[thread_id] = tokenizer
                logger.debug(f"Assigned pooled tokenizer to thread {thread_id}")
                return tokenizer
            
            # 如果池已满，创建新的tokenizer
            if len(self._tokenizer_pool) >= self.pool_size:
                logger.warning(f"Tokenizer pool exhausted, creating new tokenizer for thread {thread_id}")
            
            # 创建新的tokenizer
            new_tokenizer: PreTrainedTokenizer = self._create_tokenizer_copy()
            self._tokenizer_pool[thread_id] = new_tokenizer
            logger.debug(f"Created new tokenizer for thread {thread_id}")
            
            return new_tokenizer
    
    def release_tokenizer(self) -> None:
        """释放当前线程的tokenizer回池中"""
        if not self.use_thread_local:
            thread_id: int = threading.get_ident()
            
            with self._pool_lock:
                if thread_id in self._tokenizer_pool:
                    # 将tokenizer移回可用池
                    self._available_thread_ids.add(thread_id)
                    logger.debug(f"Released tokenizer for thread {thread_id}")
    
    @contextmanager
    def get_tokenizer_context(self) -> Generator[PreTrainedTokenizer, None, None]:
        """
        获取tokenizer的上下文管理器
        
        使用方法：
        with tokenizer_manager.get_tokenizer_context() as tokenizer:
            # 使用tokenizer
            result = tokenizer.encode(text)
        """
        tokenizer = self.get_tokenizer()
        try:
            yield tokenizer
        finally:
            if not self.use_thread_local:
                self.release_tokenizer()
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """
        获取tokenizer管理器信息
        
        Returns:
            Dict[str, Any]: 管理器状态信息
        """
        info = {
            "model_path": self.model_path,
            "strategy": "thread_local" if self.use_thread_local else "pool",
            "master_tokenizer_loaded": self._master_tokenizer is not None,
        }
        
        if self.use_thread_local:
            # 统计线程本地tokenizer数量（困难，只能估算）
            info["active_threads"] = cast(Any, threading.active_count())
        else:
            with self._pool_lock:
                info["pool_size"] = cast(Any, len(self._tokenizer_pool))
                info["available_tokenizers"] = cast(Any, len(self._available_thread_ids))
        
        return info
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.use_thread_local:
            # 清理线程本地存储
            if hasattr(self._thread_local, 'tokenizer'):
                delattr(self._thread_local, 'tokenizer')
        else:
            # 清理tokenizer池
            with self._pool_lock:
                self._tokenizer_pool.clear()
                self._available_thread_ids.clear()
        
        # 清理主tokenizer
        self._master_tokenizer = None
        
        logger.info("Tokenizer manager cleaned up")


class GlobalTokenizerManager:
    """全局tokenizer管理器单例"""
    
    _instance: Optional[ThreadSafeTokenizerManager] = None
    _lock = threading.Lock()
    
    @classmethod
    def initialize(cls, model_path: str, **kwargs: Any) -> ThreadSafeTokenizerManager:
        """
        初始化全局tokenizer管理器
        
        Args:
            model_path: 模型路径
            **kwargs: 传递给ThreadSafeTokenizerManager的参数
            
        Returns:
            ThreadSafeTokenizerManager: tokenizer管理器实例
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = ThreadSafeTokenizerManager(model_path, **kwargs)
                logger.info("Global tokenizer manager initialized")
            return cls._instance
    
    @classmethod
    def get_instance(cls) -> ThreadSafeTokenizerManager:
        """
        获取全局tokenizer管理器实例
        
        Returns:
            ThreadSafeTokenizerManager: tokenizer管理器实例
            
        Raises:
            RuntimeError: 如果管理器未初始化
        """
        if cls._instance is None:
            raise RuntimeError("Global tokenizer manager not initialized. Call initialize() first.")
        return cls._instance
    
    @classmethod
    def cleanup(cls) -> None:
        """清理全局tokenizer管理器"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.cleanup()
                cls._instance = None
                logger.info("Global tokenizer manager cleaned up")


# 便捷函数
def get_tokenizer_manager() -> ThreadSafeTokenizerManager:
    """获取全局tokenizer管理器"""
    return GlobalTokenizerManager.get_instance()


def initialize_tokenizer_manager(model_path: str, **kwargs: Any) -> ThreadSafeTokenizerManager:
    """初始化全局tokenizer管理器"""
    return GlobalTokenizerManager.initialize(model_path, **kwargs)
"""
Batch optimization module for embedding model provider.

This module provides intelligent batch processing optimization including
length-based grouping and dynamic batching strategies.
"""

import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from ..core.config import Config
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BatchGroup:
    """批处理组数据类"""
    texts: List[str]
    indices: List[int]  # 原始索引，用于保持顺序
    avg_length: float
    max_length: int
    size: int
    
    def __post_init__(self):
        self.size = len(self.texts)


class LengthBasedBatchOptimizer:
    """基于长度的批处理优化器"""
    
    def __init__(self, config: Config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.length_tolerance = config.length_group_tolerance
        
    def optimize_batch(self, inputs: List[str]) -> List[BatchGroup]:
        """
        优化批处理：按长度分组以减少padding开销
        
        Args:
            inputs: 输入文本列表
            
        Returns:
            List[BatchGroup]: 优化后的批处理组列表
        """
        if not self.config.enable_length_grouping or len(inputs) <= 1:
            # 如果禁用长度分组或只有一个输入，直接返回单个组
            return [BatchGroup(
                texts=inputs,
                indices=list(range(len(inputs))),
                avg_length=self._get_avg_length(inputs),
                max_length=self._get_max_length(inputs),
                size=len(inputs)
            )]
        
        # 计算每个输入的token长度
        text_lengths = []
        for i, text in enumerate(inputs):
            length = len(self.tokenizer.encode(text, add_special_tokens=False))
            text_lengths.append((i, text, length))
        
        # 按长度排序
        text_lengths.sort(key=lambda x: x[2])
        
        # 分组：每组长度差异不超过tolerance
        groups = self._group_by_length(text_lengths)
        
        logger.debug(f"Optimized batch: {len(inputs)} inputs -> {len(groups)} groups")
        
        return groups
    
    def _group_by_length(self, text_lengths: List[Tuple[int, str, int]]) -> List[BatchGroup]:
        """按长度分组文本"""
        groups = []
        current_group = []
        current_length = None
        
        for idx, text, length in text_lengths:
            if current_length is None:
                # 第一个元素
                current_length = length
                current_group = [(idx, text, length)]
            elif length <= current_length * (1 + self.length_tolerance):
                # 长度在容忍范围内，加入当前组
                current_group.append((idx, text, length))
                # 更新当前长度为组内最大长度
                current_length = max(current_length, length)
            else:
                # 长度超出容忍范围，创建新组
                if current_group:
                    groups.append(self._create_batch_group(current_group))
                current_group = [(idx, text, length)]
                current_length = length
        
        # 处理最后一组
        if current_group:
            groups.append(self._create_batch_group(current_group))
        
        return groups
    
    def _create_batch_group(self, group_data: List[Tuple[int, str, int]]) -> BatchGroup:
        """创建批处理组"""
        indices = [item[0] for item in group_data]
        texts = [item[1] for item in group_data]
        lengths = [item[2] for item in group_data]
        
        return BatchGroup(
            texts=texts,
            indices=indices,
            avg_length=np.mean(lengths),
            max_length=max(lengths),
            size=len(texts)
        )
    
    def _get_avg_length(self, texts: List[str]) -> float:
        """获取平均长度"""
        if not texts:
            return 0.0
        lengths = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        return np.mean(lengths)
    
    def _get_max_length(self, texts: List[str]) -> int:
        """获取最大长度"""
        if not texts:
            return 0
        lengths = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        return max(lengths)
    
    def calculate_padding_efficiency(self, groups: List[BatchGroup]) -> Dict:
        """计算padding效率"""
        total_tokens = 0
        effective_tokens = 0
        
        for group in groups:
            # 实际有效的token数量
            effective_tokens += sum(
                len(self.tokenizer.encode(text, add_special_tokens=False)) 
                for text in group.texts
            )
            # 包含padding的总token数量
            total_tokens += group.max_length * group.size
        
        padding_ratio = 1.0 - (effective_tokens / total_tokens) if total_tokens > 0 else 0.0
        
        return {
            'total_groups': len(groups),
            'total_tokens': total_tokens,
            'effective_tokens': effective_tokens,
            'padding_ratio': padding_ratio,
            'efficiency_score': 1.0 - padding_ratio
        }


class DynamicBatchOptimizer:
    """动态批处理优化器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.max_wait_time = config.max_wait_time_ms / 1000.0  # 转换为秒
        self.min_batch_size = config.min_batch_size
        self.max_batch_size = config.max_batch_size
        
    def should_wait_for_more_requests(self, current_batch_size: int, wait_time: float) -> bool:
        """
        判断是否应该等待更多请求
        
        Args:
            current_batch_size: 当前批处理大小
            wait_time: 已经等待的时间
            
        Returns:
            bool: 是否应该继续等待
        """
        # 如果已经达到最大批处理大小，不等待
        if current_batch_size >= self.max_batch_size:
            return False
        
        # 如果已经达到最小批处理大小且等待时间超过阈值，不等待
        if current_batch_size >= self.min_batch_size and wait_time >= self.max_wait_time:
            return False
        
        # 如果等待时间超过最大等待时间，不等待
        if wait_time >= self.max_wait_time * 2:
            return False
        
        return True
    
    def calculate_optimal_batch_size(self, available_requests: int, avg_request_rate: float) -> int:
        """
        计算最优批处理大小
        
        Args:
            available_requests: 可用的请求数量
            avg_request_rate: 平均请求速率（请求/秒）
            
        Returns:
            int: 最优批处理大小
        """
        if avg_request_rate == 0:
            return min(available_requests, self.min_batch_size)
        
        # 基于请求速率动态调整
        # 高请求速率时使用更大的批处理大小
        if avg_request_rate > 10:  # 每秒超过10个请求
            optimal_size = min(self.max_batch_size, available_requests)
        elif avg_request_rate > 5:  # 每秒5-10个请求
            optimal_size = min(self.max_batch_size // 2, available_requests)
        else:  # 低请求速率
            optimal_size = min(self.min_batch_size * 2, available_requests)
        
        return max(self.min_batch_size, optimal_size)


class BatchProcessingOptimizer:
    """批处理优化器主类"""
    
    def __init__(self, config: Config, tokenizer):
        self.config = config
        self.length_optimizer = LengthBasedBatchOptimizer(config, tokenizer)
        self.dynamic_optimizer = DynamicBatchOptimizer(config)
        
    def optimize_batch_processing(self, inputs: List[str]) -> Tuple[List[BatchGroup], Dict]:
        """
        优化批处理流程
        
        Args:
            inputs: 输入文本列表
            
        Returns:
            Tuple[List[BatchGroup], Dict]: 优化后的批处理组和效率信息
        """
        # 1. 长度分组优化
        groups = self.length_optimizer.optimize_batch(inputs)
        
        # 2. 计算效率指标
        efficiency_info = self.length_optimizer.calculate_padding_efficiency(groups)
        
        # 3. 记录优化结果
        logger.info(
            f"Batch optimization completed: {len(inputs)} inputs -> {len(groups)} groups, "
            f"padding efficiency: {efficiency_info['efficiency_score']:.2%}"
        )
        
        return groups, efficiency_info
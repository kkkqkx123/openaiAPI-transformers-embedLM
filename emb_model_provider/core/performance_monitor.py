"""
Performance monitoring module for embedding model provider.

This module provides functionality to monitor and analyze the performance
of batch processing operations.
"""

import time
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
import torch

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    request_count: int = 0
    total_processing_time: float = 0.0
    batch_sizes: List[int] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    gpu_memory_usage: List[float] = field(default_factory=list)
    input_lengths: List[int] = field(default_factory=list)
    padding_ratios: List[float] = field(default_factory=list)
    
    def get_avg_latency(self) -> float:
        """获取平均延迟"""
        if self.request_count == 0:
            return 0.0
        return self.total_processing_time / self.request_count
    
    def get_throughput(self) -> float:
        """获取吞吐量（请求/秒）"""
        if self.total_processing_time == 0:
            return 0.0
        return self.request_count / self.total_processing_time
    
    def get_avg_batch_size(self) -> float:
        """获取平均批处理大小"""
        if not self.batch_sizes:
            return 0.0
        return sum(self.batch_sizes) / len(self.batch_sizes)
    
    def get_p95_latency(self) -> float:
        """获取95分位延迟"""
        if not self.processing_times:
            return 0.0
        return float(np.percentile(self.processing_times, 95))
    
    def get_p99_latency(self) -> float:
        """获取99分位延迟"""
        if not self.processing_times:
            return 0.0
        return float(np.percentile(self.processing_times, 99))
    
    def get_avg_padding_ratio(self) -> float:
        """获取平均填充比例"""
        if not self.padding_ratios:
            return 0.0
        return sum(self.padding_ratios) / len(self.padding_ratios)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self) -> None:
        self.metrics = PerformanceMetrics()
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        
    @contextmanager
    def monitor_request(self, batch_size: int, input_texts: List[str]) -> Any:
        """
        监控单个请求的性能
        
        Args:
            batch_size: 批处理大小
            input_texts: 输入文本列表
        """
        start_time = time.time()
        start_memory = self._get_gpu_memory_usage()
        
        # 计算输入长度和填充比例
        input_lengths = [len(text.split()) for text in input_texts]
        max_length = max(input_lengths) if input_lengths else 0
        padding_ratios = [(max_length - length) / max_length if max_length > 0 else 0 
                         for length in input_lengths]
        avg_padding_ratio = sum(padding_ratios) / len(padding_ratios) if padding_ratios else 0
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_gpu_memory_usage()
            
            processing_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # 线程安全地更新指标
            with self._lock:
                self.metrics.request_count += batch_size
                self.metrics.total_processing_time += processing_time
                self.metrics.batch_sizes.append(batch_size)
                self.metrics.processing_times.append(processing_time)
                self.metrics.gpu_memory_usage.append(memory_delta)
                self.metrics.input_lengths.extend(input_lengths)
                self.metrics.padding_ratios.append(avg_padding_ratio)
            
            # 记录性能日志
            logger.debug(
                f"Batch processing completed: size={batch_size}, "
                f"time={processing_time:.3f}s, memory_delta={memory_delta:.1f}MB, "
                f"padding_ratio={avg_padding_ratio:.2f}"
            )
    
    def _get_gpu_memory_usage(self) -> float:
        """获取当前GPU内存使用量（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        return 0.0
    
    def get_performance_report(self) -> Dict:
        """生成性能报告"""
        with self._lock:
            return {
                'request_count': self.metrics.request_count,
                'avg_latency': self.metrics.get_avg_latency(),
                'throughput': self.metrics.get_throughput(),
                'avg_batch_size': self.metrics.get_avg_batch_size(),
                'p95_latency': self.metrics.get_p95_latency(),
                'p99_latency': self.metrics.get_p99_latency(),
                'avg_memory_per_request': (
                    sum(self.metrics.gpu_memory_usage) / len(self.metrics.gpu_memory_usage)
                    if self.metrics.gpu_memory_usage else 0
                ),
                'avg_padding_ratio': self.metrics.get_avg_padding_ratio(),
                'total_requests_processed': len(self.metrics.batch_sizes)
            }
    
    def reset_metrics(self) -> None:
        """重置性能指标"""
        with self._lock:
            self.metrics = PerformanceMetrics()
        logger.info("Performance metrics reset")
    
    def start_monitoring(self) -> None:
        """开始监控"""
        self._start_time = time.time()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """停止监控并返回最终报告"""
        if self._start_time:
            total_time = time.time() - self._start_time
            report = self.get_performance_report()
            report['total_monitoring_time'] = total_time
            logger.info(f"Performance monitoring stopped after {total_time:.2f}s")
            return report
        return self.get_performance_report()


# 全局性能监控器实例
performance_monitor = PerformanceMonitor()
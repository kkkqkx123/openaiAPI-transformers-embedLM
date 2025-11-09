"""
测试批处理优化功能
"""

import pytest
import torch
from unittest.mock import Mock, patch
from emb_model_provider.services.batch_optimizer import (
    LengthBasedBatchOptimizer, 
    DynamicBatchOptimizer, 
    BatchProcessingOptimizer
)
from emb_model_provider.core.config import Config
from emb_model_provider.core.performance_monitor import PerformanceMonitor


class TestLengthBasedBatchOptimizer:
    """测试基于长度的批处理优化器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = Config()
        self.config.enable_length_grouping = True
        self.config.length_group_tolerance = 0.2
        
        # Mock tokenizer
        self.mock_tokenizer = Mock()
        
    def test_basic_length_grouping(self):
        """测试基本的长度分组功能"""
        # Mock tokenizer编码结果
        def mock_encode(text, add_special_tokens=False):
            return list(range(len(text.split())))  # 简单按单词数计算长度
        
        self.mock_tokenizer.encode.side_effect = mock_encode
        
        optimizer = LengthBasedBatchOptimizer(self.config, self.mock_tokenizer)
        
        # 测试输入：不同长度的文本
        inputs = [
            "short",           # 1个单词
            "medium length",   # 2个单词
            "very long text with many words",  # 6个单词
            "another medium",  # 2个单词
            "tiny",            # 1个单词
            "extremely long text with many many words"  # 8个单词
        ]
        
        groups = optimizer.optimize_batch(inputs)
        
        # 验证分组结果
        assert len(groups) > 1, "应该产生多个分组"
        
        # 验证每个组内的长度差异不超过容忍度
        for group in groups:
            lengths = [len(text.split()) for text in group.texts]
            max_length = max(lengths)
            min_length = min(lengths)
            
            # 长度差异应该在容忍范围内
            assert max_length <= min_length * (1 + self.config.length_group_tolerance + 0.1), \
                f"组内长度差异过大: {min_length} -> {max_length}"
    
    def test_disabled_length_grouping(self):
        """测试禁用长度分组"""
        self.config.enable_length_grouping = False
        
        # 设置mock tokenizer
        def mock_encode(text, add_special_tokens=False):
            return [1] * len(text.split())
        self.mock_tokenizer.encode.side_effect = mock_encode
        
        optimizer = LengthBasedBatchOptimizer(self.config, self.mock_tokenizer)
        
        inputs = ["short", "medium length", "very long text"]
        groups = optimizer.optimize_batch(inputs)
        
        # 禁用分组时应该返回单个组
        assert len(groups) == 1
        assert groups[0].texts == inputs
    
    def test_single_input(self):
        """测试单个输入"""
        # 设置mock tokenizer
        def mock_encode(text, add_special_tokens=False):
            return [1] * len(text.split())
        self.mock_tokenizer.encode.side_effect = mock_encode
        
        optimizer = LengthBasedBatchOptimizer(self.config, self.mock_tokenizer)
        
        inputs = ["single input"]
        groups = optimizer.optimize_batch(inputs)
        
        # 单个输入应该返回单个组
        assert len(groups) == 1
        assert groups[0].texts == inputs
    
    def test_padding_efficiency_calculation(self):
        """测试padding效率计算"""
        # Mock tokenizer
        def mock_encode(text, add_special_tokens=False):
            return [1] * len(text.split())  # 返回与单词数相同的长度
        
        self.mock_tokenizer.encode.side_effect = mock_encode
        
        optimizer = LengthBasedBatchOptimizer(self.config, self.mock_tokenizer)
        
        # 创建测试组
        from emb_model_provider.services.batch_optimizer import BatchGroup
        groups = [
            BatchGroup(
                texts=["short", "medium"],  # 长度分别为1和2
                indices=[0, 1],
                avg_length=1.5,
                max_length=2,
                size=2
            )
        ]
        
        efficiency = optimizer.calculate_padding_efficiency(groups)
        
        # 验证效率计算
        assert 'padding_ratio' in efficiency
        assert 'efficiency_score' in efficiency
        assert efficiency['efficiency_score'] >= 0
        assert efficiency['efficiency_score'] <= 1


class TestDynamicBatchOptimizer:
    """测试动态批处理优化器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = Config()
        self.config.max_wait_time_ms = 100
        self.config.min_batch_size = 2
        self.config.max_batch_size = 32
        
        self.optimizer = DynamicBatchOptimizer(self.config)
    
    def test_should_wait_for_more_requests(self):
        """测试是否应该等待更多请求"""
        # 测试达到最大批处理大小
        assert not self.optimizer.should_wait_for_more_requests(32, 0.05)
        
        # 测试达到最小批处理大小且等待时间足够
        assert not self.optimizer.should_wait_for_more_requests(2, 0.15)
        
        # 测试应该等待的情况
        assert self.optimizer.should_wait_for_more_requests(1, 0.05)
        assert self.optimizer.should_wait_for_more_requests(2, 0.05)
    
    def test_calculate_optimal_batch_size(self):
        """测试最优批处理大小计算"""
        # 高请求速率
        optimal_size = self.optimizer.calculate_optimal_batch_size(20, 15)
        assert optimal_size == 20  # 不超过可用请求数
        
        # 中等请求速率
        optimal_size = self.optimizer.calculate_optimal_batch_size(20, 7)
        assert optimal_size <= 16  # max_batch_size // 2
        
        # 低请求速率
        optimal_size = self.optimizer.calculate_optimal_batch_size(20, 2)
        assert optimal_size <= 4  # min_batch_size * 2


class TestPerformanceMonitor:
    """测试性能监控器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.monitor = PerformanceMonitor()
    
    def test_monitor_request_context(self):
        """测试请求监控上下文"""
        batch_size = 3
        input_texts = ["text1", "text2", "text3"]
        
        with self.monitor.monitor_request(batch_size, input_texts):
            # 模拟处理时间
            import time
            time.sleep(0.01)
        
        # 验证指标被记录
        report = self.monitor.get_performance_report()
        assert report['request_count'] == batch_size
        assert report['total_requests_processed'] == 1
        assert report['avg_latency'] > 0
    
    def test_performance_report(self):
        """测试性能报告生成"""
        # 添加一些测试数据
        with self.monitor.monitor_request(2, ["a", "b"]):
            pass
        with self.monitor.monitor_request(3, ["c", "d", "e"]):
            pass
        
        report = self.monitor.get_performance_report()
        
        # 验证报告包含所有必要字段
        required_fields = [
            'request_count', 'avg_latency', 'throughput', 
            'avg_batch_size', 'p95_latency', 'p99_latency'
        ]
        
        for field in required_fields:
            assert field in report
    
    def test_reset_metrics(self):
        """测试重置指标"""
        # 添加一些数据
        with self.monitor.monitor_request(1, ["test"]):
            pass
        
        # 重置
        self.monitor.reset_metrics()
        
        # 验证指标被重置
        report = self.monitor.get_performance_report()
        assert report['request_count'] == 0
        assert report['total_requests_processed'] == 0


class TestBatchProcessingOptimizer:
    """测试批处理优化器主类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = Config()
        self.config.enable_length_grouping = True
        
        # Mock tokenizer
        self.mock_tokenizer = Mock()
        def mock_encode(text, add_special_tokens=False):
            return [1] * len(text.split())
        self.mock_tokenizer.encode.side_effect = mock_encode
        
        self.optimizer = BatchProcessingOptimizer(self.config, self.mock_tokenizer)
    
    def test_optimize_batch_processing(self):
        """测试批处理优化流程"""
        inputs = [
            "short text",
            "medium length text here",
            "very long text with many words indeed",
            "another short",
            "medium text"
        ]
        
        groups, efficiency_info = self.optimizer.optimize_batch_processing(inputs)
        
        # 验证返回结果
        assert len(groups) > 0
        assert 'efficiency_score' in efficiency_info
        assert 'padding_ratio' in efficiency_info
        
        # 验证所有输入都被处理
        total_processed = sum(group.size for group in groups)
        assert total_processed == len(inputs)


if __name__ == "__main__":
    pytest.main([__file__])
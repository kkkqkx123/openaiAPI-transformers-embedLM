"""
性能测试，验证并发处理能力
"""

import pytest
import asyncio
import time
import threading
import concurrent.futures
from typing import List, Dict, Any
from fastapi.testclient import TestClient
from emb_model_provider.main import app
from emb_model_provider.core.config import config


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


class TestPerformance:
    """性能测试类"""
    
    def test_single_request_performance(self, client):
        """测试单个请求的性能"""
        test_text = "This is a performance test sentence."
        
        # 预热模型
        warmup_request = {
            "input": test_text,
            "model": config.model_name
        }
        client.post("/v1/embeddings", json=warmup_request)
        
        # 测试多个单个请求的性能
        request_times = []
        for _ in range(10):
            request = {
                "input": test_text,
                "model": config.model_name
            }
            
            start_time = time.time()
            response = client.post("/v1/embeddings", json=request)
            end_time = time.time()
            
            assert response.status_code == 200
            request_times.append(end_time - start_time)
        
        # 计算性能指标
        avg_time = sum(request_times) / len(request_times)
        min_time = min(request_times)
        max_time = max(request_times)
        
        # 验证性能指标 - 调整阈值以适应CPU环境
        assert avg_time < 5.0, f"Average request time too high: {avg_time}s"
        assert max_time < 10.0, f"Maximum request time too high: {max_time}s"
        
        print(f"Single request performance - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
    
    def test_batch_request_performance(self, client):
        """测试批量请求的性能"""
        batch_sizes = [1, 5, 10, 20, 32]  # 不同的批处理大小
        
        for batch_size in batch_sizes:
            # 创建测试文本
            test_texts = [f"This is test sentence {i} for batch size {batch_size}." for i in range(batch_size)]
            
            request = {
                "input": test_texts,
                "model": config.model_name
            }
            
            # 测试批量请求性能
            start_time = time.time()
            response = client.post("/v1/embeddings", json=request)
            end_time = time.time()
            
            assert response.status_code == 200
            
            processing_time = end_time - start_time
            avg_time_per_item = processing_time / batch_size
            
            # 验证批量请求性能 - 调整阈值以适应CPU环境
            assert processing_time < 20.0, f"Batch request time too high for size {batch_size}: {processing_time}s"
            assert avg_time_per_item < 2.0, f"Average time per item too high for batch size {batch_size}: {avg_time_per_item}s"
            
            print(f"Batch size {batch_size} - Total: {processing_time:.3f}s, Avg per item: {avg_time_per_item:.3f}s")
    
    def test_concurrent_requests_performance(self, client):
        """测试并发请求的性能"""
        def make_request(request_data):
            """发送单个请求的函数"""
            start_time = time.time()
            response = client.post("/v1/embeddings", json=request_data)
            end_time = time.time()
            
            assert response.status_code == 200
            return end_time - start_time
        
        # 测试不同并发级别
        concurrency_levels = [1, 5, 10, 20]
        
        for concurrency in concurrency_levels:
            # 创建请求数据
            requests = []
            for i in range(concurrency):
                request_data = {
                    "input": f"Concurrent test sentence {i}",
                    "model": config.model_name
                }
                requests.append(request_data)
            
            # 使用线程池进行并发请求
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_request = {executor.submit(make_request, req): req for req in requests}
                request_times = []
                
                for future in concurrent.futures.as_completed(future_to_request):
                    request_time = future.result()
                    request_times.append(request_time)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 计算性能指标
            avg_request_time = sum(request_times) / len(request_times)
            max_request_time = max(request_times)
            min_request_time = min(request_times)
            throughput = concurrency / total_time  # 请求/秒
            
            # 验证并发性能 - 调整阈值以适应CPU环境
            assert total_time < 15.0, f"Total time too high for concurrency {concurrency}: {total_time}s"
            assert max_request_time < 10.0, f"Max request time too high for concurrency {concurrency}: {max_request_time}s"
            assert throughput > 0.1, f"Throughput too low for concurrency {concurrency}: {throughput} req/s"
            
            print(f"Concurrency {concurrency} - Total: {total_time:.3f}s, Avg: {avg_request_time:.3f}s, "
                  f"Min: {min_request_time:.3f}s, Max: {max_request_time:.3f}s, Throughput: {throughput:.2f} req/s")
    
    def test_mixed_workload_performance(self, client):
        """测试混合工作负载的性能"""
        def make_single_request():
            """发送单个请求"""
            request = {
                "input": "Single request test",
                "model": config.model_name
            }
            start_time = time.time()
            response = client.post("/v1/embeddings", json=request)
            end_time = time.time()
            assert response.status_code == 200
            return end_time - start_time
        
        def make_batch_request(batch_size):
            """发送批量请求"""
            texts = [f"Batch request test {i}" for i in range(batch_size)]
            request = {
                "input": texts,
                "model": config.model_name
            }
            start_time = time.time()
            response = client.post("/v1/embeddings", json=request)
            end_time = time.time()
            assert response.status_code == 200
            return end_time - start_time
        
        # 混合工作负载：单个请求和批量请求
        start_time = time.time()
        
        # 发送混合请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            # 添加单个请求
            for _ in range(5):
                futures.append(executor.submit(make_single_request))
            
            # 添加不同大小的批量请求
            for batch_size in [2, 5, 10]:
                futures.append(executor.submit(make_batch_request, batch_size))
            
            # 等待所有请求完成
            request_times = []
            for future in concurrent.futures.as_completed(futures):
                request_time = future.result()
                request_times.append(request_time)
        
        end_time = time.time()
        total_time = end_time - start_time
        total_requests = len(request_times)
        avg_request_time = sum(request_times) / len(request_times)
        
        # 验证混合工作负载性能
        assert total_time < 20.0, f"Mixed workload time too high: {total_time}s"
        assert avg_request_time < 5.0, f"Average request time too high: {avg_request_time}s"
        
        print(f"Mixed workload - Total: {total_time:.3f}s, Requests: {total_requests}, "
              f"Avg: {avg_request_time:.3f}s")
    
    def test_memory_usage_stability(self, client):
        """测试内存使用稳定性"""
        import psutil
        import os
        
        # 获取当前进程
        process = psutil.Process(os.getpid())
        
        # 记录初始内存使用
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行大量请求
        for i in range(50):
            request = {
                "input": f"Memory stability test sentence {i}",
                "model": config.model_name
            }
            
            response = client.post("/v1/embeddings", json=request)
            assert response.status_code == 200
            
            # 每10个请求检查一次内存使用
            if (i + 1) % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # 验证内存增长不会过大
                assert memory_increase < 500, f"Memory increase too large: {memory_increase}MB"
                
                print(f"Request {i+1} - Memory: {current_memory:.1f}MB, Increase: {memory_increase:.1f}MB")
    
    def test_long_running_stability(self, client):
        """测试长时间运行稳定性"""
        # 测试长时间运行的稳定性
        start_time = time.time()
        request_count = 0
        error_count = 0
        
        # 运行5分钟的持续请求
        while time.time() - start_time < 60:  # 1分钟测试（可以调整为更长）
            try:
                request = {
                    "input": f"Long running stability test {request_count}",
                    "model": config.model_name
                }
                
                response = client.post("/v1/embeddings", json=request)
                if response.status_code != 200:
                    error_count += 1
                
                request_count += 1
                
                # 每10个请求暂停一小段时间
                if request_count % 10 == 0:
                    time.sleep(0.1)
                    
            except Exception as e:
                error_count += 1
                print(f"Error in request {request_count}: {e}")
        
        total_time = time.time() - start_time
        success_rate = (request_count - error_count) / request_count if request_count > 0 else 0
        requests_per_second = request_count / total_time if total_time > 0 else 0
        
        # 确保至少有一些请求被处理
        assert request_count > 0, "No requests were processed"
        
        # 验证长时间运行稳定性
        assert success_rate > 0.95, f"Success rate too low: {success_rate}"
        assert requests_per_second > 0.5, f"Requests per second too low: {requests_per_second}"
        
        print(f"Long running test - Time: {total_time:.1f}s, Requests: {request_count}, "
              f"Errors: {error_count}, Success rate: {success_rate:.2%}, "
              f"RPS: {requests_per_second:.2f}")
    
    def test_scalability_limits(self, client):
        """测试可扩展性限制"""
        # 测试接近批处理大小限制的情况
        max_batch_size = config.max_batch_size
        
        # 创建接近最大批处理大小的请求
        large_batch = [f"Scalability test sentence {i}" for i in range(max_batch_size)]
        
        request = {
            "input": large_batch,
            "model": config.model_name
        }
        
        start_time = time.time()
        response = client.post("/v1/embeddings", json=request)
        end_time = time.time()
        
        assert response.status_code == 200
        
        processing_time = end_time - start_time
        avg_time_per_item = processing_time / max_batch_size
        
        # 验证最大批处理大小的性能
        assert processing_time < 30.0, f"Max batch size processing time too high: {processing_time}s"
        assert avg_time_per_item < 1.0, f"Average time per item too high for max batch: {avg_time_per_item}s"
        
        print(f"Max batch size ({max_batch_size}) - Total: {processing_time:.3f}s, "
              f"Avg per item: {avg_time_per_item:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__])
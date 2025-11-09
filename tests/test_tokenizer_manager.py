"""
测试线程安全的tokenizer管理器
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch
from emb_model_provider.core.tokenizer_manager import (
    ThreadSafeTokenizerManager, 
    GlobalTokenizerManager,
    initialize_tokenizer_manager,
    get_tokenizer_manager
)


class TestThreadSafeTokenizerManager:
    """测试线程安全的tokenizer管理器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.model_path = "test_model_path"
        
    @patch('emb_model_provider.core.tokenizer_manager.AutoTokenizer')
    def test_initialization(self, mock_auto_tokenizer):
        """测试初始化"""
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        manager = ThreadSafeTokenizerManager(self.model_path)
        
        assert manager.model_path == self.model_path
        assert manager._master_tokenizer == mock_tokenizer
        mock_auto_tokenizer.from_pretrained.assert_called_once()
    
    @patch('emb_model_provider.core.tokenizer_manager.AutoTokenizer')
    def test_thread_local_strategy(self, mock_auto_tokenizer):
        """测试线程本地存储策略"""
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        manager = ThreadSafeTokenizerManager(
            self.model_path, 
            use_thread_local=True
        )
        
        # 获取tokenizer
        tokenizer1 = manager.get_tokenizer()
        tokenizer2 = manager.get_tokenizer()
        
        # 在同一线程中应该返回同一个实例
        assert tokenizer1 is tokenizer2
    
    @patch('emb_model_provider.core.tokenizer_manager.AutoTokenizer')
    @patch('emb_model_provider.core.tokenizer_manager.copy.deepcopy')
    def test_tokenizer_copy_creation(self, mock_deepcopy, mock_auto_tokenizer):
        """测试tokenizer副本创建"""
        mock_master_tokenizer = Mock()
        mock_tokenizer_copy = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_master_tokenizer
        mock_deepcopy.return_value = mock_tokenizer_copy
        
        manager = ThreadSafeTokenizerManager(self.model_path)
        tokenizer = manager.get_tokenizer()
        
        # 验证深拷贝被调用
        mock_deepcopy.assert_called_once_with(mock_master_tokenizer)
        assert tokenizer == mock_tokenizer_copy
    
    @patch('emb_model_provider.core.tokenizer_manager.AutoTokenizer')
    @patch('emb_model_provider.core.tokenizer_manager.copy.deepcopy')
    def test_context_manager(self, mock_deepcopy, mock_auto_tokenizer):
        """测试上下文管理器"""
        mock_tokenizer = Mock()
        mock_tokenizer_copy = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_deepcopy.return_value = mock_tokenizer_copy
        
        manager = ThreadSafeTokenizerManager(
            self.model_path,
            use_thread_local=False  # 使用池策略
        )
        
        # 使用上下文管理器
        with manager.get_tokenizer_context() as tokenizer:
            assert tokenizer == mock_tokenizer_copy
        
        # 验证tokenizer被释放回池中
        assert len(manager._pool_available) == 1
    
    def test_get_tokenizer_info(self):
        """测试获取tokenizer信息"""
        with patch('emb_model_provider.core.tokenizer_manager.AutoTokenizer') as mock_auto_tokenizer:
            mock_tokenizer = Mock()
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            manager = ThreadSafeTokenizerManager(self.model_path)
            info = manager.get_tokenizer_info()
            
            assert info["model_path"] == self.model_path
            assert info["strategy"] == "thread_local"
            assert info["master_tokenizer_loaded"] is True
    
    def test_cleanup(self):
        """测试清理资源"""
        with patch('emb_model_provider.core.tokenizer_manager.AutoTokenizer') as mock_auto_tokenizer:
            mock_tokenizer = Mock()
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            manager = ThreadSafeTokenizerManager(self.model_path)
            manager.cleanup()
            
            assert manager._master_tokenizer is None


class TestGlobalTokenizerManager:
    """测试全局tokenizer管理器"""
    
    def setup_method(self):
        """设置测试环境"""
        # 清理全局实例
        GlobalTokenizerManager._instance = None
    
    def teardown_method(self):
        """清理测试环境"""
        # 清理全局实例
        GlobalTokenizerManager._instance = None
    
    @patch('emb_model_provider.core.tokenizer_manager.ThreadSafeTokenizerManager')
    def test_initialize(self, mock_manager_class):
        """测试初始化全局管理器"""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        result = initialize_tokenizer_manager("test_path")
        
        assert result == mock_manager
        mock_manager_class.assert_called_once_with("test_path")
    
    @patch('emb_model_provider.core.tokenizer_manager.ThreadSafeTokenizerManager')
    def test_get_instance(self, mock_manager_class):
        """测试获取全局实例"""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # 初始化
        initialize_tokenizer_manager("test_path")
        
        # 获取实例
        result = get_tokenizer_manager()
        
        assert result == mock_manager
    
    def test_get_instance_not_initialized(self):
        """测试未初始化时获取实例"""
        with pytest.raises(RuntimeError) as exc_info:
            get_tokenizer_manager()
        
        assert "not initialized" in str(exc_info.value)
    
    @patch('emb_model_provider.core.tokenizer_manager.ThreadSafeTokenizerManager')
    def test_cleanup(self, mock_manager_class):
        """测试清理全局管理器"""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # 初始化
        initialize_tokenizer_manager("test_path")
        
        # 清理
        GlobalTokenizerManager.cleanup()
        
        # 验证清理被调用
        mock_manager.cleanup.assert_called_once()
        assert GlobalTokenizerManager._instance is None


class TestConcurrentAccess:
    """测试并发访问"""
    
    def setup_method(self):
        """设置测试环境"""
        self.model_path = "test_model_path"
        GlobalTokenizerManager._instance = None
    
    def teardown_method(self):
        """清理测试环境"""
        GlobalTokenizerManager._instance = None
    
    @patch('emb_model_provider.core.tokenizer_manager.AutoTokenizer')
    def test_concurrent_tokenizer_access(self, mock_auto_tokenizer):
        """测试并发tokenizer访问"""
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # 初始化全局管理器
        initialize_tokenizer_manager(self.model_path)
        
        # 并发访问结果
        results = []
        errors = []
        
        def worker():
            try:
                manager = get_tokenizer_manager()
                tokenizer = manager.get_tokenizer()
                # 模拟一些工作
                time.sleep(0.01)
                results.append(tokenizer is not None)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(results), "Not all workers got valid tokenizers"
    
    @patch('emb_model_provider.core.tokenizer_manager.AutoTokenizer')
    def test_concurrent_context_manager(self, mock_auto_tokenizer):
        """测试并发上下文管理器使用"""
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # 初始化全局管理器（使用池策略）
        initialize_tokenizer_manager(self.model_path, use_thread_local=False)
        
        # 并发访问结果
        results = []
        errors = []
        
        def worker():
            try:
                manager = get_tokenizer_manager()
                with manager.get_tokenizer_context() as tokenizer:
                    # 模拟一些工作
                    time.sleep(0.01)
                    results.append(tokenizer is not None)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(results), "Not all workers got valid tokenizers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
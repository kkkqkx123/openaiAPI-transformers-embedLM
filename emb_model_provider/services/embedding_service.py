from __future__ import annotations

from typing import List, Union, Any, Optional
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from emb_model_provider.api.embeddings import EmbeddingRequest, EmbeddingResponse, EmbeddingData, Usage
from emb_model_provider.api.exceptions import EmbeddingAPIError, BatchSizeExceededError, ContextLengthExceededError, ModelNotFoundError
from emb_model_provider.core.config import Config
from emb_model_provider.core.performance_monitor import performance_monitor
from emb_model_provider.core.tokenizer_manager import initialize_tokenizer_manager, get_tokenizer_manager
from emb_model_provider.services.batch_optimizer import BatchProcessingOptimizer, BatchGroup
import os
import time
from emb_model_provider.core.model_manager import ModelManager
from emb_model_provider.services.realtime_batch_processor import RealtimeBatchProcessor
import asyncio


class EmbeddingService:
    """
    嵌入服务类，负责处理嵌入请求和生成嵌入向量
    支持多模型架构和动态模型加载
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # 预加载所有配置的模型
        if config.enable_multi_source_loading:
            from emb_model_provider.core.model_manager import preload_models
            preload_models()
        
        # 初始化默认模型管理器（用于向后兼容）
        self.default_model_manager = None
        self.model_managers = {}  # 存储不同模型的manager实例
        self.tokenizer_managers = {}  # 存储不同模型的tokenizer管理器
        
        # 启动性能监控
        performance_monitor.start_monitoring()
    
    def _get_model_manager(self, model_alias: str) -> ModelManager:
        """获取指定模型的ModelManager实例"""
        from emb_model_provider.core.model_manager import get_model_manager
        
        # 检查模型是否已加载
        if model_alias in self.model_managers:
            return self.model_managers[model_alias]
        
        # 获取新的ModelManager实例
        model_manager = get_model_manager(model_alias)
        
        # 如果动态加载被禁用且模型未预加载，则抛出错误
        if not self.config.enable_dynamic_model_loading and not self.config.is_model_preloaded(model_alias):
            raise ModelNotFoundError(model_name=model_alias)
        
        # 加载模型（如果动态加载启用）
        if not model_manager.is_model_loaded():
            model_manager.load_model()
        
        # 缓存manager实例
        self.model_managers[model_alias] = model_manager
        return model_manager
    
    def _get_tokenizer_manager(self, model_alias: str):
        """获取指定模型的tokenizer管理器"""
        model_manager = self._get_model_manager(model_alias)
        model_config = self.config.get_model_info(model_alias)
        
        # 为每个模型创建独立的tokenizer管理器
        if model_alias not in self.tokenizer_managers:
            self.tokenizer_managers[model_alias] = initialize_tokenizer_manager(
                model_config["path"],
                use_thread_local=True,
                pool_size=4
            )
        
        return self.tokenizer_managers[model_alias]
    
    def get_tokenizer(self, model_alias: str) -> Any:
        """获取指定模型的线程安全tokenizer实例"""
        tokenizer_manager = self._get_tokenizer_manager(model_alias)
        return tokenizer_manager.get_tokenizer()
        
    def validate_request(self, request: EmbeddingRequest) -> None:
        """
        验证请求参数
        """
        # 检查输入是否为空
        if not request.input:
            raise EmbeddingAPIError(
                message="Input cannot be empty.",
                error_type="invalid_request_error",
                param="input"
            )
        
        # 验证模型存在
        model_info = self.config.get_model_info(request.model)
        if not model_info:
            # 如果没有找到，则不支持该模型
            raise ModelNotFoundError(model_name=request.model)
        
        # 检查动态加载设置
        if not self.config.enable_dynamic_model_loading and not self.config.is_model_preloaded(request.model):
            raise ModelNotFoundError(model_name=request.model)
        
        # 将输入转换为列表进行统一处理
        inputs = request.input if isinstance(request.input, list) else [request.input]
        
        # 检查批处理大小是否超出限制
        if len(inputs) > self.config.max_batch_size:
            raise BatchSizeExceededError(
                max_size=self.config.max_batch_size,
                actual_size=len(inputs)
            )
        
        # 检查每个输入的上下文长度（使用指定模型的tokenizer）
        tokenizer = self.get_tokenizer(request.model)
        for input_text in inputs:
            tokens = tokenizer.encode(input_text, add_special_tokens=True)
            if len(tokens) > self.config.max_context_length:
                raise ContextLengthExceededError(
                    max_length=self.config.max_context_length,
                    actual_length=len(tokens)
                )
    
    def generate_embeddings(self, inputs: List[str], model_alias: str) -> List[EmbeddingData]:
        """
        生成嵌入向量（多模型优化版本）
        """
        # 获取对应的模型管理器
        model_manager = self._get_model_manager(model_alias)
        model = model_manager.model
        device = model_manager.device
        
        # 为当前模型创建批处理优化器
        batch_optimizer = BatchProcessingOptimizer(self.config, self.get_tokenizer(model_alias))
        
        # 使用性能监控
        with performance_monitor.monitor_request(len(inputs), inputs):
            # 1. 批处理优化：按长度分组
            batch_groups, efficiency_info = batch_optimizer.optimize_batch_processing(inputs)
            
            # 2. 处理每个批处理组
            all_embeddings = []
            original_indices = []
            
            for group in batch_groups:
                group_embeddings = self._process_batch_group(group, model, device, model_alias)
                all_embeddings.extend(group_embeddings)
                original_indices.extend(group.indices)
            
            # 3. 恢复原始顺序
            sorted_embeddings: List[Optional[List[float]]] = [None] * len(all_embeddings)
            for i, original_idx in enumerate(original_indices):
                sorted_embeddings[original_idx] = all_embeddings[i]
            
            # 4. 创建EmbeddingData对象列表
            embedding_data_list: List[EmbeddingData] = []
            for i, embedding in enumerate(sorted_embeddings):
                if embedding is not None:
                    embedding_data = EmbeddingData(
                        embedding=embedding,
                        index=i
                    )
                    embedding_data_list.append(embedding_data)
            
            return embedding_data_list
    
    def _process_batch_group(self, group: BatchGroup, model: Any, device: str, model_alias: str) -> List[List[float]]:
        """
        处理单个批处理组（多模型线程安全版本）
        """
        # 使用指定模型的线程安全tokenizer
        tokenizer_manager = self._get_tokenizer_manager(model_alias)
        with tokenizer_manager.get_tokenizer_context() as tokenizer:
            # 对输入进行编码，使用组内最大长度减少padding
            encoded_inputs = tokenizer(
                group.texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=min(group.max_length + 2, self.config.max_context_length)  # +2 for special tokens
            )
        
        # 确保输入张量在正确的设备上
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        
        # 生成嵌入
        with torch.no_grad():
            if model is None:
                raise RuntimeError(f"Model '{model_alias}' not initialized")
            model_output = model(**encoded_inputs)
            # 使用mean pooling获取句子嵌入
            embeddings = self._mean_pooling(model_output, encoded_inputs['attention_mask'])
            # 归一化嵌入向量
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 转换为列表格式
        embeddings_list = embeddings.tolist()
        
        return embeddings_list
    
    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> Any:
        """
        使用attention mask进行平均池化
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def count_tokens(self, inputs: Union[str, List[str]], model_alias: str) -> int:
        """
        计算输入的token数量（多模型线程安全版本）
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        total_tokens = 0
        tokenizer = self.get_tokenizer(model_alias)
        for input_text in inputs:
            tokens = tokenizer.encode(input_text, add_special_tokens=True)
            total_tokens += len(tokens)
        
        return total_tokens
    
    def create_embedding_response(self, request: EmbeddingRequest, embedding_data: List[EmbeddingData]) -> EmbeddingResponse:
        """
        创建嵌入响应
        """
        # 计算token使用情况
        prompt_tokens = self.count_tokens(request.input, request.model)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens
        )
        
        return EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage=usage
        )
    
    def process_embedding_request(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        处理嵌入请求的完整流程（多模型版本）
        """
        # 验证请求
        self.validate_request(request)
        
        # 将输入转换为列表进行统一处理
        inputs = request.input if isinstance(request.input, list) else [request.input]
        
        # 生成嵌入向量
        embedding_data = self.generate_embeddings(inputs, request.model)
        
        # 创建响应
        return self.create_embedding_response(request, embedding_data)
        
        return response
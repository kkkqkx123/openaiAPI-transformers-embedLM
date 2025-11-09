from typing import List, Union
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


class EmbeddingService:
    """
    嵌入服务类，负责处理嵌入请求和生成嵌入向量
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager(config.model_path)
        self.model_manager.load_model()  # 显式加载模型
        self.model = self.model_manager.model
        # 确保模型在正确的设备上
        self.device = self.model_manager.device
        
        # 初始化线程安全的tokenizer管理器
        self.tokenizer_manager = initialize_tokenizer_manager(
            config.model_path,
            use_thread_local=True,  # 使用线程本地存储策略
            pool_size=4
        )
        
        # 初始化批处理优化器（使用tokenizer管理器）
        self.batch_optimizer = BatchProcessingOptimizer(config, self.get_tokenizer())
        
        # 启动性能监控
        performance_monitor.start_monitoring()
    
    def get_tokenizer(self):
        """获取线程安全的tokenizer实例"""
        return self.tokenizer_manager.get_tokenizer()
        
    def validate_request(self, request: EmbeddingRequest) -> None:
        """
        验证请求参数
        """
        # 检查模型名称是否匹配
        if request.model != self.config.model_name:
            raise ModelNotFoundError(model_name=request.model)
        
        # 检查输入是否为空
        if not request.input:
            raise EmbeddingAPIError(
                message="Input cannot be empty.",
                error_type="invalid_request_error",
                param="input"
            )
        
        # 将输入转换为列表进行统一处理
        inputs = request.input if isinstance(request.input, list) else [request.input]
        
        # 检查批处理大小是否超出限制
        if len(inputs) > self.config.max_batch_size:
            raise BatchSizeExceededError(
                max_size=self.config.max_batch_size,
                actual_size=len(inputs)
            )
        
        # 检查每个输入的上下文长度（使用线程安全的tokenizer）
        tokenizer = self.get_tokenizer()
        for input_text in inputs:
            tokens = tokenizer.encode(input_text, add_special_tokens=True)
            if len(tokens) > self.config.max_context_length:
                raise ContextLengthExceededError(
                    max_length=self.config.max_context_length,
                    actual_length=len(tokens)
                )
    
    def generate_embeddings(self, inputs: List[str]) -> List[EmbeddingData]:
        """
        生成嵌入向量（优化版本）
        """
        # 使用性能监控
        with performance_monitor.monitor_request(len(inputs), inputs):
            # 1. 批处理优化：按长度分组
            batch_groups, efficiency_info = self.batch_optimizer.optimize_batch_processing(inputs)
            
            # 2. 处理每个批处理组
            all_embeddings = []
            original_indices = []
            
            for group in batch_groups:
                group_embeddings = self._process_batch_group(group)
                all_embeddings.extend(group_embeddings)
                original_indices.extend(group.indices)
            
            # 3. 恢复原始顺序
            sorted_embeddings = [None] * len(all_embeddings)
            for i, original_idx in enumerate(original_indices):
                sorted_embeddings[original_idx] = all_embeddings[i]
            
            # 4. 创建EmbeddingData对象列表
            embedding_data_list = []
            for i, embedding in enumerate(sorted_embeddings):
                embedding_data = EmbeddingData(
                    embedding=embedding,
                    index=i
                )
                embedding_data_list.append(embedding_data)
            
            return embedding_data_list
    
    def _process_batch_group(self, group: BatchGroup) -> List[List[float]]:
        """
        处理单个批处理组（线程安全版本）
        """
        # 使用线程安全的tokenizer
        with self.tokenizer_manager.get_tokenizer_context() as tokenizer:
            # 对输入进行编码，使用组内最大长度减少padding
            encoded_inputs = tokenizer(
                group.texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=min(group.max_length + 2, self.config.max_context_length)  # +2 for special tokens
            )
        
        # 确保输入张量在正确的设备上
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        
        # 生成嵌入
        with torch.no_grad():
            model_output = self.model(**encoded_inputs)
            # 使用mean pooling获取句子嵌入
            embeddings = self._mean_pooling(model_output, encoded_inputs['attention_mask'])
            # 归一化嵌入向量
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 转换为列表格式
        embeddings_list = embeddings.tolist()
        
        return embeddings_list
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        使用attention mask进行平均池化
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def count_tokens(self, inputs: Union[str, List[str]]) -> int:
        """
        计算输入的token数量（线程安全版本）
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        total_tokens = 0
        tokenizer = self.get_tokenizer()
        for input_text in inputs:
            tokens = tokenizer.encode(input_text, add_special_tokens=True)
            total_tokens += len(tokens)
        
        return total_tokens
    
    def create_embedding_response(self, request: EmbeddingRequest, embedding_data: List[EmbeddingData]) -> EmbeddingResponse:
        """
        创建嵌入响应
        """
        # 计算token使用情况
        prompt_tokens = self.count_tokens(request.input)
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
        处理嵌入请求的完整流程
        """
        # 验证请求
        self.validate_request(request)
        
        # 将输入转换为列表
        inputs = request.input if isinstance(request.input, list) else [request.input]
        
        # 生成嵌入
        embedding_data = self.generate_embeddings(inputs)
        
        # 创建响应
        response = self.create_embedding_response(request, embedding_data)
        
        return response
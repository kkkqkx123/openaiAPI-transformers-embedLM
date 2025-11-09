from typing import List, Union
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from emb_model_provider.api.embeddings import EmbeddingRequest, EmbeddingResponse, EmbeddingData, Usage
from emb_model_provider.api.exceptions import EmbeddingAPIError, BatchSizeExceededError, ContextLengthExceededError
from emb_model_provider.core.config import Config
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
        self.tokenizer = self.model_manager.tokenizer
        self.model = self.model_manager.model
        
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
        
        # 将输入转换为列表进行统一处理
        inputs = request.input if isinstance(request.input, list) else [request.input]
        
        # 检查批处理大小是否超出限制
        if len(inputs) > self.config.max_batch_size:
            raise BatchSizeExceededError(
                max_size=self.config.max_batch_size,
                actual_size=len(inputs)
            )
        
        # 检查每个输入的上下文长度
        for input_text in inputs:
            tokens = self.tokenizer.encode(input_text, add_special_tokens=True)
            if len(tokens) > self.config.max_context_length:
                raise ContextLengthExceededError(
                    max_length=self.config.max_context_length,
                    actual_length=len(tokens)
                )
    
    def generate_embeddings(self, inputs: List[str]) -> List[EmbeddingData]:
        """
        生成嵌入向量
        """
        # 对输入进行编码
        encoded_inputs = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.config.max_context_length
        )
        
        # 生成嵌入
        with torch.no_grad():
            model_output = self.model(**encoded_inputs)
            # 使用mean pooling获取句子嵌入
            embeddings = self._mean_pooling(model_output, encoded_inputs['attention_mask'])
            # 归一化嵌入向量
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 转换为列表格式
        embeddings_list = embeddings.tolist()
        
        # 创建EmbeddingData对象列表
        embedding_data_list = []
        for i, embedding in enumerate(embeddings_list):
            embedding_data = EmbeddingData(
                embedding=embedding,
                index=i
            )
            embedding_data_list.append(embedding_data)
        
        return embedding_data_list
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        使用attention mask进行平均池化
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def count_tokens(self, inputs: Union[str, List[str]]) -> int:
        """
        计算输入的token数量
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        total_tokens = 0
        for input_text in inputs:
            tokens = self.tokenizer.encode(input_text, add_special_tokens=True)
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
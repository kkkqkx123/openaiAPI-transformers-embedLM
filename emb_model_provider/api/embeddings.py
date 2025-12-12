from typing import List, Union, Optional, Any
from pydantic import BaseModel
from fastapi import APIRouter
from emb_model_provider.core.performance_monitor import performance_monitor
from emb_model_provider.core.config import config
import asyncio


# 创建API路由器
router = APIRouter()


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


# 全局 variable，用于存储 EmbeddingService 实例
_embedding_service_instance = None
_realtime_batch_processor = None  # Global instance of the batch processor

# 延迟导入服务，避免循环导入
def get_embedding_service() -> Any:
    global _embedding_service_instance, _realtime_batch_processor
    if _embedding_service_instance is None:
        from emb_model_provider.services.embedding_service import EmbeddingService
        from emb_model_provider.services.realtime_batch_processor import RealtimeBatchProcessor
        from emb_model_provider.core.config import config
        
        _embedding_service_instance = EmbeddingService(config)
        
        # Initialize the real-time batch processor only if dynamic batching is enabled
        if config.enable_dynamic_batching:
            _realtime_batch_processor = RealtimeBatchProcessor(config, _embedding_service_instance)
            # Initialize the processor with the embedding service
            asyncio.create_task(_realtime_batch_processor.start())
    
    return _embedding_service_instance


def get_realtime_batch_processor() -> Any:
    """Get the instance of the real-time batch processor"""
    global _realtime_batch_processor
    if _realtime_batch_processor is None:
        get_embedding_service()  # This will initialize the batch processor if needed
    return _realtime_batch_processor


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    创建嵌入向量的API端点
    对应需求: UB-1.1, UB-1.2, UB-1.3
    """
    # Get the embedding service
    embedding_service = get_embedding_service()
    
    # Use real-time batching if enabled
    if config.enable_dynamic_batching:
        # Submit request to real-time batch processor
        batch_processor = get_realtime_batch_processor()
        embedding_data = await batch_processor.submit_request(request)
    else:
        # Process directly without batching
        # This maintains backward compatibility
        response = embedding_service.process_embedding_request(request)
        embedding_data = response.data
    
    # Calculate token usage
    prompt_tokens = embedding_service.count_tokens(request.input, request.model)
    usage = Usage(
        prompt_tokens=prompt_tokens,
        total_tokens=prompt_tokens
    )
    
    return EmbeddingResponse(
        data=embedding_data,
        model=request.model,
        usage=usage
    )


@router.get("/v1/performance")
async def get_performance_metrics() -> Any:
    """
    获取性能监控指标
    """
    return performance_monitor.get_performance_report()


@router.post("/v1/performance/reset")
async def reset_performance_metrics() -> Any:
    """
    重置性能监控指标
    """
    performance_monitor.reset_metrics()
    return {"message": "Performance metrics reset successfully"}
from typing import List, Union, Optional
from pydantic import BaseModel
from fastapi import APIRouter
from emb_model_provider.core.performance_monitor import performance_monitor


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


# 全局变量，用于存储 EmbeddingService 实例
_embedding_service_instance = None

# 延迟导入服务，避免循环导入
def get_embedding_service():
    global _embedding_service_instance
    if _embedding_service_instance is None:
        from emb_model_provider.services.embedding_service import EmbeddingService
        from emb_model_provider.core.config import config
        _embedding_service_instance = EmbeddingService(config)
    return _embedding_service_instance


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    创建嵌入向量的API端点
    对应需求: UB-1.1, UB-1.2, UB-1.3
    """
    # 延迟导入服务，避免循环导入
    embedding_service = get_embedding_service()
    # 处理嵌入请求
    response = embedding_service.process_embedding_request(request)
    return response


@router.get("/v1/performance")
async def get_performance_metrics():
    """
    获取性能监控指标
    """
    return performance_monitor.get_performance_report()


@router.post("/v1/performance/reset")
async def reset_performance_metrics():
    """
    重置性能监控指标
    """
    performance_monitor.reset_metrics()
    return {"message": "Performance metrics reset successfully"}
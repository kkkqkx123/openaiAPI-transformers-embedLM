from typing import List, Union, Optional
from pydantic import BaseModel
from fastapi import APIRouter


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


# 延迟导入服务，避免循环导入
def get_embedding_service():
    from emb_model_provider.services.embedding_service import EmbeddingService
    from emb_model_provider.core.config import config
    return EmbeddingService(config)


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
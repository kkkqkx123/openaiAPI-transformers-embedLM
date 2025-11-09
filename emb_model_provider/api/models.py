from typing import List
from pydantic import BaseModel
import time
from fastapi import APIRouter


# 创建API路由器
router = APIRouter()


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    列出可用模型的API端点
    对应需求: UB-1.4
    """
    # 延迟导入，避免循环导入
    from emb_model_provider.core.config import config
    from emb_model_provider.api.exceptions import ModelNotFoundError
    
    # 验证模型是否存在
    model_path = config.model_path
    import os
    if not os.path.exists(model_path):
        from emb_model_provider.core.model_manager import ModelManager
        try:
            # 尝试初始化模型管理器，这会触发模型下载
            model_manager = ModelManager(model_path)
        except Exception:
            # 如果模型不存在且无法下载，抛出异常
            raise ModelNotFoundError(config.model_name)
    
    model_info = ModelInfo(
        id=config.model_name,
        owned_by="organization-owner"  # 默认值，可根据需要修改
    )
    response = ModelsResponse(data=[model_info])
    return response
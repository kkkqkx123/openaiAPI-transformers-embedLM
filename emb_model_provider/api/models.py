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
async def list_models() -> ModelsResponse:
    """
    列出所有可用模型的API端点（包括别名）
    """
    from emb_model_provider.core.config import config
    
    model_info_list = []
    
    # 添加主模型
    if config.model_name:
        model_info_list.append(ModelInfo(
            id=config.model_name,
            owned_by="organization-owner"
        ))
    
    # 添加映射的模型和别名
    model_mapping = config.get_model_mapping()
    for alias, model_config in model_mapping.items():
        # 获取实际模型名称（从配置字典中）
        actual_model_name = model_config.get("name", alias)
        
        # 添加别名
        model_info_list.append(ModelInfo(
            id=alias,
            owned_by="organization-owner"
        ))
        # 添加实际模型（如果尚未添加）
        if actual_model_name != config.model_name and actual_model_name not in [m.id for m in model_info_list]:
            model_info_list.append(ModelInfo(
                id=actual_model_name,
                owned_by="organization-owner"
            ))
    
    return ModelsResponse(data=model_info_list)
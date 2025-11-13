from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from emb_model_provider.core.config import config
from emb_model_provider.core.logging import setup_logging, shutdown_logging
from emb_model_provider.api.embeddings import router as embeddings_router
from emb_model_provider.api.models import router as models_router
from emb_model_provider.api.middleware import exception_handlers, global_exception_handler
import uvicorn
import logging
import atexit
import asyncio


# 设置日志
setup_logging()

# 注册退出处理函数
atexit.register(shutdown_logging)


# 创建FastAPI应用实例
app = FastAPI(
    title="Embedding Model Provider API",
    description="OpenAI compatible embedding model API provider",
    version="0.1.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)


# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 注册异常处理器
app.add_exception_handler(Exception, global_exception_handler)
for exc_type, handler in exception_handlers.items():
    app.add_exception_handler(exc_type, handler)


# 配置API路由
app.include_router(embeddings_router)
app.include_router(models_router)


@app.get("/health")
async def health_check():
    """
    健康检查端点
    对应需求: 设计文档中的测试策略
    """
    return {"status": "healthy"}


@app.get("/")
async def root():
    """
    根端点，提供API基本信息
    """
    return {
        "message": "Embedding Model Provider API",
        "version": "0.1.0",
        "endpoints": ["/v1/embeddings", "/v1/models", "/health"]
    }


@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭时的清理工作
    """
    from emb_model_provider.api.embeddings import get_realtime_batch_processor
    try:
        batch_processor = get_realtime_batch_processor()
        if batch_processor:
            await batch_processor.stop()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error during shutdown: {e}")


def run_server():
    """
    运行服务器的便捷函数
    """
    uvicorn.run(
        "emb_model_provider.main:app",
        host=config.host,
        port=config.port,
        reload=False,  # 生产环境中应设为False
        log_level=config.log_level.lower()
    )


if __name__ == "__main__":
    run_server()
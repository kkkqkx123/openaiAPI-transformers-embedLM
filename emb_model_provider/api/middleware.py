from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from emb_model_provider.api.exceptions import EmbeddingAPIError
from typing import Union


async def embedding_api_exception_handler(request: Request, exc: Union[EmbeddingAPIError, Exception]) -> JSONResponse:
    """
    嵌入API异常处理器
    根据设计文档中的错误处理部分实现
    """
    if isinstance(exc, EmbeddingAPIError):
        status_code = 400  # 默认状态码

        # 根据错误类型设置状态码
        if exc.type == "context_length_exceeded":
            status_code = 429  # Too Many Requests
        elif exc.type == "batch_size_exceeded":
            status_code = 429  # Too Many Requests
        elif exc.type == "model_not_found":
            status_code = 404  # Not Found
        elif exc.type == "invalid_request_error":
            status_code = 400  # Bad Request
        elif exc.type == "internal_server_error":
            status_code = 500  # Internal Server Error

        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.type,
                    "param": exc.param,
                    "code": exc.code
                }
            }
        )
    else:
        # Fallback for non-EmbeddingAPIError exceptions
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_server_error",
                    "param": None,
                    "code": "500"
                }
            }
        )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    全局异常处理器，处理未预期的异常
    """
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.detail,
                    "type": "http_exception",
                    "param": None,
                    "code": str(exc.status_code)
                }
            }
        )

    # 对于其他异常，返回500错误
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_server_error",
                "param": None,
                "code": "500"
            }
        }
    )


# 定义异常处理映射
exception_handlers = {
    EmbeddingAPIError: embedding_api_exception_handler,
}
from typing import Optional

class EmbeddingAPIError(Exception):
    """嵌入API的基础异常类"""
    def __init__(self, message: str, error_type: str, param: Optional[str] = None, code: Optional[str] = None):
        self.message = message
        self.type = error_type
        self.param = param
        self.code = code
        super().__init__(self.message)


class ContextLengthExceededError(EmbeddingAPIError):
    """上下文长度超出限制的异常"""
    def __init__(self, max_length: int, actual_length: int):
        super().__init__(
            message=f"This model's maximum context length is {max_length} tokens. However, your messages resulted in {actual_length} tokens.",
            error_type="context_length_exceeded",
            param="input"
        )


class BatchSizeExceededError(EmbeddingAPIError):
    """批处理大小超出限制的异常"""
    def __init__(self, max_size: int, actual_size: int):
        super().__init__(
            message=f"Maximum batch size is {max_size}. However, your request contains {actual_size} inputs.",
            error_type="batch_size_exceeded",
            param="input"
        )


class ModelNotFoundError(EmbeddingAPIError):
    """模型未找到的异常"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(
            message=f"Model '{model_name}' not found.",
            error_type="model_not_found",
            param="model"
        )


class InvalidRequestError(EmbeddingAPIError):
    """无效请求的异常"""
    def __init__(self, message: str, param: Optional[str] = None):
        super().__init__(
            message=message,
            error_type="invalid_request_error",
            param=param
        )


class InternalServerError(EmbeddingAPIError):
    """内部服务器错误"""
    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_type="internal_server_error"
        )
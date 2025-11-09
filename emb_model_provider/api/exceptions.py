class EmbeddingAPIError(Exception):
    """嵌入API的基础异常类"""
    def __init__(self, message: str, error_type: str, param: str = None, code: str = None):
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
            error_type="invalid_request_error",
            param="input"
        )


class BatchSizeExceededError(EmbeddingAPIError):
    """批处理大小超出限制的异常"""
    def __init__(self, max_size: int, actual_size: int):
        super().__init__(
            message=f"Maximum batch size is {max_size}. However, your request contains {actual_size} inputs.",
            error_type="invalid_request_error",
            param="input"
        )
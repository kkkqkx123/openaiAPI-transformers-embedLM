# 使用 Python 3.10 作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv 包管理器
RUN pip install uv

# 复制项目文件
COPY pyproject.toml uv.lock ./
COPY emb_model_provider/ ./emb_model_provider/

# 使用 uv 安装依赖
RUN uv sync --frozen --no-dev

# 创建模型目录
RUN mkdir -p /models

# 设置默认环境变量
ENV EMB_PROVIDER_MODEL_PATH=/models/all-MiniLM-L12-v2
ENV EMB_PROVIDER_MODEL_NAME=all-MiniLM-L12-v2
ENV EMB_PROVIDER_MAX_BATCH_SIZE=32
ENV EMB_PROVIDER_MAX_CONTEXT_LENGTH=512
ENV EMB_PROVIDER_EMBEDDING_DIMENSION=384
ENV EMB_PROVIDER_MEMORY_LIMIT=2GB
ENV EMB_PROVIDER_DEVICE=auto
ENV EMB_PROVIDER_HOST=0.0.0.0
ENV EMB_PROVIDER_PORT=9000

# 日志配置（容器环境默认禁用文件日志）
ENV EMB_PROVIDER_LOG_LEVEL=INFO
ENV EMB_PROVIDER_LOG_TO_FILE=false
ENV EMB_PROVIDER_LOG_DIR=/app/logs
ENV EMB_PROVIDER_LOG_FILE_MAX_SIZE=10
ENV EMB_PROVIDER_LOG_RETENTION_DAYS=7
ENV EMB_PROVIDER_LOG_CLEANUP_INTERVAL_HOURS=1
ENV EMB_PROVIDER_LOG_MAX_DIR_SIZE_MB=50
ENV EMB_PROVIDER_LOG_CLEANUP_TARGET_SIZE_MB=20
ENV EMB_PROVIDER_LOG_CLEANUP_RETENTION_DAYS=7,3,1

# 暴露端口
EXPOSE 9000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# 启动命令
CMD ["uv", "run", "python", "-m", "emb_model_provider.main"]
# Docker 设置指南

本文档说明如何使用改进的 Docker 配置来运行 emb-model-provider 服务。

## 目录结构

```
.
├── Dockerfile                 # CPU版本的多阶段构建 Dockerfile
├── Dockerfile.gpu             # GPU版本的 Dockerfile（基于PyTorch镜像）
├── docker-compose.yml         # CPU版本的 Docker Compose 配置
├── docker-compose.gpu.yml     # GPU版本的 Docker Compose 配置
├── .dockerignore              # Docker构建忽略文件
├── init-docker-dirs.sh        # Linux/macOS 初始化脚本
├── init-docker-dirs.bat       # Windows 初始化脚本
├── models/                    # 模型文件目录（需要手动创建）
├── logs/                      # 日志目录（需要手动创建）
└── data/                      # 数据目录（可选）
```

## 快速开始

### 1. 初始化挂载目录

在运行 Docker Compose 之前，需要先创建必要的挂载目录：

#### Linux/macOS
```bash
chmod +x init-docker-dirs.sh
./init-docker-dirs.sh
```

#### Windows
```cmd
init-docker-dirs.bat
```

### 2. 启动服务

#### CPU版本（默认）
```bash
docker-compose up -d
```

#### GPU版本（需要NVIDIA Docker支持）
```bash
docker-compose -f docker-compose.gpu.yml up -d
```

### 3. 查看服务状态

```bash
# CPU版本
docker-compose ps
docker-compose logs -f emb-model-provider

# GPU版本
docker-compose -f docker-compose.gpu.yml ps
docker-compose -f docker-compose.gpu.yml logs -f emb-model-provider
```

## Dockerfile 改进说明

### 多阶段构建优化

新的 Dockerfile 采用优化的多阶段构建，具有以下优势：

1. **更小的镜像大小**：最终镜像只包含运行时依赖，移除了冗余的uv安装
2. **更好的缓存利用**：依赖安装和代码构建分离，添加了.dockerignore优化
3. **更快的构建速度**：代码变更时无需重新安装依赖
4. **更安全的配置**：使用非 root 用户运行应用
5. **依赖分类优化**：将测试依赖移至可选依赖组，减少生产镜像大小

### 构建阶段

- **builder 阶段**：安装构建依赖和 Python 包
- **runtime 阶段**：创建最终运行镜像，只包含必要组件

### GPU支持

新增了GPU版本的Dockerfile (`Dockerfile.gpu`)：

1. **基于PyTorch官方镜像**：包含CUDA运行时环境
2. **优化的GPU配置**：默认使用CUDA设备，支持更大的批次大小
3. **独立的GPU配置**：通过`docker-compose.gpu.yml`管理

### 安全改进

1. **非 root 用户**：应用以 `appuser` 身份运行
2. **最小权限**：只授予必要的目录权限
3. **安全的基础镜像**：使用官方 Python slim 镜像
4. **环境变量优化**：移除Dockerfile中的重复环境变量定义

### 健康检查改进

- 增加了超时限制（5秒）
- 调整了启动等待时间（40秒）
- 优化了重试逻辑

## 挂载目录说明

### `/models` 目录

- **用途**：存储模型文件，避免每次启动重新下载
- **权限**：755 (appuser:appuser)
- **注意**：首次运行前需要下载模型文件到此目录

### `/app/logs` 目录

- **用途**：存储应用日志文件
- **权限**：755 (appuser:appuser)
- **配置**：通过 `EMB_PROVIDER_LOG_TO_FILE` 环境变量控制

### `/app/data` 目录

- **用途**：存储应用数据文件（可选）
- **权限**：755 (appuser:appuser)

## 环境变量配置

### 模型配置

```yaml
EMB_PROVIDER_MODEL_PATH=/models/all-MiniLM-L12-v2
EMB_PROVIDER_MODEL_NAME=all-MiniLM-L12-v2
```

### 性能配置

```yaml
EMB_PROVIDER_MAX_BATCH_SIZE=32
EMB_PROVIDER_MAX_CONTEXT_LENGTH=512
EMB_PROVIDER_EMBEDDING_DIMENSION=384
EMB_PROVIDER_MEMORY_LIMIT=2GB
EMB_PROVIDER_DEVICE=auto
```

### 日志配置

```yaml
EMB_PROVIDER_LOG_LEVEL=INFO
EMB_PROVIDER_LOG_TO_FILE=false
EMB_PROVIDER_LOG_DIR=/app/logs
```

## GPU使用前置条件

### 1. NVIDIA Docker支持

使用GPU版本需要以下环境：

```bash
# 安装NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. 验证GPU支持

```bash
# 测试NVIDIA Docker支持
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## 常见问题

### 1. 权限问题

如果遇到权限错误，确保：

- 挂载目录存在且有正确权限
- 容器内用户有读写权限

### 2. 模型文件缺失

首次运行前需要下载模型文件：

```bash
# 使用提供的下载脚本
python download_model.py

# 或手动下载到 models 目录
```

### 3. 健康检查失败

检查：

- 应用是否正常启动
- 端口 9000 是否可访问
- 健康检查端点是否响应

### 4. 构建缓存问题

如果需要强制重新构建：

```bash
# CPU版本
docker-compose build --no-cache

# GPU版本
docker-compose -f docker-compose.gpu.yml build --no-cache
```

### 5. GPU相关问题

**GPU不可用**：
- 确认NVIDIA驱动已正确安装
- 验证NVIDIA Container Toolkit配置
- 检查docker-compose.gpu.yml中的GPU配置

**CUDA内存不足**：
- 调整EMB_PROVIDER_MAX_BATCH_SIZE环境变量
- 减少EMB_PROVIDER_MEMORY_LIMIT设置
- 监控GPU内存使用情况

## 开发和调试

### 查看构建日志

```bash
docker-compose build --progress=plain
```

### 进入容器调试

```bash
docker-compose exec emb-model-provider bash
```

### 查看详细日志

```bash
docker-compose logs -f --tail=100 emb-model-provider
```

## 性能优化

### 1. 镜像优化

- 使用多阶段构建减小镜像大小
- 清理不必要的包和缓存
- 使用 .dockerignore 排除无关文件

### 2. 运行时优化

- 调整内存限制和批处理大小
- 使用适当的设备配置（CPU/GPU）
- 优化日志级别和输出

### 3. 网络优化

- 使用专用网络
- 配置适当的超时设置
- 启用连接池

## 安全建议

1. **定期更新基础镜像**
2. **使用最小权限原则**
3. **启用内容信任**
4. **扫描镜像漏洞**
5. **使用 secrets 管理敏感信息**

## 故障排除

### 检查服务状态

```bash
docker-compose ps
docker-compose top
```

### 查看资源使用

```bash
docker stats emb-model-provider
```

### 重启服务

```bash
docker-compose restart emb-model-provider
```

### 完全重置

```bash
docker-compose down -v
docker-compose up --build -d
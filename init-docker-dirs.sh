#!/bin/bash

# 初始化 Docker Compose 挂载目录脚本
# 用于创建 emb-model-provider 服务所需的挂载目录

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否在项目根目录
check_project_root() {
    if [[ ! -f "docker-compose.yml" ]] || [[ ! -f "Dockerfile" ]]; then
        log_error "请在项目根目录（包含 docker-compose.yml 和 Dockerfile 的目录）中运行此脚本"
        exit 1
    fi
    log_info "项目根目录检查通过"
}

# 创建模型目录
create_models_dir() {
    local models_dir="./models"
    
    if [[ -d "$models_dir" ]]; then
        log_warn "模型目录 $models_dir 已存在"
    else
        mkdir -p "$models_dir"
        log_info "创建模型目录: $models_dir"
    fi
    
    # 设置目录权限
    chmod 755 "$models_dir"
    log_info "设置模型目录权限: 755"
    
    # 创建 .gitkeep 文件以确保目录被 git 跟踪
    if [[ ! -f "$models_dir/.gitkeep" ]]; then
        touch "$models_dir/.gitkeep"
        log_info "创建 .gitkeep 文件以跟踪模型目录"
    fi
}

# 创建日志目录
create_logs_dir() {
    local logs_dir="./logs"
    
    if [[ -d "$logs_dir" ]]; then
        log_warn "日志目录 $logs_dir 已存在"
    else
        mkdir -p "$logs_dir"
        log_info "创建日志目录: $logs_dir"
    fi
    
    # 设置目录权限
    chmod 755 "$logs_dir"
    log_info "设置日志目录权限: 755"
    
    # 创建 .gitkeep 文件以确保目录被 git 跟踪
    if [[ ! -f "$logs_dir/.gitkeep" ]]; then
        touch "$logs_dir/.gitkeep"
        log_info "创建 .gitkeep 文件以跟踪日志目录"
    fi
}

# 创建其他可能需要的目录
create_additional_dirs() {
    # 创建数据目录（如果需要）
    local data_dir="./data"
    
    if [[ -d "$data_dir" ]]; then
        log_warn "数据目录 $data_dir 已存在"
    else
        mkdir -p "$data_dir"
        log_info "创建数据目录: $data_dir"
        chmod 755 "$data_dir"
        touch "$data_dir/.gitkeep"
        log_info "创建 .gitkeep 文件以跟踪数据目录"
    fi
}

# 检查 Docker 和 Docker Compose 是否可用
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装或不在 PATH 中"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose 未安装或不在 PATH 中"
        exit 1
    fi
    
    log_info "Docker 和 Docker Compose 检查通过"
}

# 显示目录结构
show_directory_structure() {
    log_info "当前项目目录结构:"
    echo ""
    find . -maxdepth 2 -type d \( -name "models" -o -name "logs" -o -name "data" \) | sort
    echo ""
}

# 显示使用提示
show_usage_tips() {
    log_info "初始化完成！使用提示:"
    echo ""
    echo "1. 启动服务:"
    echo "   docker-compose up -d"
    echo ""
    echo "2. 查看日志:"
    echo "   docker-compose logs -f emb-model-provider"
    echo ""
    echo "3. 停止服务:"
    echo "   docker-compose down"
    echo ""
    echo "4. 重新构建并启动:"
    echo "   docker-compose up --build -d"
    echo ""
    echo "注意: 首次启动前，请确保已下载所需的模型文件到 ./models 目录"
}

# 主函数
main() {
    log_info "开始初始化 Docker Compose 挂载目录..."
    echo ""
    
    check_project_root
    check_docker
    echo ""
    
    create_models_dir
    create_logs_dir
    create_additional_dirs
    echo ""
    
    show_directory_structure
    show_usage_tips
}

# 运行主函数
main "$@"
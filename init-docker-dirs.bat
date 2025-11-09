@echo off
REM 初始化 Docker Compose 挂载目录脚本 (Windows 版本)
REM 用于创建 emb-model-provider 服务所需的挂载目录

setlocal enabledelayedexpansion

REM 颜色定义 (Windows 10+ 支持 ANSI 转义序列)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "NC=[0m"

REM 日志函数
:log_info
echo %GREEN%[INFO]%NC% %~1
goto :eof

:log_warn
echo %YELLOW%[WARN]%NC% %~1
goto :eof

:log_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM 检查是否在项目根目录
:check_project_root
if not exist "docker-compose.yml" (
    call :log_error "请在项目根目录（包含 docker-compose.yml 和 Dockerfile 的目录）中运行此脚本"
    exit /b 1
)
if not exist "Dockerfile" (
    call :log_error "请在项目根目录（包含 docker-compose.yml 和 Dockerfile 的目录）中运行此脚本"
    exit /b 1
)
call :log_info "项目根目录检查通过"
goto :eof

REM 创建模型目录
:create_models_dir
set "models_dir=models"

if exist "%models_dir%" (
    call :log_warn "模型目录 %models_dir% 已存在"
) else (
    mkdir "%models_dir%"
    if !errorlevel! equ 0 (
        call :log_info "创建模型目录: %models_dir%"
    ) else (
        call :log_error "无法创建模型目录: %models_dir%"
        exit /b 1
    )
)

REM 创建 .gitkeep 文件以确保目录被 git 跟踪
if not exist "%models_dir%\.gitkeep" (
    echo. > "%models_dir%\.gitkeep"
    call :log_info "创建 .gitkeep 文件以跟踪模型目录"
)
goto :eof

REM 创建日志目录
:create_logs_dir
set "logs_dir=logs"

if exist "%logs_dir%" (
    call :log_warn "日志目录 %logs_dir% 已存在"
) else (
    mkdir "%logs_dir%"
    if !errorlevel! equ 0 (
        call :log_info "创建日志目录: %logs_dir%"
    ) else (
        call :log_error "无法创建日志目录: %logs_dir%"
        exit /b 1
    )
)

REM 创建 .gitkeep 文件以确保目录被 git 跟踪
if not exist "%logs_dir%\.gitkeep" (
    echo. > "%logs_dir%\.gitkeep"
    call :log_info "创建 .gitkeep 文件以跟踪日志目录"
)
goto :eof

REM 创建其他可能需要的目录
:create_additional_dirs
set "data_dir=data"

if exist "%data_dir%" (
    call :log_warn "数据目录 %data_dir% 已存在"
) else (
    mkdir "%data_dir%"
    if !errorlevel! equ 0 (
        call :log_info "创建数据目录: %data_dir%"
    ) else (
        call :log_error "无法创建数据目录: %data_dir%"
        exit /b 1
    )
)

REM 创建 .gitkeep 文件以确保目录被 git 跟踪
if not exist "%data_dir%\.gitkeep" (
    echo. > "%data_dir%\.gitkeep"
    call :log_info "创建 .gitkeep 文件以跟踪数据目录"
)
goto :eof

REM 检查 Docker 和 Docker Compose 是否可用
:check_docker
docker --version >nul 2>&1
if !errorlevel! neq 0 (
    call :log_error "Docker 未安装或不在 PATH 中"
    exit /b 1
)

docker-compose --version >nul 2>&1
if !errorlevel! neq 0 (
    docker compose version >nul 2>&1
    if !errorlevel! neq 0 (
        call :log_error "Docker Compose 未安装或不在 PATH 中"
        exit /b 1
    )
)

call :log_info "Docker 和 Docker Compose 检查通过"
goto :eof

REM 显示目录结构
:show_directory_structure
call :log_info "当前项目目录结构:"
echo.

if exist "models" echo models\
if exist "logs" echo logs\
if exist "data" echo data\
echo.
goto :eof

REM 显示使用提示
:show_usage_tips
call :log_info "初始化完成！使用提示:"
echo.
echo 1. 启动服务:
echo    docker-compose up -d
echo.
echo 2. 查看日志:
echo    docker-compose logs -f emb-model-provider
echo.
echo 3. 停止服务:
echo    docker-compose down
echo.
echo 4. 重新构建并启动:
echo    docker-compose up --build -d
echo.
echo 注意: 首次启动前，请确保已下载所需的模型文件到 .\models 目录
goto :eof

REM 主函数
:main
call :log_info "开始初始化 Docker Compose 挂载目录..."
echo.

call :check_project_root
if !errorlevel! neq 0 exit /b 1

call :check_docker
if !errorlevel! neq 0 exit /b 1
echo.

call :create_models_dir
call :create_logs_dir
call :create_additional_dirs
echo.

call :show_directory_structure
call :show_usage_tips
goto :eof

REM 运行主函数
call :main
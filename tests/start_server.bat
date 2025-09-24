@echo off
chcp 65001 > nul
echo 🚀 启动流式TTS服务器
echo ========================

REM 检查Python是否可用
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安装或未添加到PATH
    pause
    exit /b 1
)

REM 启动服务器
echo 正在启动服务器...
python start_streaming_server.py --host 127.0.0.1 --port 8000

pause

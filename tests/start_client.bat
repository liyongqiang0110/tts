@echo off
chcp 65001 > nul
echo 🎵 流式TTS客户端启动器
echo ========================

echo 请选择启动方式：
echo 1. 打开Web界面客户端
echo 2. 运行Python测试脚本
echo 3. 退出
echo.

set /p choice="请输入选择 (1-3): "

if "%choice%"=="1" (
    echo 正在打开Web界面...
    start "" "streaming_web_client.html"
    echo ✅ Web界面已打开
) else if "%choice%"=="2" (
    echo 正在运行Python测试...
    python simple_streaming_test.py
) else if "%choice%"=="3" (
    echo 👋 再见！
    exit /b 0
) else (
    echo ❌ 无效选择，请重新运行
)

echo.
pause

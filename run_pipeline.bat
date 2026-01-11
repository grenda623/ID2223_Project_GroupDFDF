@echo off
REM =====================================================
REM 电价预测系统 - API版本一键运行脚本
REM =====================================================

echo =====================================================
echo 电价预测系统 - API版本
echo =====================================================
echo.

REM 设置Python路径（根据实际情况修改）
set PYTHON=D:\anaconda\python.exe

REM 检查Python是否存在
if not exist "%PYTHON%" (
    echo [ERROR] Python未找到: %PYTHON%
    echo 请修改脚本中的PYTHON路径
    pause
    exit /b 1
)

echo [INFO] 使用Python: %PYTHON%
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"

echo =====================================================
echo 步骤 1/4: 获取API数据
echo =====================================================
echo.
"%PYTHON%" scripts\fetch_data.py
if %errorlevel% neq 0 (
    echo [ERROR] 数据获取失败
    pause
    exit /b 1
)
echo.
echo [OK] 数据获取完成
echo.

echo =====================================================
echo 步骤 2/4: 创建特征
echo =====================================================
echo.
"%PYTHON%" features\create_features.py
if %errorlevel% neq 0 (
    echo [ERROR] 特征创建失败
    pause
    exit /b 1
)
echo.
echo [OK] 特征创建完成
echo.

echo =====================================================
echo 步骤 3/4: 训练模型
echo =====================================================
echo.
"%PYTHON%" models\train_models.py
if %errorlevel% neq 0 (
    echo [ERROR] 模型训练失败
    pause
    exit /b 1
)
echo.
echo [OK] 模型训练完成
echo.

echo =====================================================
echo 步骤 4/4: 启动Web应用
echo =====================================================
echo.
echo 正在启动Streamlit应用...
echo 浏览器将自动打开 http://localhost:8501
echo.
echo 按 Ctrl+C 停止应用
echo.
"%PYTHON%" -m streamlit run app_api.py

pause

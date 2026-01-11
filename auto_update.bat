@echo off
REM =====================================================
REM Automatic Data Update Script
REM Fetches data, creates features, and trains models
REM =====================================================

REM Set Python path (modify if needed)
set PYTHON=D:\anaconda\python.exe

REM Get current directory
cd /d "%~dp0"

REM Create log directory
if not exist "logs" mkdir logs

REM Set log file with timestamp
set LOG_FILE=logs\auto_update_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

REM Start logging
echo ===================================================== >> "%LOG_FILE%" 2>&1
echo Automatic Update Started >> "%LOG_FILE%" 2>&1
echo Time: %date% %time% >> "%LOG_FILE%" 2>&1
echo ===================================================== >> "%LOG_FILE%" 2>&1
echo. >> "%LOG_FILE%" 2>&1

REM Step 1: Fetch latest data
echo [%time%] Step 1/3: Fetching latest data from ENTSO-E API... >> "%LOG_FILE%" 2>&1
"%PYTHON%" scripts\fetch_data.py >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [%time%] ERROR: Data fetch failed >> "%LOG_FILE%" 2>&1
    exit /b 1
)
echo [%time%] SUCCESS: Data fetched >> "%LOG_FILE%" 2>&1
echo. >> "%LOG_FILE%" 2>&1

REM Step 2: Create features
echo [%time%] Step 2/3: Creating features... >> "%LOG_FILE%" 2>&1
"%PYTHON%" features\create_features.py >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [%time%] ERROR: Feature creation failed >> "%LOG_FILE%" 2>&1
    exit /b 1
)
echo [%time%] SUCCESS: Features created >> "%LOG_FILE%" 2>&1
echo. >> "%LOG_FILE%" 2>&1

REM Step 3: Train models
echo [%time%] Step 3/3: Training models... >> "%LOG_FILE%" 2>&1
"%PYTHON%" models\train_models.py >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [%time%] ERROR: Model training failed >> "%LOG_FILE%" 2>&1
    exit /b 1
)
echo [%time%] SUCCESS: Models trained >> "%LOG_FILE%" 2>&1
echo. >> "%LOG_FILE%" 2>&1

REM Completion
echo ===================================================== >> "%LOG_FILE%" 2>&1
echo Automatic Update Completed Successfully >> "%LOG_FILE%" 2>&1
echo Time: %date% %time% >> "%LOG_FILE%" 2>&1
echo ===================================================== >> "%LOG_FILE%" 2>&1

exit /b 0

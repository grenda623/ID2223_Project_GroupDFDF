@echo off
REM =====================================================
REM Setup Daily Auto-Update Task
REM Runs at 08:00 Stockholm time (CET/CEST)
REM =====================================================

echo =====================================================
echo Setting up Daily Auto-Update Task
echo =====================================================
echo.

REM Set task name
set TASK_NAME=ElectricityPriceAutoUpdate

REM Set script path
set SCRIPT_PATH=%~dp0auto_update.bat

REM Set time (08:00 Stockholm time)
set START_TIME=08:00

echo Task Name: %TASK_NAME%
echo Script Path: %SCRIPT_PATH%
echo Start Time: %START_TIME% (Stockholm time)
echo.

REM Delete existing task if it exists
schtasks /query /tn "%TASK_NAME%" >nul 2>&1
if %errorlevel% == 0 (
    echo Removing existing task...
    schtasks /delete /tn "%TASK_NAME%" /f
    echo.
)

REM Create new scheduled task
echo Creating new scheduled task...
schtasks /create /tn "%TASK_NAME%" /tr "\"%SCRIPT_PATH%\"" /sc daily /st %START_TIME% /f

if %errorlevel% == 0 (
    echo.
    echo =====================================================
    echo SUCCESS! Daily auto-update task created!
    echo =====================================================
    echo.
    echo Task Details:
    echo - Task Name: %TASK_NAME%
    echo - Runs Daily at: %START_TIME% Stockholm time
    echo - Script: %SCRIPT_PATH%
    echo.
    echo The task will automatically:
    echo   1. Fetch latest data from ENTSO-E API
    echo   2. Create features
    echo   3. Train models
    echo.
    echo To view the task:
    echo   schtasks /query /tn "%TASK_NAME%" /fo LIST /v
    echo.
    echo To delete the task:
    echo   schtasks /delete /tn "%TASK_NAME%" /f
    echo.
) else (
    echo.
    echo =====================================================
    echo ERROR! Failed to create scheduled task
    echo =====================================================
    echo.
    echo Please run this script as Administrator
    echo.
)

pause

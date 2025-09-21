@echo off
REM Face Validator API Startup Script
REM This script starts the Face Validator API server

echo.
echo ========================================
echo   FACE VALIDATOR API STARTUP
echo ========================================
echo.

REM Change to the correct directory
cd /d "C:\Development\facevalidator\facevalidator"

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please ensure the .venv folder exists in the current directory.
    pause
    exit /b 1
)

echo Starting Face Validator API...
echo.
echo The API will be available at:
echo   http://localhost:5000
echo.
echo Available endpoints:
echo   GET  /                    - API information
echo   GET  /health              - Health check
echo   POST /validate            - Complete validation
echo   POST /validate/nsfw       - NSFW detection
echo   POST /validate/deepfake   - Deepfake detection
echo   POST /validate/pose       - Pose validation
echo   POST /validate/celebrity  - Celebrity detection
echo   GET  /duplicates/stats    - Duplicate statistics (requires auth)
echo.
echo Press Ctrl+C to stop the server
echo ========================================

REM Start the server
".venv\Scripts\python.exe" server.py

echo.
echo API server stopped.
pause
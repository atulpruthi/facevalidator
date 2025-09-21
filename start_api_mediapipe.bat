@echo off
REM Face Validator API Startup Script - WITH MEDIAPIPE SUPPORT
REM This script starts the Face Validator API server using Python 3.12 for MediaPipe support

echo.
echo ========================================
echo   FACE VALIDATOR API STARTUP (MediaPipe)
echo ========================================
echo.

REM Change to the correct directory
cd /d "C:\Development\facevalidator\facevalidator"

REM Check if Python 3.12 virtual environment exists
if not exist ".venv312\Scripts\python.exe" (
    echo ERROR: Python 3.12 virtual environment not found!
    echo Please ensure the .venv312 folder exists in the current directory.
    pause
    exit /b 1
)

echo Starting Face Validator API with MediaPipe support...
echo.
echo The API will be available at:
echo   http://localhost:5000
echo.
echo Features enabled:
echo   ✅ NSFW/Weapons Detection
echo   ✅ Deepfake Detection  
echo   ✅ Advanced Pose Analysis (MediaPipe)
echo   ✅ Celebrity Detection
echo   ✅ Duplicate Detection
echo.
echo Available endpoints:
echo   GET  /                    - API information
echo   GET  /health              - Health check
echo   POST /validate            - Complete validation
echo   POST /validate/nsfw       - NSFW detection
echo   POST /validate/deepfake   - Deepfake detection
echo   POST /validate/pose       - Advanced pose validation
echo   POST /validate/celebrity  - Celebrity detection
echo   GET  /duplicates/stats    - Duplicate statistics (requires auth)
echo.
echo Press Ctrl+C to stop the server
echo ========================================

REM Start the server with Python 3.12
".venv312\Scripts\python.exe" server.py

echo.
echo API server stopped.
pause
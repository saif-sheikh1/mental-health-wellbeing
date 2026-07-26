@echo off
echo =========================================
echo   Starting MindSense AI Web Server...
echo =========================================
echo.
echo Launching Uvicorn on localhost:8000
python -m uvicorn web_app:app --host 0.0.0.0 --port 8000
if %ERRORLEVEL% neq 0 (
    echo.
    echo Trying fallback 'py' command...
    py -m uvicorn web_app:app --host 0.0.0.0 --port 8000
)
pause

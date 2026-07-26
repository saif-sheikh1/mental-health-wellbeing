@echo off
:: Use Python 3.10 system install (has TF freshly installed via pip)
set PYTHON=C:\Users\Star\AppData\Local\Programs\Python\Python310\python.exe

echo =========================================
echo   MindSense AI - Starting on port 9000
echo =========================================
echo.
"%PYTHON%" --version
echo.
echo Open your browser at: http://localhost:9000
echo.
"%PYTHON%" -m uvicorn web_app:app --host 0.0.0.0 --port 9000

pause

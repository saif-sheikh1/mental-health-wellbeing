@echo off
setlocal

set PYTHON=C:\Users\Star\PyCharmMiscProject\.venv\Scripts\python.exe

echo =========================================
echo   MindSense AI - Installing Packages
echo =========================================
"%PYTHON%" -m pip install fastapi "uvicorn[standard]" supabase pyserial opencv-python-headless numpy scikit-learn jinja2 python-multipart --quiet
echo Done.

echo.
echo =========================================
echo   Starting Server on http://localhost:9000
echo =========================================
echo.
"%PYTHON%" -m uvicorn web_app:app --host 0.0.0.0 --port 9000

pause

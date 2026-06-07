@echo off
setlocal

set VENV_PY="D:\fyp\first model\.venv\Scripts\python.exe"

echo Testing python...
%VENV_PY% --version
echo Exit: %ERRORLEVEL%
pause

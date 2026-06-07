@echo off
set VENV=D:\fyp\prediction model\.venv\Scripts\python.exe
echo Testing venv Python...
"%VENV%" --version
echo.
echo Testing TensorFlow...
"%VENV%" -c "import tensorflow; print('TF OK:', tensorflow.__version__)"
echo ExitCode: %ERRORLEVEL%
pause

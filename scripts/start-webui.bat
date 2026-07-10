@echo off
setlocal

cd /d "%~dp0.."

set "ROOT=%CD%"
set "PYTHON_EXE=%ROOT%\.venv312\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python runtime not found: "%PYTHON_EXE%"
  echo Run start.bat install first.
  exit /b 1
)

echo Starting web UI on http://127.0.0.1:8765
"%PYTHON_EXE%" webapp.py

endlocal

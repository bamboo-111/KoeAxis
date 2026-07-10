@echo off
setlocal

cd /d "%~dp0.."

set "ROOT=%CD%"
set "VENV_DIR=%ROOT%\.venv312"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python launcher "py" not found.
  echo Install Python 3.12 first, then rerun this script.
  exit /b 1
)

echo [1/5] Creating virtual environment: .venv312
if not exist "%PYTHON_EXE%" (
  py -3.12 -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create Python 3.12 virtual environment.
    exit /b 1
  )
)

echo [2/5] Upgrading pip
"%PYTHON_EXE%" -m pip install -U pip setuptools wheel
if errorlevel 1 exit /b 1

echo [3/5] Installing dependencies
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [4/5] Checking ffmpeg
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [WARN] ffmpeg not found in PATH. prepare stage will fail until ffmpeg is installed.
) else (
  ffmpeg -version | more +0 >nul
  echo [OK] ffmpeg found
)

echo [5/5] Done
echo.
echo Web UI:
echo   start.bat web
echo.
echo CLI shell:
echo   start.bat cli
echo.
echo One-shot pipeline example:
echo   .venv312\Scripts\python.exe main.py run --media samples\test.mp3 --workdir .\work-test --with-align --with-split --with-normalize

endlocal

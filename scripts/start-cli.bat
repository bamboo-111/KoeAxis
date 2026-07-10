@echo off
setlocal

cd /d "%~dp0.."

set "ROOT=%CD%"
set "ACTIVATE=%ROOT%\.venv312\Scripts\activate.bat"

if not exist "%ACTIVATE%" (
  echo [ERROR] Virtual environment not found.
  echo Run start.bat install first.
  exit /b 1
)

call "%ACTIVATE%"
echo Activated .venv312
cmd /k

endlocal

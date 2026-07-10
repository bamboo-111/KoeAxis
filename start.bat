@echo off
setlocal

set "ROOT=%~dp0"
set "CMD=%~1"

if "%CMD%"=="" set "CMD=web"

if /I "%CMD%"=="web" (
  call "%ROOT%scripts\start-webui.bat"
  exit /b %ERRORLEVEL%
)

if /I "%CMD%"=="cli" (
  call "%ROOT%scripts\start-cli.bat"
  exit /b %ERRORLEVEL%
)

if /I "%CMD%"=="install" (
  call "%ROOT%scripts\install.bat"
  exit /b %ERRORLEVEL%
)

echo Usage:
echo   start.bat web
echo   start.bat cli
echo   start.bat install
exit /b 1

@echo off
setlocal

cd /d "%~dp0.."

set "ROOT_DIR=%CD%"
set "DIST_DIR=%ROOT_DIR%\dist"
set "STAGE_DIR=%DIST_DIR%\qwen3-asr"
set "ZIP_PATH=%DIST_DIR%\qwen3-asr-package.zip"

if exist "%STAGE_DIR%" rmdir /s /q "%STAGE_DIR%"
if exist "%ZIP_PATH%" del /f /q "%ZIP_PATH%"
mkdir "%STAGE_DIR%"

for %%F in (
  main.py
  README.md
  requirements.txt
  start.bat
  webapp.py
  .gitignore
) do copy "%%F" "%STAGE_DIR%\" >nul

xcopy "qwen_asr" "%STAGE_DIR%\qwen_asr\" /E /I /Y >nul
xcopy "optimizer" "%STAGE_DIR%\optimizer\" /E /I /Y >nul
xcopy "scripts" "%STAGE_DIR%\scripts\" /E /I /Y >nul
if exist "docs" xcopy "docs" "%STAGE_DIR%\docs\" /E /I /Y >nul

for %%F in (
  "%STAGE_DIR%\qwen_asr\web\templates\index.html"
  "%STAGE_DIR%\optimizer\text_utils.py"
  "%STAGE_DIR%\optimizer\asr_cleanup.py"
  "%STAGE_DIR%\optimizer\text_metrics.py"
  "%STAGE_DIR%\optimizer\fixed_terms.py"
  "%STAGE_DIR%\optimizer\llm_config.py"
) do if not exist "%%~F" (
  echo [ERROR] Missing packaged file: %%~F
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Path '%STAGE_DIR%\*' -DestinationPath '%ZIP_PATH%' -Force"

if errorlevel 1 (
  echo [ERROR] Failed to build package zip.
  exit /b 1
)

echo Package created:
echo   %ZIP_PATH%

endlocal

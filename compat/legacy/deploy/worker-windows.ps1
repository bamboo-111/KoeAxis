# Example: run a KoeAxis CI GPU worker on Windows. Adjust paths / ids / server URL to your setup.
# Usage (PowerShell, from anywhere):
#   $env:KOEAXIS_WORKER_TOKEN = "kxw_..."     # created with: python -m qwen_asr.ci create-token --kind worker --name gpu-1
#   $env:GEMINI_API_KEY = "..."                # text LLM (default gemini-3.8-flash); keys stay on the worker
#   $env:GEMINI_BASE_URL = "https://relay.example/"  # optional relay
#   $env:MIMO_API_KEY = "..."                  # MiMo audio proofread
#   .\deploy\worker-windows.ps1 -Server https://ci.example.com
# To keep it running after logoff, register it with Task Scheduler or NSSM (left to the deployer).
param(
    [string]$Server = $env:KOEAXIS_SERVER,
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\.."),
    [string]$Python = "",
    [string]$WorkerId = $env:COMPUTERNAME,
    [string]$DataDir = "ci-worker",
    [string]$ModelCacheDir = ".model-cache",
    [int]$MaxRuns = 2,
    [string[]]$Preset = @()
)
$ErrorActionPreference = "Stop"
if (-not $Server) { throw "set -Server or `$env:KOEAXIS_SERVER" }
if (-not $env:KOEAXIS_WORKER_TOKEN) { throw "set `$env:KOEAXIS_WORKER_TOKEN" }
if (-not $Python) { $Python = Join-Path $ProjectRoot ".venv312\Scripts\python.exe" }

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
Set-Location $ProjectRoot

$argsList = @("-B", "-m", "qwen_asr.ci", "worker", "--server", $Server, "--id", $WorkerId,
              "--data-dir", $DataDir, "--model-cache-dir", $ModelCacheDir, "--max-runs", $MaxRuns)
foreach ($p in $Preset) { $argsList += @("--preset", $p) }

# Restart on crash / network loss; Ctrl+C to stop.
while ($true) {
    & $Python @argsList
    Write-Warning "worker exited with code $LASTEXITCODE; restarting in 15s"
    Start-Sleep -Seconds 15
}

param(
    [switch]$SkipRuff
)

$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv312\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked -Name "compileall" -Command { & $python -m compileall qwen_asr optimizer tests -q }
Invoke-Checked -Name "pytest" -Command { & $python -m pytest -q }

if (-not $SkipRuff) {
    Invoke-Checked -Name "ruff" -Command { & $python -m ruff check qwen_asr optimizer tests }
}

Invoke-Checked -Name "git diff --check" -Command { git diff --check }

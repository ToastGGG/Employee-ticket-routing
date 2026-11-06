param(
    [string]$PythonPath = "",
    [switch]$SkipRuff
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot

if (-not $PythonPath) {
    $Candidate = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $Candidate) {
        $PythonPath = $Candidate
    }
    else {
        $PythonPath = (Get-Command python -ErrorAction Stop).Source
    }
}

Write-Host "[QUALITY] Python: $PythonPath" -ForegroundColor Cyan
Write-Host "[QUALITY] Compile all Python modules" -ForegroundColor Cyan
& $PythonPath -m compileall -q "src" "tests"
if ($LASTEXITCODE -ne 0) {
    throw "Python compile check failed."
}

Write-Host "[QUALITY] Run unit tests" -ForegroundColor Cyan
& $PythonPath -m unittest discover -s tests -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) {
    throw "Unit tests failed."
}

if (-not $SkipRuff) {
    $Ruff = Get-Command ruff -ErrorAction SilentlyContinue
    if ($Ruff) {
        Write-Host "[QUALITY] Run Ruff" -ForegroundColor Cyan
        & $Ruff.Source check "src" "tests"
        if ($LASTEXITCODE -ne 0) {
            throw "Ruff check failed."
        }
    }
    else {
        Write-Host "[QUALITY] Ruff not installed; static lint skipped." -ForegroundColor Yellow
    }
}

Write-Host "[QUALITY] All checks passed." -ForegroundColor Green

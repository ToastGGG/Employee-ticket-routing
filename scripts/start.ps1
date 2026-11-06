param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload,
    [switch]$CheckOnly,
    [switch]$SkipInstall,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Continue"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$RequirementsPath = Join-Path $RepoRoot "requirements.txt"
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsStamp = Join-Path $VenvDir "requirements.sha256"

function Write-Step {
    param([string]$Message)
    Write-Host "[START] $Message" -ForegroundColor Cyan
}

function Get-BasePython {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        return @{
            Executable = $PythonCommand.Source
            Prefix = @()
        }
    }

    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        return @{
            Executable = $PyLauncher.Source
            Prefix = @("-3")
        }
    }

    throw "Python was not found. Install Python 3.11 or newer."
}

function Invoke-BasePython {
    param(
        [hashtable]$BasePython,
        [string[]]$Arguments
    )

    & $BasePython.Executable @($BasePython.Prefix) @Arguments
}

function Invoke-NativeCommand {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [switch]$Quiet
    )

    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($Quiet) {
            & $FilePath @Arguments *> $null
        }
        else {
            & $FilePath @Arguments | Out-Host
        }
        $NativeExitCode = $LASTEXITCODE
        return [int]$NativeExitCode
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }
}

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($RawLine in Get-Content -LiteralPath $Path -Encoding utf8) {
        $Line = $RawLine.Trim()
        if (-not $Line -or $Line.StartsWith("#")) {
            continue
        }
        if ($Line -notmatch '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
            continue
        }

        $Name = $Matches[1]
        $Value = $Matches[2].Trim()
        if (
            $Value.Length -ge 2 -and
            (($Value.StartsWith('"') -and $Value.EndsWith('"')) -or
             ($Value.StartsWith("'") -and $Value.EndsWith("'")))
        ) {
            $Value = $Value.Substring(1, $Value.Length - 2)
        }
        Set-Item -Path "Env:$Name" -Value $Value
    }
}

Set-Location -LiteralPath $RepoRoot
$env:PYTHONUTF8 = "1"
Write-Step "Project root: $RepoRoot"

if (-not (Test-Path -LiteralPath $RequirementsPath)) {
    throw "Missing dependency file: $RequirementsPath"
}

if ($PythonPath) {
    $Python = (Resolve-Path -LiteralPath $PythonPath).Path
    Write-Step "Using Python: $Python"
}
else {
    $NeedCreateVenv = -not (Test-Path -LiteralPath $VenvPython)

    if (-not $NeedCreateVenv) {
        $ExitCode = Invoke-NativeCommand -FilePath $VenvPython -Arguments @(
            "-c", "import sys; print(sys.version)"
        ) -Quiet
        if ($ExitCode -ne 0) {
            $NeedCreateVenv = $true
        }
        else {
            $ExitCode = Invoke-NativeCommand -FilePath $VenvPython -Arguments @(
                "-m", "pip", "--version"
            ) -Quiet
            if ($ExitCode -ne 0) {
                $NeedCreateVenv = $true
            }
        }
    }

    if ($NeedCreateVenv) {
        if (Test-Path -LiteralPath $VenvDir) {
            $BrokenVenv = "$VenvDir.broken-$(Get-Date -Format 'yyyyMMddHHmmss')"
            Write-Step "Preserving incomplete .venv as $(Split-Path $BrokenVenv -Leaf)"
            Move-Item -LiteralPath $VenvDir -Destination $BrokenVenv
        }

        Write-Step "Creating virtual environment without ensurepip"
        $BasePython = Get-BasePython
        Invoke-BasePython -BasePython $BasePython -Arguments @(
            "-m", "venv", "--without-pip", "--system-site-packages", $VenvDir
        )
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
            throw "Failed to create .venv."
        }
    }

    $Python = $VenvPython

    $ExitCode = Invoke-NativeCommand -FilePath $Python -Arguments @(
        "-c", "import sys; print(sys.version)"
    ) -Quiet
    if ($ExitCode -ne 0) {
        throw "The virtual environment Python cannot run: $Python"
    }

    $ExitCode = Invoke-NativeCommand -FilePath $Python -Arguments @(
        "-m", "pip", "--version"
    ) -Quiet
    if ($ExitCode -ne 0) {
        throw "pip is unavailable. Install pip for the base Python and run start.bat again."
    }
}

Import-DotEnv -Path (Join-Path $RepoRoot ".env")

$DependenciesReady = $false
$ExitCode = Invoke-NativeCommand -FilePath $Python -Arguments @(
    "-c",
    "import fastapi, uvicorn, aiosqlite, httpx, jwt, numpy, faiss, argon2, jinja2, multipart, pypdf"
) -Quiet
if ($ExitCode -eq 0) {
    $DependenciesReady = $true
}

$RequirementsHash = (Get-FileHash -LiteralPath $RequirementsPath -Algorithm SHA256).Hash
$InstalledHash = if (Test-Path -LiteralPath $RequirementsStamp) {
    (Get-Content -LiteralPath $RequirementsStamp -Raw).Trim()
}
else {
    ""
}

if (-not $SkipInstall -and (-not $DependenciesReady -or $InstalledHash -ne $RequirementsHash)) {
    Write-Step "Installing or updating project dependencies"
    $ExitCode = Invoke-NativeCommand -FilePath $Python -Arguments @(
        "-m", "pip", "install", "--disable-pip-version-check",
        "-r", $RequirementsPath
    )
    if ($ExitCode -ne 0) {
        throw "Dependency installation failed. Check network, proxy, and Python version."
    }
    Set-Content -LiteralPath $RequirementsStamp -Value $RequirementsHash -Encoding ascii
    $DependenciesReady = $true
}

if (-not $DependenciesReady) {
    throw "Dependencies are incomplete. Run again without -SkipInstall."
}

Write-Step "Checking application imports and static resources"
$ExitCode = Invoke-NativeCommand -FilePath $Python -Arguments @(
    "-c",
    "from src.employee_ticket_routing.main import app, STATIC_DIR, TEMPLATE_DIR; assert STATIC_DIR.exists(); assert TEMPLATE_DIR.exists(); print('APP_IMPORT_OK')"
)
if ($ExitCode -ne 0) {
    throw "Application import failed. See the error above."
}

if ($CheckOnly) {
    Write-Step "Check passed. Service was not started."
    exit 0
}

$UvicornArgs = @(
    "-m", "uvicorn",
    "src.employee_ticket_routing.main:app",
    "--host", $BindHost,
    "--port", "$Port"
)
if (-not $NoReload) {
    $UvicornArgs += "--reload"
}

$LoginUrl = "http://$BindHost`:$Port/login"
Write-Host ""
Write-Host "Starting service: $LoginUrl" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the service." -ForegroundColor DarkGray
Write-Host ""

& $Python @UvicornArgs
exit $LASTEXITCODE

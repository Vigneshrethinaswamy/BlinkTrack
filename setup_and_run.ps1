# BlinkTrack one-command setup, test, and launch (Windows PowerShell).
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Find a working Python 3.11 interpreter. The Windows Python Launcher (py.exe)
# is optional; some installations expose only python.exe on PATH.
$PythonExe = $null
$PythonPrefixArgs = @()
$candidates = @()

$pyCommand = Get-Command py -ErrorAction SilentlyContinue
if ($pyCommand) {
    $candidates += [PSCustomObject]@{ Exe = $pyCommand.Source; Prefix = @("-3.11") }
}

foreach ($commandName in @("python", "python3")) {
    $command = Get-Command $commandName -ErrorAction SilentlyContinue
    if ($command) {
        $candidates += [PSCustomObject]@{ Exe = $command.Source; Prefix = @() }
    }
}

# Common per-user and machine-wide Python 3.11 install locations.
$commonPaths = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
    (Join-Path $env:ProgramFiles "Python311\python.exe"),
    "C:\Python311\python.exe"
)
foreach ($path in $commonPaths) {
    if ($path -and (Test-Path $path)) {
        $candidates += [PSCustomObject]@{ Exe = $path; Prefix = @() }
    }
}

foreach ($candidate in $candidates) {
    try {
        $version = & $candidate.Exe @($candidate.Prefix) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -eq 0 -and "$version".Trim() -eq "3.11") {
            $PythonExe = $candidate.Exe
            $PythonPrefixArgs = @($candidate.Prefix)
            break
        }
    }
    catch {
        # Try the next discovered interpreter.
    }
}

if (-not $PythonExe) {
    Write-Host "Could not find Python 3.11." -ForegroundColor Red
    Write-Host "Checked: py -3.11, python, python3, and common Python311 install paths."
    Write-Host "Run 'python --version' and 'where.exe python' to inspect your installation."
    throw "Install Python 3.11 or add its python.exe to PATH, then rerun this script."
}

$pythonLabel = & $PythonExe @PythonPrefixArgs --version
Write-Host "Using $pythonLabel ($PythonExe)" -ForegroundColor Green

$VenvPath = Join-Path $PSScriptRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating project virtual environment..." -ForegroundColor Cyan
    & $PythonExe @PythonPrefixArgs -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

Write-Host "Installing project and test dependencies..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $VenvPython -m pip install -r (Join-Path $PSScriptRoot "requirements-dev.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

Write-Host "Running unit tests..." -ForegroundColor Cyan
& $VenvPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed. Fix the reported issue before launching the webcam app." }

Write-Host "Starting BlinkTrack. Press Q/Esc to quit, or R to reset the count." -ForegroundColor Green
& $VenvPython (Join-Path $PSScriptRoot "main.py")
if ($LASTEXITCODE -ne 0) { throw "BlinkTrack exited with an error." }

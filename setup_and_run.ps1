# BlinkTrack one-command setup, test, and launch (Windows PowerShell).
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $PythonLauncher) {
    throw "Python Launcher (py.exe) was not found. Install Python 3.11 from python.org and enable the launcher."
}

& py -3.11 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 is required. Install it, then run this script again."
}

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating project virtual environment..." -ForegroundColor Cyan
    & py -3.11 -m venv (Join-Path $PSScriptRoot ".venv")
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

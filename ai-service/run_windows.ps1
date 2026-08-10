<#
 .SYNOPSIS
   Windows launch script for the Vigilant Eye AI service.
 .DESCRIPTION
   Verifies the environment, uses .venv if present, warns clearly when .env
   is missing, and starts the existing FastAPI service via run.py.
   Does NOT install packages or request admin privileges.
#>

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

Write-Host "=== Vigilant Eye AI Service (Windows) ===" -ForegroundColor Cyan

# --- 1. Verify Python is available ---
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
} else {
    Write-Host "ERROR: Python is not installed or not on PATH." -ForegroundColor Red
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/ and retry."
    exit 1
}

$pyVersion = & $pythonCmd --version 2>&1
Write-Host "Found: $pyVersion"

# --- 2. Activate virtual environment if present ---
$venvActivate = Join-Path $scriptDir ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment (.venv)..." -ForegroundColor Green
    & $venvActivate
    $pythonCmd = "python"
} else {
    Write-Host "WARNING: No .venv found. Using system Python." -ForegroundColor Yellow
    Write-Host "Create one with: python -m venv .venv && .venv\Scripts\Activate.ps1"
    Write-Host "Then install deps:  pip install -r requirements.txt"
}

# --- 3. Warn if .env is missing ---
$envFile = Join-Path $scriptDir ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "WARNING: .env file not found." -ForegroundColor Yellow
    Write-Host "Copy .env.example to .env and fill in SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,"
    Write-Host "and AI_SERVICE_KEY before running the service for real."
    Write-Host "The service will start but will report configuration problems."
}

# --- 4. Start the service ---
Write-Host "Starting FastAPI service via run.py..." -ForegroundColor Green
& $pythonCmd run.py

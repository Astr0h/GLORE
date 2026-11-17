# GLORE one-time setup (Windows).
# Easiest: double-click "Setup GLORE.bat" in the GLORE folder.
# Safe to run again - it skips what's already done.
# Everything shown here is also saved to setup.log in the GLORE folder.

$ErrorActionPreference = "Continue"   # native tools write warnings to stderr; we check exit codes instead
$Root = Split-Path -Parent $PSScriptRoot
$Engine = Join-Path $Root "engine"
$Venv = Join-Path $Engine ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Log = Join-Path $Root "setup.log"

try { Start-Transcript -Path $Log -Force | Out-Null } catch {}

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Finish($code) {
    try { Stop-Transcript | Out-Null } catch {}
    Write-Host ""
    Read-Host "Press Enter to close"
    exit $code
}
function Fail($msg) {
    Write-Host "`n[X] $msg" -ForegroundColor Red
    Write-Host "    The full log is in setup.log - send it to Claude." -ForegroundColor Red
    Finish 1
}
function Has($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Write-Host "GLORE setup" -ForegroundColor White
Write-Host "Folder: $Root"

# 1. Python ------------------------------------------------------------
Step "Finding Python (3.10 - 3.12 recommended)"
$base = $null
if (Has "py") {
    foreach ($v in @("3.12", "3.11", "3.10", "3.13")) {
        & py "-$v" -c "import sys" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $base = @("py", "-$v"); Ok "Using Python $v (py launcher)"; break }
    }
}
if (-not $base -and (Has "python")) {
    $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -eq 0 -and $ver) { $base = @("python"); Ok "Using python $ver" }
}
if (-not $base) { Fail "Python not found. Install Python 3.12 from python.org (tick 'Add python.exe to PATH'), then run this again." }

# 2. Virtual environment -----------------------------------------------
Step "Creating engine virtual environment"
if (-not (Test-Path $Py)) {
    if ($base.Count -gt 1) { & $base[0] $base[1] -m venv $Venv } else { & $base[0] -m venv $Venv }
    if (-not (Test-Path $Py)) { Fail "Couldn't create the virtual environment at $Venv" }
    Ok "Created"
} else { Ok "Already exists" }
& $Py -m pip install --upgrade pip --quiet --disable-pip-version-check

# 3. PyTorch with CUDA ---------------------------------------------------
Step "Installing PyTorch with CUDA (large download, ~2.5 GB, first time only)"
& $Py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    & $Py -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed. Check your internet connection and run this again." }
    Ok "Installed"
} else { Ok "Already installed with CUDA" }

# 4. Engine packages ---------------------------------------------------
Step "Installing GLORE engine packages"
& $Py -m pip install -r (Join-Path $Engine "requirements.txt") --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Fail "Package install failed." }
& $Py -m pip install --upgrade yt-dlp --quiet --disable-pip-version-check 2>&1 | Out-Null
Ok "Done"

# 5. Check the GPU -----------------------------------------------------
Step "Checking GPU"
$gpu = & $Py -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" 2>$null
if (-not $gpu -or $gpu -eq "NONE") {
    Warn "CUDA isn't available - GLORE will run on CPU (slow). Update your NVIDIA driver and run this again."
} else { Ok "GPU: $gpu" }

# 6. Pre-download the vocal separation model ---------------------------
Step "Downloading vocal separation model (htdemucs_ft, ~300 MB)"
& $Py -c "from demucs.pretrained import get_model; get_model('htdemucs_ft'); print('ready')"
if ($LASTEXITCODE -ne 0) { Warn "Model download failed - it will retry the first time you use Extract vocal." } else { Ok "Ready" }

# 7. Desktop app -------------------------------------------------------
Step "Installing desktop app (Electron)"
if (-not (Has "node")) { Fail "Node.js not found. Install the LTS version from nodejs.org, then run this again." }
$nodeVer = & node -v
Ok "Node $nodeVer"
Push-Location (Join-Path $Root "app")
& npm.cmd install --no-audit --no-fund
$npmExit = $LASTEXITCODE
Pop-Location
if ($npmExit -ne 0 -or -not (Test-Path (Join-Path $Root "app\node_modules\electron"))) { Fail "npm install failed." }
Ok "Installed"

# 8. Self-test ---------------------------------------------------------
Step "Running engine self-test"
Push-Location $Engine
& $Py -m pip install pytest httpx --quiet --disable-pip-version-check 2>&1 | Out-Null
& $Py -m pytest -q
$testExit = $LASTEXITCODE
Pop-Location
if ($testExit -ne 0) { Warn "Some tests failed - send Claude setup.log." } else { Ok "All tests passed" }

Write-Host "`nGLORE is ready. Double-click 'Start GLORE.bat' to launch." -ForegroundColor Green
Finish 0

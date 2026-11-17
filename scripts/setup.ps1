# GLORE one-time setup (Windows).
# Run from the repo folder:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Safe to run again - it skips what's already done.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Engine = Join-Path $Root "engine"
$Venv = Join-Path $Engine ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "`n[X] $msg" -ForegroundColor Red; exit 1 }

# 1. Python ------------------------------------------------------------
Step "Finding Python (3.10 - 3.12 recommended)"
$base = $null
foreach ($v in @("3.12", "3.11", "3.10", "3.13")) {
    try { $out = & py "-$v" -c "import sys; print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $out) { $base = @("py", "-$v"); Ok "Using Python $v ($out)"; break } } catch {}
}
if (-not $base) {
    try { $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"; $base = @("python"); Ok "Using python $ver" }
    catch { Fail "Python not found. Install Python 3.12 from python.org (tick 'Add to PATH'), then run this again." }
}

# 2. Virtual environment -----------------------------------------------
Step "Creating engine virtual environment"
if (-not (Test-Path $Py)) {
    if ($base.Count -gt 1) { & $base[0] $base[1] -m venv $Venv } else { & $base[0] -m venv $Venv }
    if (-not (Test-Path $Py)) { Fail "Couldn't create the virtual environment at $Venv" }
    Ok "Created $Venv"
} else { Ok "Already exists" }
& $Py -m pip install --upgrade pip --quiet

# 3. PyTorch with CUDA (for the RTX 4090) -------------------------------
Step "Installing PyTorch with CUDA (large download, ~2.5 GB, first time only)"
$hasCuda = & $Py -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($hasCuda -ne "True") {
    & $Py -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed. Check your internet connection and run again." }
} else { Ok "PyTorch + CUDA already installed" }

# 4. Engine packages ---------------------------------------------------
Step "Installing GLORE engine packages"
& $Py -m pip install -r (Join-Path $Engine "requirements.txt")
if ($LASTEXITCODE -ne 0) { Fail "Package install failed - send Claude the red error text above." }

# yt-dlp needs to stay current - YouTube changes often
& $Py -m pip install --upgrade yt-dlp --quiet

# 5. Check the GPU -----------------------------------------------------
Step "Checking GPU"
$gpu = & $Py -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
if ($gpu -eq "NONE") {
    Write-Host "    CUDA isn't available - GLORE will run on CPU (slow). Update your NVIDIA driver and run this again." -ForegroundColor Yellow
} else { Ok "GPU: $gpu" }

# 6. Pre-download the vocal separation model ---------------------------
Step "Downloading vocal separation model (htdemucs_ft, ~300 MB)"
& $Py -c "from demucs.pretrained import get_model; get_model('htdemucs_ft'); print('ready')"
if ($LASTEXITCODE -ne 0) { Write-Host "    Model download failed - it will retry the first time you use Extract vocal." -ForegroundColor Yellow }

# 7. Desktop app -------------------------------------------------------
Step "Installing desktop app (Electron)"
try { $node = & node -v; Ok "Node $node" } catch { Fail "Node.js not found. Install the LTS version from nodejs.org, then run this again." }
Push-Location (Join-Path $Root "app")
npm install --no-audit --no-fund
$npmExit = $LASTEXITCODE
Pop-Location
if ($npmExit -ne 0) { Fail "npm install failed." }

# 8. Self-test ---------------------------------------------------------
Step "Running engine self-test"
Push-Location $Engine
& $Py -m pip install pytest httpx --quiet
& $Py -m pytest tests -q
$testExit = $LASTEXITCODE
Pop-Location
if ($testExit -ne 0) { Write-Host "    Some tests failed - send Claude the output above." -ForegroundColor Yellow } else { Ok "All tests passed" }

Write-Host "`nGLORE is ready. Double-click 'Start GLORE.bat' to launch." -ForegroundColor Green

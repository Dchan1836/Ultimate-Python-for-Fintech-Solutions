$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "c:\Python312\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

if (-not (Test-Path "$root\frontend\node_modules")) {
  Push-Location "$root\frontend"
  npm install
  Pop-Location
}

Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "& '$python' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -WorkingDirectory "$root\frontend" -ArgumentList "-NoExit", "-Command", "npm run dev"

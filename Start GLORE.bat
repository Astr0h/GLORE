@echo off
cd /d "%~dp0app"
if not exist node_modules (
  echo GLORE isn't set up yet. Run:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
  pause
  exit /b 1
)
npm start

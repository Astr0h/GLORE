@echo off
cd /d "%~dp0app"
if not exist node_modules\electron (
  echo GLORE isn't set up yet - running setup now...
  call "%~dp0Setup GLORE.bat"
  if not exist node_modules\electron exit /b 1
)
call npm.cmd start

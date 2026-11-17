@echo off
rem One-click setup. -NoProfile skips your own PowerShell profile, so its errors can't get in the way.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"

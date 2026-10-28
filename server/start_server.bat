@echo off
REM === Gineung AI server: start (keep this window open) ===
cd /d "%~dp0"
title Gineung AI server
if "%VENV%"=="" set VENV=%USERPROFILE%\gineung-venv
call "%VENV%\Scripts\activate"
set PYTHONUTF8=1
:loop
python run.py
echo Server stopped. Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop

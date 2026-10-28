@echo off
REM === Gineung AI server: one-time setup (Windows + NVIDIA) ===
cd /d "%~dp0"
REM Prefer Anaconda (the "python" on PATH can be the Microsoft Store stub)
set PY=
if exist "%USERPROFILE%\anaconda3\python.exe" set PY="%USERPROFILE%\anaconda3\python.exe"
if "%PY%"=="" if exist "%ProgramData%\anaconda3\python.exe" set PY="%ProgramData%\anaconda3\python.exe"
if "%PY%"=="" (python -c "import sys" >nul 2>nul && set PY=python)
if "%PY%"=="" (echo [!] Python not found. Install Python 3.11 from python.org and CHECK "Add python.exe to PATH", then run this again. & pause & exit /b 1)
echo Using Python: %PY%
set VENV=%USERPROFILE%\gineung-venv
if not exist "%VENV%" %PY% -m venv "%VENV%"
if "%VENV%"=="" set VENV=%USERPROFILE%\gineung-venv
call "%VENV%\Scripts\activate"
set PYTHONUTF8=1
python -m pip install --upgrade pip
echo [1/3] PyTorch (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121
echo [2/3] Server packages
pip install -r requirements.txt
echo [3/3] cloudflared (public address tunnel)
if not exist bin mkdir bin
if not exist bin\cloudflared.exe powershell -NoProfile -Command "Invoke-WebRequest -Uri https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -OutFile bin\cloudflared.exe"
python -c "import torch;print(\"GPU:\",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")"
echo.
echo Setup done. Put the store CSV into the data folder, then run start_server.bat
pause

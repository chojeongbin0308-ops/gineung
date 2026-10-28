@echo off
REM === Gineung model: one-time setup (run in Anaconda Prompt) ===
cd /d "%~dp0"
call conda create -n gineung python=3.11 -y
call conda activate gineung
set PYTHONUTF8=1
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo [GPU] NVIDIA GPU found - installing CUDA build of PyTorch
  pip install torch --index-url https://download.pytorch.org/whl/cu121
) else (
  echo [CPU] No NVIDIA GPU - installing CPU build of PyTorch
  pip install torch
)
pip install -r requirements.txt
python -c "import torch;print('torch',torch.__version__,'GPU:',torch.cuda.is_available())"
echo.
echo Setup done. Next: run train.bat
pause

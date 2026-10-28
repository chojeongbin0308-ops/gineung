@echo off
REM === Check NVIDIA GPU and reinstall CUDA PyTorch. Writes gpu_check.txt ===
cd /d "%~dp0"
set VENV=%USERPROFILE%\gineung-venv
call "%VENV%\Scripts\activate"
set PYTHONUTF8=1
echo ===== BEFORE ===== > gpu_check.txt
python --version >> gpu_check.txt 2>&1
python -c "import torch;print('torch',torch.__version__,'cuda build',torch.version.cuda,'gpu',torch.cuda.is_available())" >> gpu_check.txt 2>&1
echo ----- nvidia-smi ----- >> gpu_check.txt
nvidia-smi >> gpu_check.txt 2>&1
echo Reinstalling CUDA PyTorch (a few minutes)...
pip uninstall -y torch >> gpu_check.txt 2>&1
pip install torch --index-url https://download.pytorch.org/whl/cu126 >> gpu_check.txt 2>&1 || pip install torch --index-url https://download.pytorch.org/whl/cu124 >> gpu_check.txt 2>&1
echo ===== AFTER ===== >> gpu_check.txt
python -c "import torch;print('torch',torch.__version__,'cuda build',torch.version.cuda,'gpu',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" >> gpu_check.txt 2>&1
type gpu_check.txt | findstr /C:"torch " /C:"Python" /C:"Driver Version" /C:"ERROR"
echo.
echo Done. Tell Claude it finished (result saved in gpu_check.txt).
pause

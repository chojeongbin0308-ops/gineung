@echo off
REM === Gineung model: train + evaluate + export (run in Anaconda Prompt) ===
cd /d "%~dp0\.."
call conda activate gineung
set PYTHONUTF8=1
python model\make_dataset.py
python model\rule_baseline.py
python model\train_intent.py --model monologg/koelectra-small-v3-discriminator --epochs 5 --out runs\koelectra-small
python model\train_intent.py --model klue/roberta-small --epochs 5 --out runs\roberta-small
echo.
echo Done. Results are in the runs folder (metrics.json, onnx\model_quantized.onnx)
pause

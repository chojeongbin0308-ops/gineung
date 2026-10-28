#!/bin/bash
# === 기능상속 모델: 데이터 만들기 → 기준선 → 두 모델 학습·평가·ONNX 내보내기 (맥북) ===
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gineung
export PYTORCH_ENABLE_MPS_FALLBACK=1   # 애플 GPU가 못 하는 연산은 CPU로 자동 처리
set -e
python make_dataset.py
python human_to_jsonl.py
python rule_baseline.py
python train_intent.py --model monologg/koelectra-small-v3-discriminator --epochs 5 --out runs/koelectra-small
python train_intent.py --model klue/roberta-small --epochs 5 --out runs/roberta-small
echo
echo "끝. 결과: runs/koelectra-small, runs/roberta-small 폴더 (metrics.json, onnx/model_quantized.onnx)"

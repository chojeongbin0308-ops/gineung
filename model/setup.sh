#!/bin/bash
# === 기능상속 모델: 맥북 처음 한 번 준비 (터미널에서 실행) ===
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n gineung python=3.11 -y
conda activate gineung
pip install torch            # 애플 칩(M1~M4)은 이것만으로 GPU(MPS) 사용
pip install -r requirements.txt
python -c "import torch;print('torch',torch.__version__,'| 애플 GPU(MPS):',torch.backends.mps.is_available())"
echo
echo "준비 끝. 다음: ./train.sh"

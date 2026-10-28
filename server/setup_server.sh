#!/bin/bash
# 기능상속 AI 서버: 맥(애플 칩·인텔) 처음 한 번 설치.  실행: bash setup_server.sh
set -e
cd "$(dirname "$0")"
PY=python3
command -v python3.11 >/dev/null && PY=python3.11
$PY -c 'import sys; assert sys.version_info >= (3,9), "Python 3.9 이상이 필요합니다"'
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
echo "[1/3] PyTorch"; pip install -q torch
echo "[2/3] 서버 패키지"; pip install -q -r requirements.txt
echo "[3/3] cloudflared (공개 주소)"
mkdir -p bin
if [ ! -x bin/cloudflared ] && ! command -v cloudflared >/dev/null; then
  ARCH=$(uname -m); [ "$ARCH" = "arm64" ] || ARCH=amd64
  curl -fsSL -o bin/cf.tgz "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-$ARCH.tgz"
  tar -xzf bin/cf.tgz -C bin && rm bin/cf.tgz && chmod +x bin/cloudflared
fi
python -c "import torch;print('MPS(GPU):', torch.backends.mps.is_available())"
echo; echo "설치 끝. data 폴더에 상가 CSV를 넣고: bash start_server.sh"

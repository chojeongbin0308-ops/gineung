#!/bin/bash
# 기능상속 AI 서버 켜기 (이 터미널 창은 닫지 마세요. 끄기: Ctrl+C 두 번)
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
caffeinate -i python run.py   # 켜져 있는 동안 맥이 잠들지 않게

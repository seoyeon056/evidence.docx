"""
로컬(GPU 없는 환경)에서 UI/파이프라인 배관만 빠르게 확인하기 위한 실행기.
작은 스탠드인 모델(Qwen2.5-0.5B-Instruct)로 강제 전환 - 품질 확인용이 아니라
화면/흐름 확인용. Kaggle 등 실제 GPU 환경에서는 그냥 app.py를 직접 실행할 것.
"""

import os
import sys
from pathlib import Path

# python scripts/local_demo.py로 실행하면 sys.path[0]이 scripts/가 되어 'src'를
# 못 찾음 - 리포 루트를 명시적으로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("EVIDENCE_DOCX_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

import src.agent.extract as extract

extract.EXTRACT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

from app import demo

if __name__ == "__main__":
    demo.launch()

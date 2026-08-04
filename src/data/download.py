"""
1주차 데이터 확보.

ACI-Bench, MTS-Dialog는 git clone으로, PLABA는 OSF(rnpmf 프로젝트)에서 신청 없이
바로 받을 수 있습니다.

사용법:
    python -m src.data.download
"""

import subprocess
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

REPOS = {
    "aci-bench": "https://github.com/wyim/aci-bench.git",
    "mts-dialog": "https://github.com/abachaa/MTS-Dialog.git",
}

# https://osf.io/rnpmf/ 의 osfstorage 파일 GUID 기준 (PDF 등 참고 문서는 제외)
PLABA_FILES = {
    "train.csv": "https://osf.io/download/g3t5x/",
    "val.csv": "https://osf.io/download/qa3hd/",
    "test.csv": "https://osf.io/download/6ksbm/",
    "data.json": "https://osf.io/download/4kp7v/",
}


def clone_repos() -> None:
    for name, url in REPOS.items():
        target = RAW_DIR / name
        if target.exists():
            print(f"[skip] {name} already exists at {target}")
            continue
        print(f"[clone] {name}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)


def download_plaba() -> None:
    target_dir = RAW_DIR / "plaba"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in PLABA_FILES.items():
        target = target_dir / filename
        if target.exists():
            print(f"[skip] plaba/{filename} already exists")
            continue
        print(f"[download] plaba/{filename}")
        urllib.request.urlretrieve(url, target)


def download_all() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    clone_repos()
    download_plaba()


if __name__ == "__main__":
    download_all()

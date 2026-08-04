"""
1주차 데이터 확보.

ACI-Bench, MTS-Dialog는 git clone으로 바로 받을 수 있습니다.
PLABA는 NLM 공식 shared task 페이지에서 신청/승인 절차가 있어 가장 먼저
신청해두는 것을 권장합니다 (다른 두 데이터보다 오래 걸릴 수 있음).

사용법:
    python -m src.data.download
"""

import subprocess
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

REPOS = {
    "aci-bench": "https://github.com/wyim/aci-bench.git",
    "mts-dialog": "https://github.com/abachaa/MTS-Dialog.git",
}


def clone_repos() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        target = RAW_DIR / name
        if target.exists():
            print(f"[skip] {name} already exists at {target}")
            continue
        print(f"[clone] {name}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)

    print(
        "\nPLABA는 자동 다운로드 불가 - NLM 공식 shared task 페이지에서 신청 후 "
        f"'{RAW_DIR / 'plaba'}' 에 수동으로 배치하세요."
    )


if __name__ == "__main__":
    clone_repos()

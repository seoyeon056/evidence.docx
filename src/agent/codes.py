"""
질병코드/약품코드 매칭 + RAG 기반 의료용어 사전.

claim/노트 텍스트에 실제로 등장하는 공식 질병명·약품명을 찾아 코드를 붙인다.
LLM이 코드를 생성하는 게 아니라, 건강보험심사평가원 공식 코드표에서 텍스트에
그대로 등장하는 명칭만 찾아 매칭하는 방식 - 코드 오생성(hallucination) 방지가
목적이라 임베딩 기반 유사도 검색이 아니라 정확 매칭을 씀 (비슷한 이름의 다른
질병/약품 코드가 붙는 것은 근거검증 취지에 어긋남).

데이터 출처 (data/raw/codes/, .gitignore 대상 - 아래 안내대로 수동 다운로드):
    disease_master.csv    - 건강보험심사평가원_상병마스터 (통계청 KCD 기반)
                             https://www.data.go.kr 에서 "상병마스터" 검색
    medication_master.csv - 건강보험심사평가원_약가마스터_의약품표준코드
                             https://www.data.go.kr/data/15067462/fileData.do
    둘 다 로그인 없이 다운로드 가능. cp949 인코딩이라 utf-8로 변환해서 사용.
"""

import csv
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "codes"
DISEASE_CSV = DATA_DIR / "disease_master.csv"
MEDICATION_CSV = DATA_DIR / "medication_master.csv"

MIN_TERM_LENGTH = 3  # 너무 짧은 명칭은 오탐(false match) 위험이 커서 매칭 대상에서 제외
MIN_TOKEN_LENGTH = 3  # 텍스트에서 뽑는 후보 단어도 동일 기준

# 실제 텍스트엔 없는 일반적인 단어라 매칭에서 제외 (필요시 계속 추가)
_STOPWORDS = {"환자", "오늘", "지금", "발생", "확인", "증상", "상태", "진료", "설명", "안내"}


def _extract_candidate_tokens(text: str) -> set[str]:
    raw_tokens = re.findall(r"[가-힣A-Za-z]{2,}", text)
    tokens: set[str] = set()
    for t in raw_tokens:
        if t not in _STOPWORDS and len(t) >= MIN_TOKEN_LENGTH:
            tokens.add(t)
        # 조사(을/를/이/가/은/는/과/와 등) 제거 휴리스틱 - 끝 1글자 뗀 버전도 후보에 추가
        if len(t) >= MIN_TOKEN_LENGTH + 1 and t[:-1] not in _STOPWORDS:
            tokens.add(t[:-1])
    return tokens


@lru_cache(maxsize=1)
def _load_disease_terms() -> list[dict]:
    with open(DISEASE_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen: set[str] = set()
    terms = []
    for row in rows:
        name = row["한글명"].strip()
        if len(name) < MIN_TERM_LENGTH or name in seen:
            continue
        seen.add(name)
        terms.append({"code": row["상병기호"].strip(), "name": name, "name_en": row["영문명"].strip()})
    return terms


@lru_cache(maxsize=1)
def _load_medication_terms() -> list[dict]:
    with open(MEDICATION_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen: set[str] = set()
    terms = []
    for row in rows:
        name = row["한글상품명"].strip()
        if len(name) < MIN_TERM_LENGTH or name in seen:
            continue
        seen.add(name)
        terms.append(
            {
                "code": row["표준코드"].strip(),
                "name": name,
                "atc_code": row["국제표준코드(ATC코드)"].strip(),
            }
        )
    return terms


def _dedupe_by_code_prefix(matches: list[dict], prefix_len: int) -> list[dict]:
    # KCD는 유형×합병증마다 별도 leaf 코드를 매겨서("당뇨병" 하나가 1800+건 매칭)
    # 원본 그대로 보여주면 못 씀 - 코드 앞 prefix_len자리(카테고리 단위, 예: E10)로
    # 묶어서 대표 1건만 남김. 대표는 이름이 가장 짧은 것(수식어 제일 적은 것)을 채택.
    best: dict[str, dict] = {}
    for m in matches:
        key = m["code"][:prefix_len]
        if key not in best or len(m["name"]) < len(best[key]["name"]):
            best[key] = m
    # 코드순이 아니라 이름 짧은(=더 일반적인/단순 매칭인) 순으로 정렬 - max_results로
    # 자를 때 "고혈압"(단순) 같은 게 "당뇨병성 자율신경병증(...)"(복합) 같은 것에
    # 밀려나지 않게 함
    return sorted(best.values(), key=lambda m: len(m["name"]))


def find_disease_codes(text: str, max_results: int = 10) -> list[dict]:
    # KCD 병명은 "고혈압"처럼 단독으로 쓰이기도 하고, "1형 당뇨병"처럼 항상
    # 세부분류가 붙기도 함 - 양방향으로 다 확인
    tokens = _extract_candidate_tokens(text)
    matches = [
        t
        for t in _load_disease_terms()
        if t["name"] in text or any(tok in t["name"] for tok in tokens)
    ]
    return _dedupe_by_code_prefix(matches, prefix_len=3)[:max_results]


def find_medication_codes(text: str, max_results: int = 10) -> list[dict]:
    # 상품명은 "어린이용타이레놀정80밀리그람(아세트아미노펜)"처럼 거의 항상
    # 브랜드명+용량/제형이 붙어있어서, 텍스트의 짧은 단어가 상품명 안에
    # 포함되는지(역방향)가 핵심 매칭 경로
    tokens = _extract_candidate_tokens(text)
    matches = [
        t
        for t in _load_medication_terms()
        if t["name"] in text or any(tok in t["name"] for tok in tokens)
    ]
    matches.sort(key=lambda m: len(m["name"]))
    return matches[:max_results]


def lookup_medical_terms(text: str) -> dict:
    """RAG 의료용어 사전: 텍스트에 등장하는 공식 질병명/약품명을 모두 찾아 반환."""
    return {
        "diseases": find_disease_codes(text),
        "medications": find_medication_codes(text),
    }

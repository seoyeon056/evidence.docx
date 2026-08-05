"""
3주차: 근거검증 모듈 (제로샷, 추가 학습 없음).

claim이 원본 대화(dialogue)에 실제로 근거하는지 NLI cross-encoder로
entailment 판별. premise=dialogue, hypothesis=claim.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
ENTAILMENT_THRESHOLD = 0.5  # TODO(3주차): 실제 claim 샘플로 임계값 튜닝 필요

# cross-encoder/nli-deberta-v3-base의 라벨 순서 (config.id2label 확인함)
_ENTAILMENT_INDEX = 1

# 모델 max_seq_length(512)는 premise+hypothesis 합산 기준. 진료 대화는 흔히
# 1000토큰 이상이라(실측: ACI-Bench 샘플 1902토큰) 그대로 넣으면 뒷부분이
# 잘려나가 그 부분 근거의 claim이 잘못 "근거 없음" 처리됨. premise를 겹치는
# 청크로 나눠 각각 검사하고 최댓값을 채택.
_PREMISE_CHUNK_TOKENS = 400  # hypothesis + 스페셜 토큰 여유 확보
_CHUNK_OVERLAP_TOKENS = 50  # 청크 경계에서 근거 문장이 잘리는 것 완화


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    return CrossEncoder(NLI_MODEL_NAME)


def _chunk_premise(premise: str) -> list[str]:
    tokenizer = _get_model().tokenizer
    ids = tokenizer(premise, add_special_tokens=False)["input_ids"]
    if len(ids) <= _PREMISE_CHUNK_TOKENS:
        return [premise]

    step = _PREMISE_CHUNK_TOKENS - _CHUNK_OVERLAP_TOKENS
    chunks = []
    for start in range(0, len(ids), step):
        chunk_ids = ids[start : start + _PREMISE_CHUNK_TOKENS]
        chunks.append(tokenizer.decode(chunk_ids))
        if start + _PREMISE_CHUNK_TOKENS >= len(ids):
            break
    return chunks


def entailment_score(premise: str, hypothesis: str) -> float:
    model = _get_model()
    chunks = _chunk_premise(premise)
    scores = model.predict([(chunk, hypothesis) for chunk in chunks], apply_softmax=True)
    return float(max(s[_ENTAILMENT_INDEX] for s in scores))


def verify(dialogue: str, claims: list[str], threshold: float = ENTAILMENT_THRESHOLD) -> list[dict]:
    return [
        {
            "text": claim,
            "entailment_score": (score := entailment_score(dialogue, claim)),
            "verified": score >= threshold,
        }
        for claim in claims
    ]

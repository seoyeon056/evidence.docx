"""
3주차: claim 추출 - SOAP 노트를 검증 가능한 개별 claim으로 분리.

sLLM 프롬프팅만으로 동작(추가 학습 불필요) - 2주차 파인튜닝된 노트 생성
모델과는 독립적이라, 그 결과를 기다리지 않고 베이스 모델로 먼저 만들고
검증할 수 있음.
"""

import json
import re
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 지금은 가벼운 베이스 모델로 프롬프트/파싱 로직을 검증. 실제 파이프라인에서는
# 2주차에 학습한 sLLM(또는 그 이상 크기의 instruct 모델)으로 교체 가능.
EXTRACT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You extract individual factual claims from a clinical note. "
    "Split the note into short, self-contained factual statements - one claim per "
    "distinct fact (symptom, diagnosis, medication, instruction, etc). "
    "Respond with a JSON array of strings only, no other text."
)


@lru_cache(maxsize=1)
def _get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(EXTRACT_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(EXTRACT_MODEL_NAME, dtype=torch.float32)
    return model, tokenizer


def _parse_claims(raw_text: str) -> list[str]:
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match:
        try:
            claims = json.loads(match.group(0))
            if isinstance(claims, list):
                return [str(c).strip() for c in claims if str(c).strip()]
        except json.JSONDecodeError:
            pass
    # JSON 파싱 실패 시 줄 단위로 fallback (불릿/번호 접두어 제거)
    lines = [line.strip(" -*\t") for line in raw_text.splitlines()]
    lines = [re.sub(r"^\d+[.)]\s*", "", line) for line in lines]
    return [line for line in lines if line]


def extract_claims_from_note(note: str, max_new_tokens: int = 512) -> list[str]:
    model, tokenizer = _get_model_and_tokenizer()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": note},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return _parse_claims(generated)

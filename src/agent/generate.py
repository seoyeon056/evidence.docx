"""
2주차 QLoRA 어댑터로 노트/요약 생성.

어댑터(checkpoints/qlora-multitask/final)가 아직 없으면 베이스 모델만으로
동작 - 품질은 낮지만 코드 경로는 동일해서, 학습이 끝나면 이 파일을 손대지
않고도 바로 좋아짐.
"""

from functools import lru_cache
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import SYSTEM_PROMPT

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "qlora-multitask" / "final"


@lru_cache(maxsize=1)
def _get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map="auto", dtype=torch.float16
        )
    else:
        # GPU 없는 환경(로컬 스모크 테스트 등) - 양자화 없이 그냥 로드
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if ADAPTER_PATH.exists():
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))

    return model, tokenizer


def _generate(task: str, input_text: str, max_new_tokens: int = 1024) -> str:
    model, tokenizer = _get_model_and_tokenizer()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[{task}] {input_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    # instruction을 잘 안 따르는 모델(특히 파인튜닝 전 베이스 모델)이 응답 맨 앞에
    # 태그를 그대로 따라 하는 경우가 있어 방어적으로 제거
    return text.removeprefix(f"[{task}]").strip()


def generate_note(dialogue: str) -> str:
    return _generate("NOTE", dialogue)


def generate_summary(note: str) -> str:
    return _generate("SUMMARY", note)

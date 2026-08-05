"""
2주차: Qwen2.5-7B-Instruct 멀티태스크 QLoRA 파인튜닝.

data/processed/train_multitask.jsonl (NOTE + SUMMARY 예시)에 [NOTE]/[SUMMARY]
태그를 프리픽스로 붙여서, 하나의 LoRA adapter로 두 태스크를 동시에 학습합니다.

Kaggle GPU 노트북에서:
    !git pull
    !python -m src.data.download && python -m src.data.preprocess   # 아직 안 했으면
    !python -m src.training.train_qlora
"""

import json
import os
from pathlib import Path

# torch import 전에 설정해야 CUDA 할당자에 반영됨 - 단편화로 인한 OOM 완화
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "train_multitask.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "qlora-multitask"

SYSTEM_PROMPT = (
    "당신은 의료 문서 작성 보조 AI입니다. "
    "[NOTE] 태그가 붙으면 진료 대화를 SOAP 형식 노트로 작성하고, "
    "[SUMMARY] 태그가 붙으면 의료 텍스트를 환자가 이해하기 쉬운 말로 풀어씁니다."
)


def load_examples(path: Path = DATA_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_messages(example: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[{example['task']}] {example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]


def build_dataset(tokenizer, path: Path = DATA_PATH) -> Dataset:
    examples = load_examples(path)
    texts = [
        tokenizer.apply_chat_template(to_messages(ex), tokenize=False, add_generation_prompt=False)
        for ex in examples
    ]
    return Dataset.from_dict({"text": texts})


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(tokenizer).train_test_split(test_size=0.05, seed=42)

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        # 1 에폭 = 파이프라인이 끝까지 도는지 먼저 확인하는 용도. 결과물 품질이
        # 부족하면 여기를 늘려 재학습 (트레이드오프는 README 참고).
        num_train_epochs=1,
        learning_rate=2e-4,
        # fp16으로 시도했으나 Kaggle 환경(accelerate 기본 설정으로 추정)에서 LoRA
        # 가중치 일부가 계속 bf16으로 남아 GradScaler.unscale_에서
        # "not implemented for BFloat16"로 반복 실패. bf16은 loss scaling이
        # 필요 없어 GradScaler 자체를 안 써서 이 문제를 원천적으로 피함.
        # T4엔 bf16 텐서 코어가 없어 fp16보다 다소 느리지만 안정성 우선.
        bf16=True,
        optim="paged_adamw_8bit",  # 8bit 페이지드 옵티마이저로 메모리 여유 확보
        logging_steps=20,
        # epoch당으로 하면 1 에폭 학습 중엔 체크포인트가 하나도 안 남아서, 세션이
        # 끊기면 진행분이 통째로 날아감 - steps 기준으로 자주 저장해 재개 가능하게 함
        save_strategy="steps",
        save_steps=20,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=20,
        # loss_type="nll"(청크 없이 한 번에 logits 계산)이라 seq_len x 15만 vocab
        # 텐서가 그대로 메모리에 올라감 - 3072에서 OOM 나서 2048로 낮춤
        # (p90=937 토큰까지 커버, 전체의 98.1%가 안 잘림).
        max_length=2048,
        packing=False,
        dataset_text_field="text",
        # trl 기본값 loss_type="chunked_nll"은 PEFT로 감싼 모델의 forward와
        # 호환되지 않아 "'functools.partial' object has no attribute '__func__'"로
        # 터짐 (trl/trainer/sft_trainer.py의 _patch_chunked_ce_lm_head). "nll"로
        # 명시해 그 패치 경로를 아예 타지 않게 함.
        loss_type="nll",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    # OUTPUT_DIR에 체크포인트가 이미 있으면(세션 끊김 후 재실행 등) 그 지점부터 재개
    has_checkpoint = OUTPUT_DIR.exists() and any(OUTPUT_DIR.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=has_checkpoint)
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))


if __name__ == "__main__":
    main()

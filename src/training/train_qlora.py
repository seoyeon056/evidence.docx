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
        # T4는 Turing(Compute Capability 7.5)이라 bf16 텐서 코어가 없음 - fp16 사용
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
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
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,  # T4엔 bf16 텐서 코어가 없어 fp16 사용
        optim="paged_adamw_8bit",  # 8bit 페이지드 옵티마이저로 메모리 여유 확보
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
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

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))


if __name__ == "__main__":
    main()

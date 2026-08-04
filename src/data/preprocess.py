"""
멀티태스크 instruction 포맷 설계 (1주차).

ACI-Bench/MTS-Dialog(대화->SOAP)와 PLABA(전문 문장->평이체)를 하나의 학습셋으로
합쳐 단일 QLoRA adapter를 태스크 프리픽스로 학습시킵니다.

[NOTE] {진료 대화}      -> {SOAP 노트}
[SUMMARY] {SOAP 노트}   -> {환자용 평이체 요약}

원본 스키마 (실제 다운로드 데이터 기준):
    ACI-Bench   data/challenge_data/train.csv         : dataset, encounter_id, dialogue, note
    MTS-Dialog  Main-Dataset/MTS-Dialog-TrainingSet.csv: ID, section_header, section_text, dialogue
    PLABA       plaba/train.csv                        : question, pmid, input_text, target_text, ...
"""

import csv
import json
from pathlib import Path
from typing import TypedDict

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

ACI_BENCH_TRAIN = RAW_DIR / "aci-bench" / "data" / "challenge_data" / "train.csv"
MTS_DIALOG_TRAIN = RAW_DIR / "mts-dialog" / "Main-Dataset" / "MTS-Dialog-TrainingSet.csv"
PLABA_TRAIN = RAW_DIR / "plaba" / "train.csv"


class InstructionExample(TypedDict):
    task: str  # "NOTE" | "SUMMARY"
    input: str
    output: str


def format_example(task: str, input_text: str, output_text: str) -> InstructionExample:
    return {"task": task, "input": input_text, "output": output_text}


def to_prompt(example: InstructionExample) -> str:
    return f"[{example['task']}] {example['input']}"


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_note_examples(
    aci_bench_csv: Path = ACI_BENCH_TRAIN, mts_dialog_csv: Path = MTS_DIALOG_TRAIN
) -> list[InstructionExample]:
    examples: list[InstructionExample] = []

    for row in _read_csv(aci_bench_csv):
        if not row["dialogue"] or not row["note"]:
            continue
        examples.append(format_example("NOTE", row["dialogue"], row["note"]))

    for row in _read_csv(mts_dialog_csv):
        if not row["dialogue"] or not row["section_text"]:
            continue
        # MTS-Dialog는 전체 노트가 아니라 섹션 단위 요약 - section_header를 태그로 남겨
        # ACI-Bench의 전체 노트와 형식을 구분
        output = f"[{row['section_header']}] {row['section_text']}"
        examples.append(format_example("NOTE", row["dialogue"], output))

    return examples


def build_summary_examples(plaba_csv: Path = PLABA_TRAIN) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    for row in _read_csv(plaba_csv):
        if not row["input_text"] or not row["target_text"]:
            continue
        examples.append(format_example("SUMMARY", row["input_text"], row["target_text"]))
    return examples


def build_dataset() -> list[InstructionExample]:
    return build_note_examples() + build_summary_examples()


def save_jsonl(examples: list[InstructionExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    note_examples = build_note_examples()
    summary_examples = build_summary_examples()
    print(f"NOTE examples: {len(note_examples)}")
    print(f"SUMMARY examples: {len(summary_examples)}")

    out_path = PROCESSED_DIR / "train_multitask.jsonl"
    save_jsonl(note_examples + summary_examples, out_path)
    print(f"saved -> {out_path}")

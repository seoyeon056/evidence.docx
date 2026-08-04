"""
멀티태스크 instruction 포맷 설계 (1주차).

ACI-Bench/MTS-Dialog(대화->SOAP)와 PLABA(전문 문장->평이체)를 하나의 학습셋으로
합쳐 단일 QLoRA adapter를 태스크 프리픽스로 학습시킵니다.

[NOTE] {진료 대화}      -> {SOAP 노트}
[SUMMARY] {SOAP 노트}   -> {환자용 평이체 요약}
"""

from typing import TypedDict


class InstructionExample(TypedDict):
    task: str  # "NOTE" | "SUMMARY"
    input: str
    output: str


def format_example(task: str, input_text: str, output_text: str) -> InstructionExample:
    return {"task": task, "input": input_text, "output": output_text}


def to_prompt(example: InstructionExample) -> str:
    return f"[{example['task']}] {example['input']}"


def build_note_examples(aci_bench_rows, mts_dialog_rows) -> list[InstructionExample]:
    # TODO(1주차): ACI-Bench/MTS-Dialog 원본 스키마에 맞춰 (대화, SOAP 노트) 쌍 추출
    raise NotImplementedError


def build_summary_examples(plaba_rows) -> list[InstructionExample]:
    # TODO(1주차): PLABA 원본 스키마에 맞춰 (전문 문장, 평이체 문장) 쌍 추출
    raise NotImplementedError

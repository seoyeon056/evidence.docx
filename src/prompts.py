"""학습(train_qlora)과 추론(generate) 양쪽에서 공유하는 프롬프트 상수.

trl/bitsandbytes 같은 학습 전용 무거운 의존성과 분리해두기 위해 별도 파일로 둠 -
추론만 하는 코드(generate.py, app.py)가 학습 스택까지 끌고 오지 않게 함.
"""

SYSTEM_PROMPT = (
    "당신은 의료 문서 작성 보조 AI입니다. "
    "[NOTE] 태그가 붙으면 진료 대화를 SOAP 형식 노트로 작성하고, "
    "[SUMMARY] 태그가 붙으면 의료 텍스트를 환자가 이해하기 쉬운 말로 풀어씁니다."
)

# evidence.docx

근거 검증 기반 AI 진료 스크라이브 & 환자 맞춤 요약 에이전트

AI가 생성한 진료 노트(SOAP)에서 근거가 약한 문장을 자동으로 표시해 의료진의 검토 부담을 줄이고, 검증을 통과한 내용만으로 환자용 쉬운 설명 요약을 생성합니다.

## 문제의식

- 국내 대형병원·병의원 다수가 이미 보이스 EMR(AI 진료 스크라이브)을 도입했지만, AI가 생성한 노트 중 어떤 문장이 근거가 약한지 시스템이 표시해주지 않아 검토 부담이 의료진에게 그대로 남아 있습니다.
- 국내 성인 헬스 리터러시 평균은 16점 만점에 11.3점(적정 수준 50.6%)으로, 의료진용으로 설계된 AI 노트를 환자가 그대로 이해하기 어렵습니다.

## 아키텍처 (LangGraph)

```
진료 대화 입력
    │
    ▼
[generate_soap_note]   대화 → SOAP 노트 (멀티태스크 QLoRA sLLM)
    │
    ▼
[extract_claims]       노트를 claim 단위로 분리 (sLLM 프롬프팅)
    │
    ▼
[verify_claims]        각 claim이 원본 대화에 근거하는지 NLI entailment 판별
    │
    ▼
[human_review]         근거 약한 claim 하이라이트 → 의료진 확인/수정 (interrupt)
    │
    ▼
[generate_patient_summary]  검증 통과한 내용만으로 환자용 평이체 요약 생성
```

사람이 개입해 상태를 수정하고 재개하는 human-in-the-loop 구조라 LangGraph의 상태 그래프 + interrupt를 사용합니다.

## 실행 일정 (4주 로드맵)

| 주차 | 목표 |
|---|---|
| 1주차 | 데이터 전처리(ACI-Bench, MTS-Dialog, PLABA), 멀티태스크 instruction 포맷 설계, 환경 셋업 |
| 2주차 | 멀티태스크 QLoRA 파인튜닝(SOAP 생성 + 환자용 요약 동일 adapter), 1차 평가 |
| 3주차 | 근거검증 모듈(claim 추출 프롬프트, NLI entailment 임계값 튜닝) |
| 4주차 | Gradio 웹 데모 통합, 버그 수정, 한국어 few-shot 데모 예시, 발표자료 |

## 데이터

- **ACI-Bench, MTS-Dialog**: 진료 대화-SOAP 노트 쌍 (SOAP 생성 파인튜닝용)
- **PLABA**: PubMed 초록 전문가 평이화 문장 쌍 (환자용 요약 파인튜닝용)
- 실제 환자 데이터는 사용하지 않음. 한국어는 MVP 단계에서 재학습 없이 few-shot 데모 예시로만 확보.

## 기술 스택

- 베이스 모델: Qwen2.5-7B-Instruct (QLoRA)
- 오케스트레이션: LangGraph
- 근거검증: 경량 사전학습 NLI 모델 (제로샷)
- 데모: Gradio

## 시작하기

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

데이터 다운로드는 `src/data/download.py` 참고.

# evidence.docx — 진행 상황 요약 (새 대화 이어받기용)

## 프로젝트 개요
- **이름**: evidence.docx (Healthcare & Bio sLLM 프로젝트 과정, 1개월 프로젝트, 국민대 남서연)
- **컨셉**: 근거 검증 기반 AI 진료 스크라이브 & 환자 맞춤 요약 에이전트 — 진료 대화 → SOAP 노트 생성 → claim 단위 근거검증 → 의료진 리뷰 → 검증된 내용만으로 환자용 쉬운 요약
- **GitHub**: https://github.com/seoyeon056/evidence.docx (public repo)
- **로컬 경로**: `C:\Users\longa\BDPV\oracleproj`
- **차별점**: AI가 생성한 노트 중 근거 약한 문장을 자동 표시 (기존 보이스 EMR엔 없는 기능)

## 완료된 것

**1주차 (데이터)**
- `src/data/download.py`: ACI-Bench, MTS-Dialog, PLABA 자동 다운로드 (PLABA는 신청 불필요, OSF 공개 다운로드로 확인됨)
- `src/data/preprocess.py`: `[NOTE]`/`[SUMMARY]` 태그로 멀티태스크 학습셋 생성 (실측 1,903개 예시: NOTE 1,268 + SUMMARY 635)

**설계 (LangGraph 파이프라인)**
- `src/agent/state.py`, `src/agent/graph.py`: 대화→SOAP노트→claim추출→근거검증→의료진리뷰(interrupt)→환자요약, 5개 노드
- `src/prompts.py`: 학습/추론 공유 SYSTEM_PROMPT (trl 등 무거운 학습 의존성과 분리)

**3주차 (근거검증) — 완료**
- `src/agent/verify.py`: NLI 기반(`cross-encoder/nli-deberta-v3-base`, 제로샷) claim 검증. 512토큰 초과 대화 청크 분할 처리(원래 잘려서 오탐되던 버그 수정함). 실데이터 검증 완료.
- `src/agent/extract.py`: claim 추출 프롬프팅 (verbatim 추출 지시 포함, 노트 내 위치 찾기용)

**4주차 (데모) — 완료**
- `src/agent/generate.py`: 노트/요약 생성. 어댑터 있으면 로드, 없으면 베이스 모델만 사용 (학습 전후 코드 동일)
- `app.py`: Gradio 데모. SOAP 노트를 `gr.HighlightedText`로 표시해 근거 약한 부분을 **노트 본문 안에서** 하이라이트 (별도 리스트로도 동시 표시). LangGraph interrupt/resume으로 "검토 완료" 버튼 클릭 시 재개.
- `scripts/local_demo.py`: 로컬 GPU 없는 환경에서 작은 모델(Qwen2.5-0.5B-Instruct)로 UI만 빠르게 확인하는 런처 (`EVIDENCE_DOCX_MODEL` 환경변수로 모델 교체 가능)
- **실제 브라우저로 한국어 입력까지 end-to-end 테스트 완료**: 노트/근거검증/요약 전부 한국어로 정상 동작, 근거 약한 부분(모델이 지어낸 조언)이 정확히 잡힘

## 미완료 — 2주차 (파인튜닝) ⚠️ 여기서 막혀있음

`src/training/train_qlora.py`: Qwen2.5-7B-Instruct + QLoRA, 멀티태스크(`[NOTE]`/`[SUMMARY]`) 학습.

**코드 자체는 안정적으로 완성됨** (아래 이슈들 다 해결 반영됨):
- bf16으로 최종 확정 (fp16은 GradScaler가 이 환경에서 계속 크래시 — 4번 다른 방식 시도 다 실패, bf16은 GradScaler 자체를 안 써서 안전)
- `per_device_eval_batch_size` 미설정으로 OOM 났던 버그 수정
- eval이 저장(save)을 막던 버그 발견 → eval 비활성화 (`eval_strategy="no"`)
- `max_length=1536`, `batch_size=1 × grad_accum=16`, `paged_adamw_8bit`로 T4 16GB에 맞춤
- **`GitBackupCallback`**: 20스텝마다 체크포인트를 `git add -f`로 강제 추가해 GitHub에 자동 푸시 (checkpoints/는 .gitignore 대상이지만, 한 번 커밋되면 이후 clone/pull에서 정상 복원됨 — 로컬 샌드박스에서 검증 완료). **`GITHUB_TOKEN` 환경변수 필요** (`%env GITHUB_TOKEN=...` 또는 `import os; os.environ["GITHUB_TOKEN"]=...`)
- `resume_from_checkpoint`: `checkpoints/qlora-multitask/` 안에 `checkpoint-*` 폴더 있으면 자동 재개

**막힌 이유는 인프라(GPU 확보), 코드 문제 아님**:
- **Kaggle**: 여러 번 시도, 세션이 완전히 죽으면(커널 인터럽트 아니라 세션 자체 종료) `/kaggle/working` 전체가 날아감(데이터+체크포인트). Git 백업 기능 완성 전엔 진행분 여러 번 통째로 날아감. T4라 애초에 느림(스텝당 ~200~250초, 전체 6~8시간)
- **AWS**: g5.xlarge(A10G, bf16 텐서코어 있어서 빠를 것으로 기대) 시도 중인데, **GPU vCPU 할당량이 0**이라 증가 신청 넣어둔 상태 (Service Quotas, "Running On-Demand G and VT instances", 요청값 4, 상태: **Case Opened, 승인 대기중**). ⚠️ 참고: "Deep Learning Base GPU AMI On Ubuntu 24.04 with Tesla T4"라는 AMI는 **Galaxys Cloud라는 제3자가 파는 유료 상품**(시간당 $2.40 추가 요금!) — 절대 구독하지 말 것. 순정 무료 **Ubuntu Server 24.04 LTS**(Canonical, Verified provider) 써야 함.
- **Colab**: AWS 승인 기다리는 동안 병행 시도하기로 함 (막 시작하려던 참). T4라 Kaggle과 똑같이 느릴 것으로 예상되지만, git 백업 덕분에 세션 끊겨도 안전함.

## 다음에 할 일 (우선순위 순)
1. **파인튜닝 완료시키기** — Colab 또는 AWS 승인되는 대로. `!git pull && !python -m src.data.download && !python -m src.data.preprocess && !python -m src.training.train_qlora` (GITHUB_TOKEN 먼저 설정)
2. 학습 끝나면 `checkpoints/qlora-multitask/final/` 생성됨 → `generate.py`가 자동으로 이걸 로드하므로 **코드 수정 불필요**, `app.py` 그대로 다시 돌려서 실제 품질 확인
3. `verify.py`의 `ENTAILMENT_THRESHOLD=0.5`는 아직 추측값 — 실데이터로 튜닝 필요
4. 한국어 few-shot 데모 예시 추가 (원래 스코프대로 — 재학습 없이 프롬프팅으로만)
5. **보류 중인 피드백** (질병코드/약품코드 생성, 효능별 약품 추천, STT 연동, RAG 기반 의료용어 사전) — 논의만 하고 구현은 안 함. 특히 "약품 추천"은 스코프보다 **안전성 리스크**(임상 의사결정 지원으로 카테고리가 바뀜) 문제로 재검토 권장. README에 "향후 확장" 섹션으로 정리하는 것까지만 논의됨, 아직 작성 안 함
6. 4주차 발표자료

## 알아두면 좋은 것
- `gh` CLI가 `seoyeon056` 계정으로 로컬에 인증되어 있음
- 로컬 Gradio 미리보기는 `C:\Users\longa\BDPV\maskingtape\.claude\launch.json`에 `"evidence-docx-demo"` 이름으로 등록해둠 (원래 리포는 oracleproj인데, 이 세션의 기본 작업 디렉토리가 maskingtape라서 거기에 등록됨)
- GitHub 토큰이 스크린샷에 한 번 노출된 적 있어서 재발급함 — 토큰 화면 캡처 조심할 것

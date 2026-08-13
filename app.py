"""
4주차: Gradio 통합 데모.

진료 대화 입력 -> SOAP 노트 + 근거검증 하이라이트 화면까지 자동 진행 후 멈춤
(LangGraph interrupt_before=["human_review"]) -> "검토 완료" 클릭 시 재개 ->
환자용 요약 생성.

실행:
    python app.py
"""

import uuid

import gradio as gr

from src.agent.graph import build_graph

graph = build_graph()


def start_analysis(dialogue: str):
    if not dialogue.strip():
        raise gr.Error("진료 대화를 입력해주세요.")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "dialogue": dialogue,
        "soap_note": "",
        "claims": [],
        "weak_claims": [],
        "review_approved": False,
        "patient_summary": "",
    }
    # interrupt_before=["human_review"]라 여기서 human_review 직전까지만 실행되고 멈춤
    result = graph.invoke(initial_state, config)

    highlighted = [
        (claim["text"], "근거 약함" if not claim["verified"] else None)
        for claim in result["claims"]
    ]
    weak_count = len(result["weak_claims"])
    status = (
        f"근거 약한 claim {weak_count}개 발견 - 검토 후 진행해주세요."
        if weak_count
        else "모든 claim이 근거검증을 통과했습니다."
    )
    return result["soap_note"], highlighted, status, thread_id


def approve_review(thread_id: str):
    if not thread_id:
        raise gr.Error("먼저 '분석 시작'을 눌러주세요.")

    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {"review_approved": True})
    result = graph.invoke(None, config)
    return result["patient_summary"]


with gr.Blocks(title="evidence.docx") as demo:
    gr.Markdown("# evidence.docx\n근거 검증 기반 AI 진료 스크라이브 & 환자 맞춤 요약 에이전트")

    thread_id_state = gr.State("")

    with gr.Row():
        dialogue_input = gr.Textbox(
            label="진료 대화", lines=10, placeholder="[doctor] ...\n[patient] ..."
        )

    start_btn = gr.Button("분석 시작", variant="primary")

    with gr.Row():
        note_output = gr.Textbox(label="① SOAP 노트 초안", lines=10)
        review_output = gr.HighlightedText(
            label="② 근거검증 리뷰 (약한 근거만 하이라이트)",
            color_map={"근거 약함": "red"},
        )

    review_status = gr.Markdown()
    approve_btn = gr.Button("검토 완료 → 환자용 요약 생성", variant="primary")

    summary_output = gr.Textbox(label="③ 환자용 쉬운 설명 요약", lines=6)

    start_btn.click(
        start_analysis,
        inputs=[dialogue_input],
        outputs=[note_output, review_output, review_status, thread_id_state],
    )
    approve_btn.click(
        approve_review,
        inputs=[thread_id_state],
        outputs=[summary_output],
    )


if __name__ == "__main__":
    demo.launch()

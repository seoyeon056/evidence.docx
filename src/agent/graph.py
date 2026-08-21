"""
evidence.docx 파이프라인.

진료 대화 -> SOAP 노트(generate.py, QLoRA 어댑터) -> claim 추출(extract.py) ->
근거검증(verify.py, NLI 제로샷) -> 의료진 리뷰(interrupt) -> 환자용 요약(generate.py).
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agent.extract import extract_claims_from_note
from src.agent.generate import generate_note, generate_summary
from src.agent.state import AgentState
from src.agent.verify import ENTAILMENT_THRESHOLD, entailment_score


def generate_soap_note(state: AgentState) -> AgentState:
    state["soap_note"] = generate_note(state["dialogue"])
    return state


def extract_claims(state: AgentState) -> AgentState:
    texts = extract_claims_from_note(state["soap_note"])
    state["claims"] = [
        {"text": text, "source_section": "", "entailment_score": 0.0, "verified": False}
        for text in texts
    ]
    return state


def verify_claims(state: AgentState) -> AgentState:
    dialogue = state["dialogue"]
    weak_claims = []
    for claim in state["claims"]:
        score = entailment_score(dialogue, claim["text"])
        claim["entailment_score"] = score
        claim["verified"] = score >= ENTAILMENT_THRESHOLD
        if not claim["verified"]:
            weak_claims.append(claim)
    state["weak_claims"] = weak_claims
    return state


def human_review(state: AgentState) -> AgentState:
    # LangGraph interrupt로 여기서 실행을 멈추고, 의료진이 weak_claims를 확인/수정한 뒤
    # review_approved=True로 재개
    return state


def generate_patient_summary(state: AgentState) -> AgentState:
    # 검증 통과한 claim만으로 요약 - weak_claims는 의료진이 수정한 뒤
    # review_approved로 넘어오므로, 그 시점 state["claims"]의 verified 값을 기준으로 함
    verified_text = "\n".join(c["text"] for c in state["claims"] if c["verified"])
    state["patient_summary"] = generate_summary(verified_text)
    return state


def route_after_review(state: AgentState) -> str:
    return "generate_patient_summary" if state["review_approved"] else END


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("generate_soap_note", generate_soap_note)
    graph.add_node("extract_claims", extract_claims)
    graph.add_node("verify_claims", verify_claims)
    graph.add_node("human_review", human_review)
    graph.add_node("generate_patient_summary", generate_patient_summary)

    graph.set_entry_point("generate_soap_note")
    graph.add_edge("generate_soap_note", "extract_claims")
    graph.add_edge("extract_claims", "verify_claims")
    graph.add_edge("verify_claims", "human_review")
    graph.add_conditional_edges("human_review", route_after_review)
    graph.add_edge("generate_patient_summary", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["human_review"])

from typing import TypedDict


class Claim(TypedDict):
    text: str
    source_section: str  # SOAP 노트 내 위치 (S/O/A/P)
    entailment_score: float
    verified: bool


class AgentState(TypedDict):
    dialogue: str
    soap_note: str
    claims: list[Claim]
    weak_claims: list[Claim]
    review_approved: bool
    patient_summary: str

"""
3주차: NLI 근거검증 임계값(ENTAILMENT_THRESHOLD) 튜닝.

ACI-Bench valid.csv(학습에 안 쓴 held-out 셋)로 라벨링된 테스트셋을 자동 생성:
- TRUE: 노트[i]의 문장을 그 대화[i]에 대고 검증 (실제로 근거 있음)
- FALSE: 노트[j]의 문장을 다른 대화[i]에 대고 검증 (다른 환자 얘기라 근거 없음 보장)

각 임계값 후보에서 accuracy/precision/recall을 계산해 최적값을 찾는다.

사용법: python -m scripts.tune_threshold
"""

import csv
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.verify import entailment_score

VALID_CSV = Path(__file__).resolve().parent.parent / "data/raw/aci-bench/data/challenge_data/valid.csv"
N_ENCOUNTERS = 12  # 몇 개 대화-노트 쌍을 쓸지
SENTENCES_PER_NOTE = 2  # 노트당 몇 문장을 claim으로 뽑을지


def split_sentences(note: str) -> list[str]:
    # 섹션 헤더(전부 대문자 줄)나 빈 줄은 제외하고, 문장 단위로 분리
    lines = [l.strip() for l in note.splitlines() if l.strip()]
    lines = [l for l in lines if not l.isupper()]
    text = " ".join(lines)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def build_test_set(rows: list[dict]) -> list[dict]:
    examples = []
    for i, row in enumerate(rows):
        sentences = split_sentences(row["note"])[:SENTENCES_PER_NOTE]
        for s in sentences:
            examples.append({"dialogue": row["dialogue"], "claim": s, "label": True})

        # false: 다른 encounter의 노트 문장을 이 대화에 대고 검증
        j = (i + 1) % len(rows)
        false_sentences = split_sentences(rows[j]["note"])[:SENTENCES_PER_NOTE]
        for s in false_sentences:
            examples.append({"dialogue": row["dialogue"], "claim": s, "label": False})
    return examples


def main() -> None:
    with open(VALID_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    random.seed(42)
    random.shuffle(rows)
    rows = rows[:N_ENCOUNTERS]

    test_set = build_test_set(rows)
    print(f"test set: {len(test_set)} examples ({sum(e['label'] for e in test_set)} true / "
          f"{sum(not e['label'] for e in test_set)} false)")

    for ex in test_set:
        ex["score"] = entailment_score(ex["dialogue"], ex["claim"])

    true_scores = sorted(e["score"] for e in test_set if e["label"])
    false_scores = sorted(e["score"] for e in test_set if not e["label"])
    print(f"\nTRUE  scores: min={min(true_scores):.3f} median={true_scores[len(true_scores)//2]:.3f} max={max(true_scores):.3f}")
    print(f"FALSE scores: min={min(false_scores):.3f} median={false_scores[len(false_scores)//2]:.3f} max={max(false_scores):.3f}")

    print("\nthreshold | accuracy | precision | recall | false_pos | false_neg")
    best = None
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp = sum(1 for e in test_set if e["label"] and e["score"] >= t)
        fp = sum(1 for e in test_set if not e["label"] and e["score"] >= t)
        tn = sum(1 for e in test_set if not e["label"] and e["score"] < t)
        fn = sum(1 for e in test_set if e["label"] and e["score"] < t)
        acc = (tp + tn) / len(test_set)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"{t:>9.1f} | {acc:>8.3f} | {precision:>9.3f} | {recall:>6.3f} | {fp:>9} | {fn:>9}")
        if best is None or acc > best[1]:
            best = (t, acc)

    print(f"\nbest threshold by accuracy: {best[0]} (accuracy={best[1]:.3f})")

    print("\n--- misclassified examples at threshold=0.5 ---")
    for e in test_set:
        predicted = e["score"] >= 0.5
        if predicted != e["label"]:
            print(f"[{'TRUE' if e['label'] else 'FALSE'} labeled, score={e['score']:.3f}] {e['claim'][:100]}")


if __name__ == "__main__":
    main()

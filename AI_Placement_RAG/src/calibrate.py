"""Pick SCORE_THRESHOLD from YOUR data (embedding calls only, no answer-generation quota).

    python -m src.calibrate

It prints the best FAISS distance of questions that ARE in the PDFs and of questions that are
NOT, then suggests a value in the gap between the two groups. Put it in .env as SCORE_THRESHOLD.
"""
from __future__ import annotations

import json

from . import config
from .pipelines import PlacementRAG


def main():
    config.get_api_key()
    questions = json.loads((config.ROOT / "evaluation" / "questions.json").read_text())
    rag = PlacementRAG()
    inside, outside = {}, {}
    for q in questions:
        best = rag.vector_search("faiss", q["question"], 1)[0][1]
        (inside if q["answerable"] else outside)[q["question"]] = best

    for title, group in (("IN the PDFs (should be answered)", inside),
                         ("NOT in the PDFs (should be refused)", outside)):
        print(f"\n{title}")
        for q, s in sorted(group.items(), key=lambda x: x[1]):
            print(f"  {s:.3f}  {q}")
    worst_in, best_out = max(inside.values()), min(outside.values())
    print(f"\nLargest in-scope distance : {worst_in:.3f}")
    print(f"Smallest out-of-scope one : {best_out:.3f}")
    if worst_in < best_out:
        print(f"\nGroups are separated. Suggested SCORE_THRESHOLD = {(worst_in + best_out) / 2:.2f}")
    else:
        print("\nGroups overlap: no perfect threshold. V4's reranker handles this better; "
              f"keep SCORE_THRESHOLD near {min(worst_in, best_out) + 0.05:.2f} for V2/V3.")


if __name__ == "__main__":
    main()

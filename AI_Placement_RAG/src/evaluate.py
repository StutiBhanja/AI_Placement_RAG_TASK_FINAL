"""Evaluation: compare V2 (FAISS), V3 (ChromaDB) and V4 (Advanced) on 30 questions.

Run from the project root:

    python -m src.evaluate                       # V2, V3, V4 with Gemini (uses API quota!)
    python -m src.evaluate --versions v2 v3      # only some versions
    python -m src.evaluate --no-llm              # retrieval only: no answer-generation quota
    python -m src.evaluate --delay 5 --resume    # slow down + continue an interrupted run

What is measured (matches section 6 of the project document)
  * Answer accuracy   : answerable question -> Gemini answered AND at least half of the
                        expected keywords appear in the answer.
                        unanswerable question -> the system correctly refused.
  * Retrieval quality : "hit rate" = an expected (file, page) is among the sources/chunks.
  * Response time     : average seconds per question.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from . import config
from .pipelines import PlacementRAG

EVAL_DIR = config.ROOT / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"


def keyword_score(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    low = answer.lower()
    return sum(k.lower() in low for k in keywords) / len(keywords)


def retrieval_hit(sources, q) -> bool:
    wanted = {(q["expected_file"], p) for p in q["expected_pages"]}
    return any((s.file_name, s.page) in wanted for s in sources)


def run(versions: list[str], use_llm: bool, delay: float, resume: bool, limit: int | None):
    questions = json.loads((EVAL_DIR / "questions.json").read_text())
    if limit:
        questions = questions[:limit]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / ("raw_results_no_llm.json" if not use_llm else "raw_results.json")
    rows: list[dict] = json.loads(raw_path.read_text()) if (resume and raw_path.exists()) else []
    done = {(r["version"], r["id"]) for r in rows}

    rag = PlacementRAG()
    for version in versions:
        for q in questions:
            if (version, q["id"]) in done:
                continue
            row = {"version": version, "id": q["id"], "question": q["question"],
                   "answerable": q["answerable"], "category": q["category"]}
            t0 = time.perf_counter()
            if use_llm:
                res = rag.ask(q["question"], version=version, use_cache=False)
                row.update(latency_s=res.latency_s, answer=res.answer, error=res.error or "",
                           refused=not res.found and res.error is None,
                           sources="; ".join(s.label for s in res.sources))
                if res.error:                      # Gemini failed: do not count it as right or wrong
                    row.update(correct=None, hit=None, kw=None)
                else:
                    kw = keyword_score(res.answer, q["expected_keywords"])
                    if q["answerable"]:
                        row.update(kw=round(kw, 2), correct=bool(res.found and kw >= 0.5),
                                   hit=retrieval_hit(res.sources, q) if res.found else False)
                    else:
                        row.update(kw=None, correct=bool(not res.found), hit=None)
                if delay:
                    time.sleep(delay)
            else:
                sources = rag.retrieve_only(q["question"], version)
                dt = round(time.perf_counter() - t0, 3)
                best = sources[0].score if sources and sources[0].score is not None else None
                refused = bool(version in ("v2", "v3") and (best is None or best > config.SCORE_THRESHOLD))
                row.update(latency_s=dt, answer="", error="", refused=refused,
                           sources="; ".join(s.label for s in sources), kw=None,
                           hit=retrieval_hit(sources, q) if q["answerable"] else None,
                           # only V2/V3 can refuse without Gemini; V1/V4 need the LLM to judge this
                           correct=(refused != q["answerable"]) if version in ("v2", "v3") else None)
            rows.append(row)
            raw_path.write_text(json.dumps(rows, indent=1))          # save after every question
            print(f"{version} {q['id']} correct={row['correct']} hit={row['hit']} {row['latency_s']}s")
    return rows


def summarise(rows: list[dict], versions: list[str]) -> str:
    def pct(a, b):
        return "n/a" if not b else f"{100 * a / b:.0f}% ({a}/{b})"

    lines = ["| Version | Answer accuracy | Retrieval hit rate | Correct refusals | Avg time (s) | Gemini errors |",
             "|---|---|---|---|---|---|"]
    for v in versions:
        vr = [r for r in rows if r["version"] == v]
        if not vr:
            continue
        ans = [r for r in vr if r["answerable"] and r["correct"] is not None]
        una = [r for r in vr if not r["answerable"] and r["correct"] is not None]
        hits = [r for r in vr if r["answerable"] and r["hit"] is not None]
        errs = sum(1 for r in vr if r["error"])
        avg = sum(r["latency_s"] for r in vr) / len(vr)
        lines.append(f"| {v.upper()} | {pct(sum(r['correct'] for r in ans), len(ans))} "
                     f"| {pct(sum(r['hit'] for r in hits), len(hits))} "
                     f"| {pct(sum(r['correct'] for r in una), len(una))} | {avg:.2f} | {errs} |")
    return "\n".join(lines)


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=["v2", "v3", "v4"], choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--no-llm", action="store_true", help="retrieval only (no answer generation)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between questions")
    ap.add_argument("--resume", action="store_true", help="skip questions already saved")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    a = ap.parse_args()

    config.get_api_key()
    config.setup_langsmith()
    rows = run(a.versions, not a.no_llm, a.delay, a.resume, a.limit)
    suffix = "_no_llm" if a.no_llm else ""
    save_csv(rows, RESULTS_DIR / f"results{suffix}.csv")
    table = summarise(rows, a.versions)
    (RESULTS_DIR / f"summary{suffix}.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nSaved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()

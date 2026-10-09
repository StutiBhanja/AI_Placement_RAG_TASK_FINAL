"""Roadmap 5 + Advanced RAG building blocks (V4).

* BM25            -> keyword search
* RRF             -> merges several ranked lists into one
* rewrite_queries -> LLM turns one casual question into better search queries
* rerank          -> LLM scores each retrieved chunk 0-10 for the question
"""
from __future__ import annotations

import json
import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import config
from .llm import LLMUnavailable, safe_invoke

try:  # LangSmith is optional
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*_a, **_k):
        def deco(fn):
            return fn
        return deco

# --------------------------------------------------------------------------- #
# Keyword search (BM25)
# --------------------------------------------------------------------------- #
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or",
    "do", "does", "how", "what", "why", "which", "can", "should", "i", "my", "me", "you", "your",
    "it", "be", "with", "as", "at", "by", "from", "that", "this",
}


def tokenize(text: str) -> list[str]:
    """Lower-case words without punctuation ('method?' -> 'method') and without filler words."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


class BM25Index:
    """Small wrapper around rank_bm25 that returns Documents (optionally one category only)."""

    def __init__(self, chunks: list[Document]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.page_content) or ["_"] for c in chunks])

    def search(self, query: str, k: int = 10, category: str | None = None) -> list[Document]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break                                   # no keyword overlap at all
            if category and self.chunks[i].metadata.get("category") != category:
                continue
            out.append(self.chunks[i])
            if len(out) == k:
                break
        return out


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def doc_key(doc: Document) -> tuple:
    return (doc.metadata.get("file_name"), doc.metadata.get("page"), doc.page_content)


def reciprocal_rank_fusion(result_lists: list[list[Document]], c: int = 60) -> list[Document]:
    """score(doc) = sum over lists of 1 / (c + rank). Uses ranks, not raw scores."""
    scores: dict[tuple, float] = {}
    docs: dict[tuple, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank + 1)
            docs[key] = doc
    return [docs[k] for k in sorted(scores, key=scores.get, reverse=True)]


# --------------------------------------------------------------------------- #
# Query rewriting
# --------------------------------------------------------------------------- #
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the student's question in up to 3 different ways to help search
placement-preparation notes (technical interview, aptitude, HR questions, resume guidelines).

Use clear wording and likely keywords. Keep the same meaning.
Output ONLY the rewritten questions, one per line, with no numbering.

Question: {question}"""
)


@traceable(name="rewrite_queries", run_type="chain")
def rewrite_queries(question: str, llm) -> tuple[list[str], str | None]:
    """Return ([original, rewrite1, ...], warning). Falls back to [original] if Gemini fails."""
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    try:
        text = safe_invoke(chain, {"question": question})
    except LLMUnavailable as err:
        return [question], f"Query rewriting skipped: {err.kind}"
    rewrites = []
    for line in text.splitlines():
        cleaned = re.sub(r"^[\s\-•*\d\.)]+", "", line).strip()
        if cleaned and cleaned.lower() != question.lower():
            rewrites.append(cleaned)
    return [question] + rewrites[:3], None


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #
RERANK_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict relevance judge.

Question: {question}

Below are numbered text chunks. Give each chunk a relevance score from 0 to 10.
10 = directly answers the question.
0 = unrelated.

Return ONLY a JSON list of numbers, one score per chunk, in order.
Example: [8, 2, 0, 9]

Chunks:
{chunks}"""
)


def parse_scores(raw: str, expected: int) -> list[float] | None:
    match = re.search(r"\[[\s\S]*?\]", raw or "")
    if not match:
        return None
    try:
        scores = [float(s) for s in json.loads(match.group())]
    except (ValueError, TypeError):
        return None
    return scores if len(scores) == expected else None


def rerank(question: str, docs: list[Document], llm, top_n: int | None = None
           ) -> tuple[list[tuple[Document, float]], bool]:
    """Return ([(doc, score), ...] best first, used_llm).

    used_llm is False when Gemini was unavailable or answered badly; the original
    retrieval order is then kept (every chunk gets the neutral score 5.0).
    """
    top_n = top_n or config.V4_RERANK_TOP_N
    if not docs:
        return [], True
    numbered = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs))
    chain = RERANK_PROMPT | llm | StrOutputParser()
    try:
        raw = safe_invoke(chain, {"question": question, "chunks": numbered})
    except LLMUnavailable:
        return [(d, 5.0) for d in docs[:top_n]], False
    scores = parse_scores(raw, len(docs))
    if scores is None:
        return [(d, 5.0) for d in docs[:top_n]], False
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)   # stable sort
    return ranked[:top_n], True

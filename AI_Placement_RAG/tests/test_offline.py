"""Offline tests: no Gemini key and no internet needed.

A small "hash" embedding model and a fake LLM replace Gemini, so we can check that the
whole pipeline (loading -> chunking -> stores -> V1..V4 -> sources -> error handling) works.

Run:  pytest -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, ingest
from src.llm import LLMUnavailable, classify_error, safe_invoke
from src.pipelines import PlacementRAG
from src.retrieval import BM25Index, parse_scores, reciprocal_rank_fusion, tokenize


class HashEmbeddings(Embeddings):
    """Bag-of-words hashing embeddings: similar words -> similar vectors. Deterministic."""

    dim = 512

    def __init__(self):
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for t in tokenize(text):
            v[hash_word(t) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def hash_word(w: str) -> int:           # python's hash() is randomised per run -> use our own
    h = 0
    for ch in w:
        h = (h * 131 + ord(ch)) % 1_000_003
    return h


def fake_llm_fn(prompt_value):
    text = prompt_value.to_string()
    if "strict relevance judge" in text:
        question = re.search(r"Question: (.*)", text).group(1)
        q_tokens = set(tokenize(question))
        chunks = re.split(r"\n\n\[\d+\] ", "\n\n" + text.split("Chunks:\n", 1)[1])[1:]
        scores = [9 if len(q_tokens & set(tokenize(c))) >= 2 else 0 for c in chunks]
        return str(scores)
    if "Rewrite the student's question" in text:
        return "1. explain the star method for behavioural interview answers\n- situation task action result"
    context = text.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    return "ANSWER: " + " ".join(context.split())[:150]


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    base = tmp_path_factory.mktemp("stores")
    config.SCORE_THRESHOLD = 1.2        # hash vectors have their own distance scale
    return PlacementRAG(embeddings=HashEmbeddings(), llm=RunnableLambda(fake_llm_fn),
                        faiss_dir=base / "faiss", chroma_dir=base / "chroma")


# ---------------------------------------------------------------- loading / chunking
def test_pdfs_load_with_metadata():
    pages = ingest.load_pages()
    assert len(pages) == 15
    cats = {p.metadata["category"] for p in pages}
    assert cats == {"technical", "resume", "hr", "aptitude"}
    m = pages[0].metadata
    assert m["file_name"] == "technical_interview_.pdf" and m["page_number"] == m["page"] + 1
    assert "producer" not in m                       # noisy metadata removed


def test_chunking_keeps_metadata():
    chunks = ingest.load_chunks()
    assert 40 < len(chunks) < 80
    assert all(c.metadata["category"] and c.metadata["page_number"] for c in chunks)


def test_missing_pdf_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/ folder"):
        ingest.load_pages(tmp_path)


# ---------------------------------------------------------------- retrieval pieces
def test_rrf_prefers_docs_found_by_both_lists():
    a, b, c = (Document(page_content=x, metadata={"file_name": "f", "page": 0}) for x in "abc")
    merged = reciprocal_rank_fusion([[a, b], [b, c]])
    assert merged[0].page_content == "b"


def test_bm25_category_filter_and_keywords():
    idx = BM25Index(ingest.load_chunks())
    assert idx.search("STAR method situation task action result", k=3)[0].metadata["category"] == "hr"
    assert all(d.metadata["category"] == "aptitude" for d in idx.search("interest", k=5, category="aptitude"))
    assert idx.search("zzzzqqq", k=3) == []


def test_parse_scores():
    assert parse_scores("Here: [8, 2, 0]", 3) == [8.0, 2.0, 0.0]
    assert parse_scores("[8, 2]", 3) is None
    assert parse_scores("no list", 1) is None


def test_error_classification():
    assert classify_error(Exception("429 RESOURCE_EXHAUSTED")) == "quota"
    assert classify_error(Exception("503 UNAVAILABLE high demand")) == "busy"
    assert classify_error(Exception("API key not valid")) == "auth"


# ---------------------------------------------------------------- the four versions
@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_in_scope_question_has_answer_and_sources(rag, version):
    res = rag.ask("What is the STAR method? Situation Task Action Result", version=version, use_cache=False)
    assert res.found and res.answer.startswith("ANSWER")
    assert any(s.file_name == "hr_questions.pdf" and s.page == 2 for s in res.sources)


@pytest.mark.parametrize("version", ["v2", "v3", "v4"])
def test_out_of_scope_question_is_refused(rag, version):
    res = rag.ask("Who won the cricket world cup in Australia?", version=version, use_cache=False)
    assert not res.found and res.answer == config.NOT_FOUND and res.sources == []


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_category_filter(rag, version):
    res = rag.ask("What is interest and how is it calculated?", version=version, category="aptitude",
                  use_cache=False)
    assert all(s.category == "aptitude" for s in res.sources)


def test_v4_uses_rewrites_and_reports_scores(rag):
    res = rag.ask("How do I answer behavioural questions using STAR?", version="v4", use_cache=False)
    assert len(res.queries) > 1 and res.queries[0].startswith("How do I answer")
    assert res.found and all(s.score is not None for s in res.sources)


def test_cache_and_empty_question(rag):
    r1 = rag.ask("What is a confusion matrix?", version="v2")
    r2 = rag.ask("what is a confusion matrix?", version="v2")
    assert not r1.cached and r2.cached
    assert rag.ask("   ").answer == "Please type a question."
    with pytest.raises(ValueError):
        rag.ask("x", version="v9")


def test_saved_indexes_are_reused_without_new_embedding_calls(tmp_path):
    emb = HashEmbeddings()
    kw = dict(embeddings=emb, llm=RunnableLambda(fake_llm_fn),
              faiss_dir=tmp_path / "f", chroma_dir=tmp_path / "c")
    a = PlacementRAG(**kw)
    a.store("faiss"); a.store("chroma")
    first = emb.calls
    b = PlacementRAG(**kw)
    b.store("faiss"); b.store("chroma")
    assert emb.calls == first                        # nothing embedded the second time
    assert b.store("chroma")._collection.count() == len(b.chunks)


# ---------------------------------------------------------------- Gemini failures
def _failing_llm(message):
    def boom(_):
        raise RuntimeError(message)
    return RunnableLambda(boom)


def test_quota_error_does_not_crash_and_still_returns_sources(tmp_path):
    r = PlacementRAG(embeddings=HashEmbeddings(), llm=_failing_llm("429 RESOURCE_EXHAUSTED quota"),
                     faiss_dir=tmp_path / "f", chroma_dir=tmp_path / "c")
    res = r.ask("What is the STAR method? Situation Task Action Result", version="v2")
    assert res.error and res.error.startswith("quota") and not res.found
    assert res.sources                                # retrieved passages are still shown


def test_v4_falls_back_when_rewrite_and_rerank_fail(tmp_path):
    r = PlacementRAG(embeddings=HashEmbeddings(), llm=_failing_llm("429 RESOURCE_EXHAUSTED"),
                     faiss_dir=tmp_path / "f", chroma_dir=tmp_path / "c")
    res = r.ask("What is the STAR method? Situation Task Action Result", version="v4")
    assert any("rewriting skipped" in n.lower() for n in res.notes)
    assert any("reranker unavailable" in n.lower() for n in res.notes)
    assert res.error and res.sources                  # answer step failed, passages still returned


def test_safe_invoke_raises_friendly_exception():
    with pytest.raises(LLMUnavailable) as e:
        safe_invoke(_failing_llm("API key not valid"), {})
    assert e.value.kind == "auth"

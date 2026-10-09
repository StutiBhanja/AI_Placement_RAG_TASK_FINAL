"""All four RAG versions behind one class, `PlacementRAG`.

    V1  Simple RAG        in-memory vector store, no refusal threshold
    V2  FAISS RAG         saved FAISS index + distance threshold
    V3  ChromaDB RAG      persistent Chroma + metadata (category) filter + threshold
    V4  Advanced RAG      query rewriting + FAISS + BM25 + RRF + LLM reranking + relevance gate

Every version returns the same `RAGResult`, so the Streamlit app and the evaluation
script treat them identically. All steps are traced in LangSmith when it is enabled.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import config, ingest, stores
from .llm import LLMUnavailable, make_embeddings, make_llm, safe_invoke
from .retrieval import BM25Index, reciprocal_rank_fusion, rerank, rewrite_queries

try:  # LangSmith is optional: without it the decorator does nothing
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*_a, **_k):
        def deco(fn):
            return fn
        return deco

VERSIONS = {
    "v1": "V1 - Simple RAG (in-memory vector store)",
    "v2": "V2 - FAISS RAG (saved index + score threshold)",
    "v3": "V3 - ChromaDB RAG (metadata filter + score threshold)",
    "v4": "V4 - Advanced RAG (rewrite + hybrid + RRF + rerank)",
}

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI Placement Preparation Assistant.

Answer the question using ONLY the context below.
If the answer is not supported by the context, reply exactly:
"{not_found}"
Do not use outside knowledge. Keep the answer clear and beginner-friendly.

Context:
{context}

Question: {question}

Answer:"""
)


@dataclass
class Source:
    file_name: str
    page: int                 # 1-based
    category: str
    snippet: str
    score: float | None = None   # distance (V1-V3) or reranker score 0-10 (V4)

    @property
    def label(self) -> str:
        return f"{self.file_name} - page {self.page}"


@dataclass
class RAGResult:
    question: str
    version: str
    answer: str
    found: bool = False
    sources: list[Source] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)     # V4: original + rewrites
    latency_s: float = 0.0
    error: str | None = None                              # set when Gemini was unavailable
    notes: list[str] = field(default_factory=list)        # warnings / fallbacks used
    best_distance: float | None = None
    cached: bool = False


# --------------------------------------------------------------------------- #
def to_source(doc: Document, score: float | None = None) -> Source:
    m = doc.metadata
    return Source(
        file_name=m.get("file_name", m.get("source", "unknown")),
        page=int(m.get("page_number", int(m.get("page", 0)) + 1)),
        category=m.get("category", "other"),
        snippet=" ".join(doc.page_content.split())[:220],
        score=score,
    )


def unique_sources(sources: list[Source]) -> list[Source]:
    seen, out = set(), []
    for s in sources:
        if (s.file_name, s.page) not in seen:
            seen.add((s.file_name, s.page))
            out.append(s)
    return out


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


class PlacementRAG:
    """Loads the PDFs once, builds the indexes lazily and answers questions."""

    def __init__(self, embeddings=None, llm=None, data_dir: Path | None = None,
                 faiss_dir: Path | None = None, chroma_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.faiss_dir = faiss_dir
        self.chroma_dir = chroma_dir
        self._embeddings = embeddings
        self._llm = llm
        self._chunks: list[Document] | None = None
        self._stores: dict[str, object] = {}
        self._bm25: BM25Index | None = None
        self._cache: dict[tuple, RAGResult] = {}

    # ---- lazy resources ---------------------------------------------------- #
    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = make_embeddings()
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            self._llm = make_llm()
        return self._llm

    @property
    def chunks(self) -> list[Document]:
        if self._chunks is None:
            self._chunks = ingest.load_chunks(self.data_dir)
        return self._chunks

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.chunks)
        return self._bm25

    def store(self, kind: str):
        """kind: 'memory' | 'faiss' | 'chroma'. Built (or loaded from disk) on first use."""
        if kind not in self._stores:
            fp = ingest.fingerprint(self.data_dir)
            if kind == "memory":
                self._stores[kind] = stores.build_inmemory(self.chunks, self.embeddings)
            elif kind == "faiss":
                self._stores[kind] = stores.build_faiss(self.chunks, self.embeddings, fp, self.faiss_dir)
            elif kind == "chroma":
                self._stores[kind] = stores.build_chroma(self.chunks, self.embeddings, fp, self.chroma_dir)
            else:
                raise ValueError(f"unknown store {kind}")
        return self._stores[kind]

    def rebuild_indexes(self) -> None:
        """Delete saved indexes and re-embed everything (use after changing the PDFs)."""
        fp = ingest.fingerprint(self.data_dir)
        self._stores.clear()
        self._stores["faiss"] = stores.build_faiss(self.chunks, self.embeddings, fp, self.faiss_dir, force=True)
        self._stores["chroma"] = stores.build_chroma(self.chunks, self.embeddings, fp, self.chroma_dir, force=True)
        self._cache.clear()

    # ---- vector search (returns (doc, distance) smaller = closer) ---------- #
    @traceable(name="vector_search", run_type="retriever")
    def vector_search(self, kind: str, query: str, k: int, category: str | None = None
                      ) -> list[tuple[Document, float]]:
        store = self.store(kind)
        if kind == "memory":
            # InMemoryVectorStore returns similarity (bigger = closer); convert so that
            # "smaller = closer" holds for every store.
            flt = (lambda d: d.metadata.get("category") == category) if category else None
            res = store.similarity_search_with_score(query, k=k, filter=flt)
            return [(d, 1.0 - float(s)) for d, s in res]
        flt = {"category": category} if category else None
        res = store.similarity_search_with_score(query, k=k, filter=flt)
        return [(d, float(s)) for d, s in res]

    @traceable(name="keyword_search_bm25", run_type="retriever")
    def keyword_search(self, query: str, k: int, category: str | None = None) -> list[Document]:
        return self.bm25.search(query, k=k, category=category)

    # ---- answer generation -------------------------------------------------- #
    @traceable(name="generate_answer", run_type="chain")
    def generate(self, question: str, docs: list[Document]) -> str:
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        return safe_invoke(chain, {"context": format_context(docs), "question": question,
                                   "not_found": config.NOT_FOUND}).strip()

    def _finish(self, res: RAGResult, question: str, docs: list[Document],
                scores: list[float | None]) -> RAGResult:
        """Common last step: call Gemini, then attach sources."""
        try:
            res.answer = self.generate(question, docs)
        except LLMUnavailable as err:
            res.error = f"{err.kind}: {err.message}"
            res.answer = ("I found relevant notes but could not generate the answer because Gemini "
                          "is unavailable right now. The matching passages are listed below.")
            res.sources = unique_sources([to_source(d, s) for d, s in zip(docs, scores)])
            return res
        res.found = config.NOT_FOUND not in res.answer
        if res.found:
            res.sources = unique_sources([to_source(d, s) for d, s in zip(docs, scores)])
        return res

    # ---- the four versions --------------------------------------------------- #
    def _ask_simple(self, res: RAGResult, question: str, kind: str, category, threshold: bool) -> RAGResult:
        hits = self.vector_search(kind, question, config.TOP_K, category)
        if not hits:
            res.answer = config.NOT_FOUND
            return res
        res.best_distance = hits[0][1]
        if threshold:
            hits = [(d, s) for d, s in hits if s <= config.SCORE_THRESHOLD]
            if not hits:                      # weak match -> refuse without calling Gemini
                res.answer = config.NOT_FOUND
                res.notes.append(f"No chunk closer than the threshold ({config.SCORE_THRESHOLD}); "
                                 "Gemini was not called.")
                return res
        return self._finish(res, question, [d for d, _ in hits], [s for _, s in hits])

    def _ask_v4(self, res: RAGResult, question: str, category) -> RAGResult:
        # 1. query rewriting
        if config.V4_USE_REWRITE:
            queries, warn = rewrite_queries(question, self.llm)
            if warn:
                res.notes.append(warn)
        else:
            queries = [question]
        res.queries = queries

        # 2. hybrid retrieval (vector + keyword) for every query
        result_lists: list[list[Document]] = []
        for i, q in enumerate(queries):
            vec = self.vector_search("faiss", q, config.V4_RETRIEVE_K, category)
            if i == 0 and vec:
                res.best_distance = vec[0][1]
            result_lists.append([d for d, _ in vec])
            result_lists.append(self.keyword_search(q, config.V4_RETRIEVE_K, category))

        # 3. merge the rankings with RRF
        candidates = reciprocal_rank_fusion(result_lists)[: config.V4_CANDIDATES]
        if not candidates:
            res.answer = config.NOT_FOUND
            return res

        # 4. rerank with Gemini (scores 0-10)
        ranked, used_llm = self._rerank(question, candidates)

        # 5. relevance gate
        if used_llm:
            kept = [(d, s) for d, s in ranked if s >= config.V4_MIN_RELEVANCE]
        else:
            res.notes.append("Reranker unavailable: kept retrieval order and used the "
                             "vector-distance threshold instead.")
            ok = res.best_distance is not None and res.best_distance <= config.SCORE_THRESHOLD
            kept = ranked if ok else []
        if not kept:
            res.answer = config.NOT_FOUND
            res.notes.append("No chunk was relevant enough; Gemini was not asked to answer.")
            return res

        # 6. generate the answer from the best chunks
        return self._finish(res, question, [d for d, _ in kept],
                            [s if used_llm else None for _, s in kept])

    def retrieve_only(self, question: str, version: str, category: str | None = None) -> list[Source]:
        """Retrieval without any Gemini call (used by `evaluate.py --no-llm`).

        V4 here = hybrid (FAISS + BM25) + RRF only; rewriting and reranking need Gemini.
        """
        version = version.lower()
        if version == "v4":
            lists = [[d for d, _ in self.vector_search("faiss", question, config.V4_RETRIEVE_K, category)],
                     self.keyword_search(question, config.V4_RETRIEVE_K, category)]
            docs = reciprocal_rank_fusion(lists)[: config.V4_RERANK_TOP_N]
            return [to_source(d) for d in docs]
        kind = {"v1": "memory", "v2": "faiss", "v3": "chroma"}[version]
        hits = self.vector_search(kind, question, config.TOP_K, category)
        return [to_source(d, s) for d, s in hits]

    @traceable(name="rerank", run_type="chain")
    def _rerank(self, question: str, candidates: list[Document]):
        return rerank(question, candidates, self.llm, config.V4_RERANK_TOP_N)

    @traceable(name="placement_rag", run_type="chain")
    def ask(self, question: str, version: str = "v4", category: str | None = None,
            use_cache: bool = True) -> RAGResult:
        version = version.lower()
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {list(VERSIONS)}")
        question = (question or "").strip()
        if not question:
            return RAGResult(question="", version=version, answer="Please type a question.")
        if category in ("", "all", "All", None):
            category = None

        key = (question.lower(), version, category)
        if use_cache and key in self._cache:
            cached = self._cache[key]
            return RAGResult(**{**cached.__dict__, "cached": True})

        res = RAGResult(question=question, version=version, answer="")
        t0 = time.perf_counter()
        if version == "v1":
            self._ask_simple(res, question, "memory", category, threshold=False)
        elif version == "v2":
            self._ask_simple(res, question, "faiss", category, threshold=True)
        elif version == "v3":
            self._ask_simple(res, question, "chroma", category, threshold=True)
        else:
            self._ask_v4(res, question, category)
        res.latency_s = round(time.perf_counter() - t0, 2)

        if res.error is None:                       # never cache a Gemini failure
            self._cache[key] = res
        return res

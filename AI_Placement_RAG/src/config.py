"""Central settings for the AI Placement Preparation Assistant.

Everything you may want to change lives here (or in the .env file),
so no other file needs editing.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional: .env is simply ignored if it is missing
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STORE_DIR = ROOT / "vectorstores"          # created automatically
FAISS_DIR = STORE_DIR / "faiss_index"
CHROMA_DIR = STORE_DIR / "chroma_db"
COLLECTION_NAME = "placement_notes"

# The project scope is exactly these four documents / categories.
CATEGORY_BY_FILE = {
    "technical_interview_.pdf": "technical",
    "resume_guidelines.pdf": "resume",
    "hr_questions.pdf": "hr",
    "aptitude.pdf": "aptitude",
}
CATEGORIES = ["technical", "resume", "hr", "aptitude"]

# --------------------------------------------------------------------------- #
# Models (change in .env, not in the code)
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Free-tier quota is counted PER MODEL, so we try the models in this order.
LLM_MODELS = [
    m.strip()
    for m in os.getenv("LLM_MODELS", "gemini-3.5-flash-lite,gemini-3.6-flash").split(",")
    if m.strip()
]

# --------------------------------------------------------------------------- #
# RAG parameters
# --------------------------------------------------------------------------- #
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

TOP_K = int(os.getenv("TOP_K", "4"))                       # chunks sent to the LLM (V1-V3)

# FAISS / Chroma use a DISTANCE: smaller = more similar.
# In V2 we measured: questions inside the PDFs ~0.45-0.61, outside ~0.85-0.94,
# so 0.73 sits in the gap. Re-measure with `python -m src.calibrate` if you change PDFs/models.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.73"))

# Advanced RAG (V4)
V4_RETRIEVE_K = int(os.getenv("V4_RETRIEVE_K", "10"))      # per retriever, per query
V4_CANDIDATES = int(os.getenv("V4_CANDIDATES", "12"))      # after RRF  (task doc: top 10-20)
V4_RERANK_TOP_N = int(os.getenv("V4_RERANK_TOP_N", "4"))   # after reranking (task doc: top 3-5)
V4_MIN_RELEVANCE = float(os.getenv("V4_MIN_RELEVANCE", "5"))  # reranker score 0-10
V4_USE_REWRITE = os.getenv("V4_USE_REWRITE", "true").lower() == "true"

NOT_FOUND = "I could not find the answer in the uploaded documents."

# --------------------------------------------------------------------------- #
# API key + LangSmith
# --------------------------------------------------------------------------- #
def get_api_key() -> str | None:
    """Return the Gemini key from the environment (GOOGLE_API_KEY or GEMINI_API_KEY)."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("Gemini_API")
    if key:
        os.environ["GOOGLE_API_KEY"] = key  # the name LangChain's Google classes read
    return key


def setup_langsmith() -> bool:
    """Turn LangSmith tracing on when LANGSMITH_API_KEY is present. Returns True if enabled."""
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    wanted = os.getenv("LANGSMITH_TRACING", "true").lower() != "false"
    if key and wanted:
        os.environ["LANGSMITH_API_KEY"] = key
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"           # older LangChain versions
        os.environ.setdefault("LANGSMITH_PROJECT", "AI-Placement-RAG")
        os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])
        return True
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    return False

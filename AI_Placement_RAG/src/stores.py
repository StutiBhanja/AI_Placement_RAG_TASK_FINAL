"""Roadmap 4: Vector databases (FAISS for V2, ChromaDB for V3, in-memory for V1).

Both persistent stores are saved to `vectorstores/` together with a manifest holding a
fingerprint of the PDFs + settings. If you change a PDF, the chunk size or the embedding
model, the index is rebuilt automatically - otherwise it is simply loaded (no new API calls).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_core.documents import Document

from . import config

MANIFEST = "manifest.json"


def _manifest_ok(folder: Path, fp: str) -> bool:
    try:
        return json.loads((folder / MANIFEST).read_text())["fingerprint"] == fp
    except Exception:  # noqa: BLE001
        return False


def _write_manifest(folder: Path, fp: str, n_chunks: int) -> None:
    (folder / MANIFEST).write_text(json.dumps({"fingerprint": fp, "chunks": n_chunks}))


def build_inmemory(chunks: list[Document], embeddings):
    """V1: vectors live in RAM only (lost when the program stops)."""
    from langchain_core.vectorstores import InMemoryVectorStore

    return InMemoryVectorStore.from_documents(chunks, embeddings)


def build_faiss(chunks: list[Document], embeddings, fp: str, folder: Path | None = None,
                force: bool = False):
    """V2: FAISS index saved to disk (index.faiss + index.pkl)."""
    from langchain_community.vectorstores import FAISS

    folder = Path(folder or config.FAISS_DIR)
    if not force and (folder / "index.faiss").exists() and _manifest_ok(folder, fp):
        # allow_dangerous_deserialization is safe here: we created this file ourselves.
        return FAISS.load_local(str(folder), embeddings, allow_dangerous_deserialization=True)

    store = FAISS.from_documents(chunks, embeddings)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    store.save_local(str(folder))
    _write_manifest(folder, fp, len(chunks))
    return store


def build_chroma(chunks: list[Document], embeddings, fp: str, folder: Path | None = None,
                 force: bool = False):
    """V3: ChromaDB persisted on disk (vectors + text + metadata, with metadata filtering)."""
    from langchain_chroma import Chroma

    folder = Path(folder or config.CHROMA_DIR)
    if not force and folder.exists() and _manifest_ok(folder, fp):
        store = Chroma(collection_name=config.COLLECTION_NAME, embedding_function=embeddings,
                       persist_directory=str(folder))
        if store._collection.count() == len(chunks):
            return store

    # Start clean: running from_documents twice on the same folder would store duplicates.
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        collection_name=config.COLLECTION_NAME, persist_directory=str(folder),
    )
    _write_manifest(folder, fp, len(chunks))
    return store

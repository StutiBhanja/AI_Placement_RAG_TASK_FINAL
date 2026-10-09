"""Roadmap 1 + 2: Document Loading and Text Splitting (chunking)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def list_pdfs(data_dir: Path | None = None) -> list[Path]:
    data_dir = Path(data_dir or config.DATA_DIR)
    return [data_dir / name for name in config.CATEGORY_BY_FILE]


def check_pdfs(data_dir: Path | None = None) -> None:
    missing = [p.name for p in list_pdfs(data_dir) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "These PDFs were not found in the data/ folder: " + ", ".join(missing)
            + ". Put all 4 placement PDFs inside the data/ folder and try again."
        )


def load_pages(data_dir: Path | None = None) -> list[Document]:
    """One Document per PDF page, tagged with category + clean metadata."""
    from langchain_community.document_loaders import PyPDFLoader

    check_pdfs(data_dir)
    pages: list[Document] = []
    for path in list_pdfs(data_dir):
        for doc in PyPDFLoader(str(path)).load():
            if not doc.page_content.strip():
                # A page with no selectable text (scanned image) needs OCR; skip it.
                print(f"[ingest] skipped empty page {doc.metadata.get('page', 0) + 1} of {path.name} "
                      "(no selectable text - OCR would be needed)")
                continue
            page = int(doc.metadata.get("page", 0))
            # Keep only the metadata we need (the PDF also carries producer/creator/dates = noise).
            meta = {
                "source": path.name,                 # file name only -> index works on any machine
                "file_name": path.name,
                "page": page,                        # 0-based, as PyPDFLoader gives it
                "page_number": page + 1,             # 1-based, what humans read
                "total_pages": int(doc.metadata.get("total_pages", 0)),
                "category": config.CATEGORY_BY_FILE.get(path.name, "other"),
            }
            pages.append(Document(page_content=doc.page_content, metadata=meta))
    if not pages:
        raise ValueError("No text could be extracted from the PDFs.")
    return pages


def split_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_documents(pages)


def load_chunks(data_dir: Path | None = None) -> list[Document]:
    return split_pages(load_pages(data_dir))


def fingerprint(data_dir: Path | None = None) -> str:
    """Hash of PDFs + chunking + embedding settings. If it changes, saved indexes are rebuilt."""
    h = hashlib.sha256()
    for p in list_pdfs(data_dir):
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    h.update(json.dumps([config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.EMBEDDING_MODEL]).encode())
    return h.hexdigest()

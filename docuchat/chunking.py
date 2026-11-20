"""Document loading and chunking.

Documents are plain text or Markdown files. Each file is split into
overlapping word-based chunks so that a fact is never stranded on a chunk
boundary. Chunks carry their source file name and index so answers can
cite them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """One retrievable piece of a document."""

    text: str
    source: str
    index: int

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"[{self.source} chunk {self.index + 1}] {self.text}"


SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}


def load_documents(folder: str | Path) -> dict[str, str]:
    """Load every supported file in *folder* (non-recursive).

    Returns a mapping of file name -> file contents. Files that cannot be
    decoded as UTF-8, or cannot be read at all, are skipped.
    """
    folder = Path(folder)
    documents: dict[str, str] = {}
    if not folder.is_dir():
        return documents
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if text.strip():
            documents[path.name] = text
    return documents


def _clean_markdown(text: str) -> str:
    """Strip Markdown heading markers and code-fence lines.

    The words stay; only the markup goes, so quoted answer sentences read
    like plain sentences instead of carrying ``#`` and backticks.
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        lines.append(stripped)
    return "\n".join(lines)


def _paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines, collapsing whitespace."""
    parts = re.split(r"\n\s*\n", _clean_markdown(text))
    return [" ".join(p.split()) for p in parts if p.strip()]


def chunk_text(text: str, source: str, max_words: int = 120,
               overlap_words: int = 30) -> list[Chunk]:
    """Split *text* into overlapping chunks of at most *max_words* words.

    Paragraphs are the packing unit: a paragraph is never split across
    chunks unless it alone exceeds *max_words* (then it is cut on word
    boundaries). Consecutive chunks share *overlap_words* words of context.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be >= 0 and < max_words")
    if not text.strip():
        return []

    # Build word windows, paragraph by paragraph.
    words: list[str] = []
    for paragraph in _paragraphs(text):
        words.extend(paragraph.split())

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunks.append(Chunk(text=" ".join(chunk_words), source=source,
                            index=index))
        index += 1
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def chunk_documents(documents: dict[str, str], max_words: int = 120,
                    overlap_words: int = 30) -> list[Chunk]:
    """Chunk every document in *documents* (file name -> text)."""
    chunks: list[Chunk] = []
    for source in sorted(documents):
        chunks.extend(chunk_text(documents[source], source,
                                 max_words=max_words,
                                 overlap_words=overlap_words))
    return chunks

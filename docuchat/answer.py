"""Grounded answer generation.

The answer is *extractive*: it is built from sentences that actually
appear in the retrieved chunks, each labelled with its source document
and chunk number. If no retrieved chunk shares a meaningful term with
the question, the answer says so plainly instead of inventing one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chunking import Chunk
from .retrieval import BM25Index, SearchResult, tokenize

NOT_FOUND_MESSAGE = (
    "I couldn't find an answer to that in your documents. "
    "Try rephrasing the question or adding a document that covers it."
)

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# A retrieved chunk only counts as evidence if it shares at least this
# many distinct content terms with the question.
MIN_SHARED_TERMS = 1


@dataclass
class Answer:
    """A grounded answer plus the evidence it was built from."""

    text: str
    found: bool
    sources: list[SearchResult]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def _best_sentences(chunk: Chunk, query_terms: set[str],
                    limit: int = 2) -> list[str]:
    """The sentences in *chunk* that share the most terms with the query."""
    scored: list[tuple[int, int, str]] = []
    for position, sentence in enumerate(_sentences(chunk.text)):
        terms = set(tokenize(sentence))
        shared = len(terms & query_terms)
        if shared:
            scored.append((shared, -position, sentence))
    scored.sort(reverse=True)
    return [sentence for _, _, sentence in scored[:limit]]


def answer_question(index: BM25Index, question: str,
                    top_k: int = 3) -> Answer:
    """Answer *question* from the documents behind *index*.

    Returns an Answer whose text is either a short extractive answer with
    inline citations like ``[gardening.md chunk 2]`` or the not-found
    message when nothing in the documents matches.
    """
    if not question.strip():
        return Answer(text="Please ask a question first.", found=False,
                      sources=[])
    query_terms = set(tokenize(question))
    if not query_terms:
        return Answer(text=NOT_FOUND_MESSAGE, found=False, sources=[])

    results = index.search(question, top_k=top_k)
    if not results:
        return Answer(text=NOT_FOUND_MESSAGE, found=False, sources=[])

    # Evidence check: the best chunk must share at least one content term
    # with the question (guaranteed by BM25 > 0, but keep the rule explicit).
    top_terms = set(tokenize(results[0].chunk.text))
    if not (top_terms & query_terms):
        return Answer(text=NOT_FOUND_MESSAGE, found=False, sources=[])

    lines: list[str] = []
    used_sources: list[SearchResult] = []
    for result in results:
        sentences = _best_sentences(result.chunk, query_terms)
        if not sentences:
            continue
        citation = f"[{result.chunk.source} chunk {result.chunk.index + 1}]"
        for sentence in sentences:
            lines.append(f"{sentence} {citation}")
        used_sources.append(result)
        if len(lines) >= 3:
            break

    if not lines:
        return Answer(text=NOT_FOUND_MESSAGE, found=False, sources=[])
    return Answer(text=" ".join(lines), found=True, sources=used_sources)


def build_index(documents: dict[str, str], max_words: int = 120,
                overlap_words: int = 30) -> tuple[BM25Index, list[Chunk]]:
    """Chunk *documents* and build a BM25 index over them."""
    from .chunking import chunk_documents

    chunks = chunk_documents(documents, max_words=max_words,
                             overlap_words=overlap_words)
    return BM25Index(chunks), chunks

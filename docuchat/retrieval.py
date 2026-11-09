"""BM25 retrieval over document chunks.

A from-scratch BM25 implementation (Robertson & Zaragoza) using only the
standard library. No embeddings, no API calls, no network: ranking comes
purely from term frequencies in your own documents.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .chunking import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English words that carry no topical signal. Removing them keeps
# questions like "what is the refund policy?" focused on "refund policy".
STOPWORDS = frozenset("""
a an and are as at be but by can could did do does for from had has have
he her his how i if in into is it its me my of on or our she so that the
their them then there these they this to was we what when where which who
will with would you your
get got make made take use used using go going come give put set way ways
thing things need needs want like
""".split())


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase word tokens, optionally dropping stopwords."""
    tokens = TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class BM25Index:
    """An in-memory BM25 index built over a list of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5,
                 b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._doc_freq: Counter[str] = Counter()
        self._term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            self._term_freqs.append(Counter(tokens))
            self._doc_lengths.append(len(tokens))
            for term in set(tokens):
                self._doc_freq[term] += 1
        self._num_docs = len(self.chunks)
        self._avg_len = (sum(self._doc_lengths) / self._num_docs
                         if self._num_docs else 0.0)

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        # Robertson's IDF with a +1 floor so unseen terms score zero
        # instead of going negative.
        return math.log(1.0 + (self._num_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score of one chunk for *query*."""
        total = 0.0
        tf = self._term_freqs[doc_idx]
        doc_len = self._doc_lengths[doc_idx]
        norm = 1.0 - self.b + self.b * (doc_len / self._avg_len
                                        if self._avg_len else 0.0)
        for term in tokenize(query):
            freq = tf.get(term, 0)
            if not freq:
                continue
            total += self._idf(term) * (freq * (self.k1 + 1.0)
                                        / (freq + self.k1 * norm))
        return total

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Return the *top_k* chunks for *query*, best score first.

        Chunks with a zero score (no query term in common) are excluded,
        so an empty list means "nothing in the documents matches".
        """
        if not query.strip() or not self.chunks:
            return []
        results = [SearchResult(chunk=c, score=self.score(query, i))
                   for i, c in enumerate(self.chunks)]
        results = [r for r in results if r.score > 0]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return self._num_docs

"""Three retrievers: BM25, dense vectors, and a fusion of the two.

BM25 is implemented here rather than imported. It is roughly forty lines, it
removes a dependency, and a retrieval project whose author cannot say what the
k1 and b parameters do is not worth much in an interview.

All three return document ids, not chunk ids. When a document is split into
several chunks, its score is the best score among its chunks (max-pooling).
Summing instead would reward long documents purely for having more chunks.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from .chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Okapi BM25.

    k1 controls how quickly term frequency saturates: a term appearing ten times
    is not ten times as meaningful as appearing once. b controls length
    normalisation, from none at b=0 to full at b=1. The defaults are the standard
    starting point from the literature and were not tuned on this dataset, which
    matters -- tuning them on the test queries would inflate the result.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.text) for c in chunks]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype="float32")
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0

        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, tokens in enumerate(self.doc_tokens):
            for term, count in Counter(tokens).items():
                self.postings[term].append((idx, count))

        n_docs = len(chunks)
        self.idf: dict[str, float] = {}
        for term, plist in self.postings.items():
            df = len(plist)
            # Robertson/Sparck-Jones idf with the +0.5 smoothing that keeps very
            # common terms from going negative.
            self.idf[term] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for idx, tf in postings:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                scores[idx] += idf * (tf * (self.k1 + 1)) / denom
        return _pool_to_documents(scores, self.chunks, top_k)


class DenseRetriever:
    """FAISS inner-product search over normalised embeddings."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        import faiss

        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunk count and embedding count must match")
        self.chunks = chunks
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search_vector(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        # Over-fetch: several chunks of one document may occupy the top hits, so
        # asking for exactly top_k chunks can yield far fewer than top_k documents.
        fetch = min(top_k * 8, self.index.ntotal)
        sims, idxs = self.index.search(query_vector.reshape(1, -1), fetch)
        scores = {int(i): float(s) for i, s in zip(idxs[0], sims[0], strict=True) if i >= 0}
        return _pool_to_documents(scores, self.chunks, top_k)


class HybridRetriever:
    """Reciprocal rank fusion of BM25 and dense results.

    RRF combines by rank rather than by score, which sidesteps the fact that BM25
    scores are unbounded while cosine similarities sit in [-1, 1]. Normalising two
    incomparable score scales is where naive hybrids usually go wrong.

    The constant k=60 damps the influence of the very top ranks; it is the value
    from the original RRF paper and, again, was not tuned here.
    """

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, k: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.k = k

    def search(
        self, query: str, query_vector: np.ndarray, top_k: int = 10
    ) -> list[tuple[str, float]]:
        depth = top_k * 4
        lexical = self.bm25.search(query, top_k=depth)
        semantic = self.dense.search_vector(query_vector, top_k=depth)

        fused: dict[str, float] = defaultdict(float)
        for ranking in (lexical, semantic):
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                fused[doc_id] += 1.0 / (self.k + rank)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return ordered[:top_k]


def _pool_to_documents(
    chunk_scores: dict[int, float], chunks: list[Chunk], top_k: int
) -> list[tuple[str, float]]:
    """Collapse chunk scores to document scores by taking each document's best chunk."""
    best: dict[str, float] = {}
    for idx, score in chunk_scores.items():
        doc_id = chunks[idx].doc_id
        if score > best.get(doc_id, float("-inf")):
            best[doc_id] = score
    ordered = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:top_k]

"""Chunking strategies.

Chunking is usually a one-line decision made once and never revisited, which is
why it is worth measuring. Every strategy here produces `Chunk` objects that keep
a `doc_id`, so retrieval can happen at chunk level while scoring happens at
document level -- the mapping back is where most naive implementations lose
recall.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .data import Document

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter.

    Deliberately dependency-free. It mis-splits on abbreviations such as "et al."
    and on decimals inside scientific text; that error is measured rather than
    hidden -- see the Limitations section of the README.
    """
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_whole(docs: Iterable[Document]) -> list[Chunk]:
    """One chunk per document. The baseline everything else has to beat."""
    return [Chunk(chunk_id=d.doc_id, doc_id=d.doc_id, text=d.full_text) for d in docs]


def chunk_sentences(docs: Iterable[Document], group_size: int = 3) -> list[Chunk]:
    """Sentence-aware chunks of `group_size` sentences, no overlap.

    The title is prepended to every chunk. Without it, a chunk taken from the
    middle of an abstract loses all indication of what the paper is about, which
    is the single most common cause of a chunk that cannot be retrieved.
    """
    chunks: list[Chunk] = []
    for d in docs:
        sentences = split_sentences(d.text)
        if not sentences:
            chunks.append(Chunk(f"{d.doc_id}::0", d.doc_id, d.full_text))
            continue
        for i in range(0, len(sentences), group_size):
            body = " ".join(sentences[i : i + group_size])
            text = f"{d.title}. {body}" if d.title else body
            chunks.append(Chunk(f"{d.doc_id}::{i // group_size}", d.doc_id, text))
    return chunks


def chunk_window(docs: Iterable[Document], size: int = 120, overlap: int = 40) -> list[Chunk]:
    """Fixed-width overlapping word windows.

    Overlap exists so a fact split across a boundary still appears intact in one
    chunk. It costs index size in exchange for recall, which is precisely the
    trade-off the evaluation reports.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    stride = size - overlap
    chunks: list[Chunk] = []
    for d in docs:
        words = d.text.split()
        if not words:
            chunks.append(Chunk(f"{d.doc_id}::0", d.doc_id, d.full_text))
            continue
        idx = 0
        for start in range(0, len(words), stride):
            window = words[start : start + size]
            if not window:
                break
            body = " ".join(window)
            text = f"{d.title}. {body}" if d.title else body
            chunks.append(Chunk(f"{d.doc_id}::{idx}", d.doc_id, text))
            idx += 1
            if start + size >= len(words):
                break
    return chunks


STRATEGIES: dict[str, Callable[[list[Document]], list[Chunk]]] = {
    "whole": chunk_whole,
    "sentence3": lambda docs: chunk_sentences(docs, group_size=3),
    "window120": lambda docs: chunk_window(docs, size=120, overlap=40),
}

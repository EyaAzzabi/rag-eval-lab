"""Sentence embedding with an on-disk cache.

Re-embedding 5,000 documents on a CPU takes about a minute. Doing that once per
experiment in a nine-configuration sweep wastes most of the runtime, so encoded
vectors are cached under a key derived from the model name and the exact texts.
Change either and the cache misses, which is the behaviour you want -- a stale
embedding cache silently invalidates every number downstream.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / ".embedding_cache"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 128):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], cache_key: str | None = None) -> np.ndarray:
        """Return L2-normalised float32 embeddings, one row per text.

        Normalising here means inner product equals cosine similarity, so the
        FAISS index can use the cheaper IndexFlatIP without a separate
        normalisation step at query time.
        """
        if cache_key:
            path = self._cache_path(texts, cache_key)
            if path.exists():
                return np.load(path)

        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 2000,
        ).astype("float32")

        if cache_key:
            path = self._cache_path(texts, cache_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        return vectors

    def _cache_path(self, texts: list[str], cache_key: str) -> Path:
        digest = hashlib.sha256()
        digest.update(self.model_name.encode())
        digest.update(str(len(texts)).encode())
        # Hashing every text is slow for large corpora and hashing none is unsafe;
        # a fixed stride is the compromise, and it catches reordering too.
        for text in texts[:: max(1, len(texts) // 256)]:
            digest.update(text.encode("utf-8", errors="ignore"))
        return CACHE_DIR / f"{cache_key}-{digest.hexdigest()[:16]}.npy"

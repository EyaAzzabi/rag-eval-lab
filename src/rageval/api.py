"""FastAPI service exposing the best configuration found by the sweep.

The index is built once at startup and held in memory. Rebuilding per request
would make latency meaningless, and lazy-building on first request hides the cost
in whichever unlucky user arrives first.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chunking import STRATEGIES
from .data import load_corpus
from .embedding import Embedder
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever

CHUNKING = os.getenv("RAG_CHUNKING", "whole")
RETRIEVER = os.getenv("RAG_RETRIEVER", "hybrid")
DATASET = os.getenv("RAG_DATASET", "scifact")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if CHUNKING not in STRATEGIES:
        raise RuntimeError(f"RAG_CHUNKING must be one of {sorted(STRATEGIES)}")
    corpus = load_corpus(DATASET)
    chunks = STRATEGIES[CHUNKING](corpus)
    embedder = Embedder()
    vectors = embedder.encode([c.text for c in chunks], cache_key=f"{DATASET}-{CHUNKING}")

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, vectors)
    state.update(
        embedder=embedder,
        bm25=bm25,
        dense=dense,
        hybrid=HybridRetriever(bm25, dense),
        docs={d.doc_id: d for d in corpus},
        n_chunks=len(chunks),
    )
    yield
    state.clear()


app = FastAPI(
    title="rag-eval-lab",
    description="Retrieval over SciFact with a measured configuration.",
    version="0.1.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    retriever: str | None = Field(default=None, pattern="^(bm25|dense|hybrid)$")


class Hit(BaseModel):
    doc_id: str
    score: float
    title: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    retriever: str
    took_ms: float
    hits: list[Hit]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if state else "starting",
        "chunking": CHUNKING,
        "retriever": RETRIEVER,
        "n_chunks": state.get("n_chunks", 0),
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    if not state:
        raise HTTPException(status_code=503, detail="index still loading")

    name = request.retriever or RETRIEVER
    start = time.perf_counter()
    if name == "bm25":
        results = state["bm25"].search(request.query, top_k=request.top_k)
    else:
        vector = state["embedder"].encode([request.query])[0]
        if name == "dense":
            results = state["dense"].search_vector(vector, top_k=request.top_k)
        else:
            results = state["hybrid"].search(request.query, vector, top_k=request.top_k)
    took_ms = (time.perf_counter() - start) * 1000

    docs = state["docs"]
    hits = [
        Hit(
            doc_id=doc_id,
            score=round(float(score), 4),
            title=docs[doc_id].title if doc_id in docs else "",
            snippet=(docs[doc_id].text[:300] if doc_id in docs else ""),
        )
        for doc_id, score in results
    ]
    return SearchResponse(
        query=request.query, retriever=name, took_ms=round(took_ms, 2), hits=hits
    )

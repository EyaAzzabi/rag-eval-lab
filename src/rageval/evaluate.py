"""Run the retriever x chunking sweep and write results to disk.

Reports quality and latency together. A configuration that gains two points of
recall for four times the query latency is not obviously better, and a table that
omits the second number cannot say so.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunking import STRATEGIES
from .data import load_corpus, load_qrels, load_queries
from .embedding import Embedder
from .metrics import evaluate_run
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
KS = (1, 3, 5, 10)


@dataclass
class RunResult:
    retriever: str
    chunking: str
    n_chunks: int
    index_build_seconds: float
    latency_ms_p50: float
    latency_ms_p95: float
    metrics: dict[str, float]


def run_sweep(dataset: str = "scifact", split: str = "test", top_k: int = 10) -> list[RunResult]:
    corpus = load_corpus(dataset)
    queries = load_queries(dataset, split)
    qrels = load_qrels(dataset, split)
    print(f"{len(corpus)} documents, {len(queries)} judged queries")

    embedder = Embedder()
    query_vectors = embedder.encode(
        [q.text for q in queries], cache_key=f"{dataset}-{split}-queries"
    )

    results: list[RunResult] = []
    for chunk_name, strategy in STRATEGIES.items():
        chunks = strategy(corpus)
        print(f"\n[{chunk_name}] {len(chunks)} chunks")

        t0 = time.perf_counter()
        chunk_vectors = embedder.encode(
            [c.text for c in chunks], cache_key=f"{dataset}-{chunk_name}"
        )
        dense = DenseRetriever(chunks, chunk_vectors)
        dense_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        bm25 = BM25Retriever(chunks)
        bm25_build = time.perf_counter() - t0

        hybrid = HybridRetriever(bm25, dense)

        configs = {
            # Retrievers are bound as default arguments rather than captured from
            # the enclosing loop: a late-binding closure here would silently
            # evaluate every chunking strategy against the last one built.
            "bm25": (lambda q, v, r=bm25: r.search(q.text, top_k), bm25_build),
            "dense": (lambda q, v, r=dense: r.search_vector(v, top_k), dense_build),
            "hybrid": (
                lambda q, v, r=hybrid: r.search(q.text, v, top_k),
                dense_build + bm25_build,
            ),
        }

        for retriever_name, (search_fn, build_seconds) in configs.items():
            run: dict[str, list[str]] = {}
            latencies: list[float] = []
            for query, vector in zip(queries, query_vectors, strict=True):
                start = time.perf_counter()
                hits = search_fn(query, vector)
                latencies.append((time.perf_counter() - start) * 1000)
                run[query.query_id] = [doc_id for doc_id, _ in hits]

            metrics = evaluate_run(run, qrels, ks=KS)
            result = RunResult(
                retriever=retriever_name,
                chunking=chunk_name,
                n_chunks=len(chunks),
                index_build_seconds=round(build_seconds, 2),
                latency_ms_p50=round(statistics.median(latencies), 2),
                latency_ms_p95=round(_percentile(latencies, 95), 2),
                metrics={k: round(v, 4) for k, v in metrics.items()},
            )
            results.append(result)
            print(
                f"  {retriever_name:7s} ndcg@10={metrics['ndcg@10']:.4f} "
                f"recall@10={metrics['recall@10']:.4f} p50={result.latency_ms_p50:.1f}ms"
            )

    _write(results, dataset, split)
    return results


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round(pct / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]


def _write(results: list[RunResult], dataset: str, split: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "split": split,
        "runs": [asdict(r) for r in results],
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = ["retriever", "chunking", "n_chunks", "latency_ms_p50", "latency_ms_p95"]
    metric_names = sorted(results[0].metrics) if results else []
    lines = [",".join(header + metric_names)]
    for r in results:
        row = [
            r.retriever,
            r.chunking,
            str(r.n_chunks),
            str(r.latency_ms_p50),
            str(r.latency_ms_p95),
        ] + [f"{r.metrics[m]:.4f}" for m in metric_names]
        lines.append(",".join(row))
    (RESULTS_DIR / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS_DIR / 'results.json'} and results.csv")


if __name__ == "__main__":
    run_sweep()

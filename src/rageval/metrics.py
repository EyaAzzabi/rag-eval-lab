"""Retrieval metrics, implemented directly rather than imported.

Each function takes a ranked list of document ids and the graded relevance map
for that query, and returns a single number. Written out in full because the
definitions differ subtly between papers and libraries, and an evaluation is only
comparable if the reader can see exactly which variant was used.
"""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Fraction of all relevant documents that appear in the top k.

    Note the denominator is the total number of relevant documents, not k. With
    fewer than k relevant documents this can reach 1.0; that is intended.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / k


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Reciprocal of the rank of the first relevant document, 0 if none in top k.

    Answers "how far down did the user have to read?", which for a RAG system
    matters more than raw recall: a relevant chunk at rank 9 rarely survives into
    a context window that holds five.
    """
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain, using graded relevance.

    The only metric here that distinguishes a highly relevant document from a
    marginally relevant one, and the one to trust when the others disagree.
    """
    if not relevant:
        return 0.0
    gains = [float(relevant.get(doc_id, 0)) for doc_id in ranked[:k]]
    ideal = sorted((float(v) for v in relevant.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def evaluate_run(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Average each metric over all judged queries.

    Queries with no judgments are skipped rather than scored as zero. Including
    them would silently depress every number and make runs incomparable across
    datasets.
    """
    scores: dict[str, list[float]] = {}
    for qid, ranked in run.items():
        relevant = qrels.get(qid)
        if not relevant:
            continue
        for k in ks:
            scores.setdefault(f"recall@{k}", []).append(recall_at_k(ranked, relevant, k))
            scores.setdefault(f"precision@{k}", []).append(precision_at_k(ranked, relevant, k))
            scores.setdefault(f"mrr@{k}", []).append(mrr_at_k(ranked, relevant, k))
            scores.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(ranked, relevant, k))
    return {name: (sum(v) / len(v) if v else 0.0) for name, v in scores.items()}

"""Loading for BEIR-format retrieval datasets.

We use SciFact: 5,183 scientific abstracts and 300 test claims, with relevance
judgments made by human annotators. The judgments are the reason this dataset was
chosen -- a retrieval score is only meaningful if somebody other than the author
decided what "relevant" means.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        """Title and body joined. Titles carry real signal in scientific abstracts."""
        return f"{self.title}. {self.text}".strip() if self.title else self.text


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


def load_corpus(dataset: str = "scifact") -> list[Document]:
    path = DATA_DIR / dataset / "corpus.jsonl"
    _require(path, dataset)
    docs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(
                Document(
                    doc_id=str(row["_id"]),
                    title=row.get("title", "") or "",
                    text=row.get("text", "") or "",
                )
            )
    return docs


def load_queries(dataset: str = "scifact", split: str = "test") -> list[Query]:
    """Return only the queries that have judgments in the requested split.

    BEIR ships every query in one file regardless of split, so filtering by the
    qrels is what actually separates train from test. Skipping this step silently
    evaluates on unjudged queries, which scores them as though every retrieved
    document were wrong.
    """
    path = DATA_DIR / dataset / "queries.jsonl"
    _require(path, dataset)
    qrels = load_qrels(dataset, split)
    queries = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["_id"])
            if qid in qrels:
                queries.append(Query(query_id=qid, text=row["text"]))
    return queries


def load_qrels(dataset: str = "scifact", split: str = "test") -> dict[str, dict[str, int]]:
    """Map query_id -> {doc_id: graded relevance}."""
    path = DATA_DIR / dataset / "qrels" / f"{split}.tsv"
    _require(path, dataset)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as fh:
        header = fh.readline()  # query-id  corpus-id  score
        if "query-id" not in header:
            fh.seek(0)
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels[qid][did] = score
    return dict(qrels)


def _require(path: Path, dataset: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}.\nRun: python scripts/download_data.py --dataset {dataset}"
        )

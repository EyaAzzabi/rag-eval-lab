import numpy as np

from rageval.chunking import Chunk
from rageval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, tokenize

CHUNKS = [
    Chunk("c1", "d1", "the cat sat on the mat"),
    Chunk("c2", "d2", "dogs are loyal animals and dogs bark"),
    Chunk("c3", "d3", "quantum entanglement in superconducting circuits"),
]


def test_tokenizer_lowercases_and_drops_punctuation():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_bm25_ranks_the_matching_document_first():
    bm25 = BM25Retriever(CHUNKS)
    hits = bm25.search("cat mat", top_k=3)
    assert hits[0][0] == "d1"


def test_bm25_returns_nothing_for_out_of_vocabulary_queries():
    bm25 = BM25Retriever(CHUNKS)
    assert bm25.search("zzzz nonexistent", top_k=3) == []


def test_chunks_are_pooled_to_one_entry_per_document():
    # Two chunks of the same document must not occupy two result slots.
    chunks = [
        Chunk("c1", "d1", "alpha beta"),
        Chunk("c2", "d1", "alpha gamma"),
        Chunk("c3", "d2", "delta"),
    ]
    hits = BM25Retriever(chunks).search("alpha", top_k=5)
    assert [doc_id for doc_id, _ in hits].count("d1") == 1


def test_dense_retriever_finds_the_nearest_vector():
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype="float32"
    )
    dense = DenseRetriever(CHUNKS, vectors)
    hits = dense.search_vector(np.array([1.0, 0.0], dtype="float32"), top_k=3)
    assert hits[0][0] == "d1"


def test_dense_retriever_rejects_mismatched_inputs():
    try:
        DenseRetriever(CHUNKS, np.zeros((2, 4), dtype="float32"))
    except ValueError:
        return
    raise AssertionError("expected ValueError when counts disagree")


def test_hybrid_surfaces_a_document_only_one_retriever_found():
    """The point of fusion: a lexical-only match still reaches the final ranking."""
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype="float32")
    bm25 = BM25Retriever(CHUNKS)
    dense = DenseRetriever(CHUNKS, vectors)
    hybrid = HybridRetriever(bm25, dense)
    hits = hybrid.search("quantum entanglement", np.array([1.0, 0.0], dtype="float32"), top_k=3)
    doc_ids = [doc_id for doc_id, _ in hits]
    assert "d3" in doc_ids  # found lexically
    assert "d1" in doc_ids  # found semantically


def test_hybrid_scores_are_descending():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype="float32")
    hybrid = HybridRetriever(BM25Retriever(CHUNKS), DenseRetriever(CHUNKS, vectors))
    scores = [s for _, s in hybrid.search("dogs", np.array([0.0, 1.0], dtype="float32"), top_k=3)]
    assert scores == sorted(scores, reverse=True)

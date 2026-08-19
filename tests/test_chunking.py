from rageval.chunking import (
    chunk_sentences,
    chunk_whole,
    chunk_window,
    split_sentences,
)
from rageval.data import Document

DOC = Document(
    doc_id="d1",
    title="Cell biology",
    text="Cells divide. Mitosis follows interphase. Errors cause disease. Repair is active.",
)


def test_whole_gives_one_chunk_per_document():
    chunks = chunk_whole([DOC])
    assert len(chunks) == 1
    assert chunks[0].doc_id == "d1"
    assert "Cell biology" in chunks[0].text


def test_sentence_chunks_keep_the_title_for_context():
    chunks = chunk_sentences([DOC], group_size=2)
    assert len(chunks) == 2
    assert all(c.text.startswith("Cell biology") for c in chunks)
    assert all(c.doc_id == "d1" for c in chunks)


def test_chunk_ids_are_unique_but_doc_id_is_shared():
    chunks = chunk_sentences([DOC], group_size=1)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.doc_id for c in chunks} == {"d1"}


def test_window_chunks_overlap():
    long_doc = Document("d2", "T", " ".join(f"w{i}" for i in range(300)))
    chunks = chunk_window([long_doc], size=100, overlap=40)
    assert len(chunks) > 1
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words, "overlapping windows must share words"


def test_window_rejects_overlap_at_or_above_size():
    try:
        chunk_window([DOC], size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= size")


def test_empty_document_still_produces_a_chunk():
    empty = Document("d3", "Title only", "")
    assert len(chunk_sentences([empty])) == 1
    assert len(chunk_window([empty])) == 1


def test_sentence_splitter_handles_trailing_whitespace():
    assert split_sentences("One. Two.  ") == ["One.", "Two."]


def test_sentence_splitter_returns_text_when_no_terminator():
    assert split_sentences("no terminator here") == ["no terminator here"]

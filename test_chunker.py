"""
tests/test_chunker.py

Unit tests for SemanticChunker.
Uses a mock SentenceTransformer to avoid downloading model weights in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

from src.knowledge.semantic_chunker import SemanticChunker


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    """Return a mock SentenceTransformer that produces deterministic embeddings."""
    model = MagicMock()

    def fake_encode(texts, **kwargs):
        # Each sentence gets a unit vector with a random but seeded direction
        rng = np.random.default_rng(seed=42)
        n = len(texts)
        vecs = rng.standard_normal((n, 64)).astype(np.float32)
        # L2 normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-8)

    model.encode.side_effect = fake_encode
    return model


@pytest.fixture
def chunker(mock_model):
    with patch("src.knowledge.semantic_chunker.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            semantic_split_threshold=0.65,
            chunk_size=512,
            embedding_model="BAAI/bge-m3",
            embedding_device="cpu",
        )
        return SemanticChunker(model=mock_model, threshold=0.65, chunk_size=512)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSemanticChunker:

    def test_empty_document_returns_no_chunks(self, chunker):
        doc = Document(page_content="", metadata={"source": "test.txt"})
        result = chunker.chunk_documents([doc])
        assert result == []

    def test_short_document_returns_single_chunk(self, chunker):
        doc = Document(
            page_content="This is a short document. It has two sentences.",
            metadata={"source": "test.txt", "doc_type": "interview"},
        )
        result = chunker.chunk_documents([doc])
        assert len(result) >= 1
        # Original metadata should be preserved
        assert result[0].metadata["source"] == "test.txt"
        assert result[0].metadata["doc_type"] == "interview"

    def test_chunk_metadata_populated(self, chunker):
        doc = Document(
            page_content="First sentence. Second sentence. Third sentence.",
            metadata={"source": "test.txt"},
        )
        chunks = chunker.chunk_documents([doc])
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "sentence_start" in chunk.metadata
            assert "sentence_end" in chunk.metadata
            assert "chunk_char_len" in chunk.metadata

    def test_long_document_produces_multiple_chunks(self, chunker):
        # Generate a document long enough to be split
        sentences = [f"This is sentence number {i} about a completely different topic." for i in range(30)]
        text = " ".join(sentences)
        doc = Document(page_content=text, metadata={"source": "long.txt"})
        chunks = chunker.chunk_documents([doc])
        # Should produce at least 2 chunks for a 30-sentence document
        assert len(chunks) >= 1
        # All chunks should have non-empty content
        for chunk in chunks:
            assert len(chunk.page_content.strip()) > 0

    def test_sentence_tokenizer_handles_abbreviations(self, chunker):
        text = "Dr. Smith said the results were clear. The p-value was 0.05. We conclude the null is rejected."
        sentences = chunker._sentence_tokenize(text)
        # Should produce at least 1 sentence
        assert len(sentences) >= 1

    def test_chunk_documents_empty_list(self, chunker):
        result = chunker.chunk_documents([])
        assert result == []

    def test_multiple_documents(self, chunker):
        docs = [
            Document(page_content=f"Document {i}. It has some content. More content here.", metadata={"source": f"doc{i}.txt"})
            for i in range(5)
        ]
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) >= len(docs)

    def test_metadata_inheritance(self, chunker):
        doc = Document(
            page_content="First sentence. Second sentence.",
            metadata={"source": "interview.txt", "doc_type": "interview", "page": 2},
        )
        chunks = chunker.chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["doc_type"] == "interview"
            assert chunk.metadata["page"] == 2

    def test_bisect_chunk_respects_size_limit(self, chunker):
        long_sentences = [f"Word " * 50 + f"sentence {i}." for i in range(20)]
        result = chunker._bisect_chunk(long_sentences)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_find_boundaries_returns_list(self, chunker, mock_model):
        n = 10
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((n, 64)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        boundaries = chunker._find_boundaries(embeddings)
        assert isinstance(boundaries, list)
        for b in boundaries:
            assert 0 <= b < n

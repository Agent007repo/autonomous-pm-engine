"""
tests/test_vector_store.py

Unit tests for VectorStore using an in-process ephemeral ChromaDB instance.
No Docker or external services required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

from src.knowledge.vector_store import VectorStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedder():
    embedder = MagicMock()

    def fake_encode(texts, **kwargs):
        rng = np.random.default_rng(seed=0)
        n = len(texts) if isinstance(texts, list) else 1
        vecs = rng.standard_normal((n, 64)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-8)

    embedder.encode.side_effect = fake_encode
    return embedder


@pytest.fixture
def vector_store(mock_embedder):
    """Build an in-process ephemeral VectorStore (no Docker required)."""
    with patch("src.knowledge.vector_store.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            chroma_mode="local",
            chroma_host="localhost",
            chroma_port=8001,
            chroma_collection="test_collection",
            embedding_model="BAAI/bge-m3",
            embedding_device="cpu",
            top_k_retrieval=5,
            hybrid_alpha=0.7,
        )
        store = VectorStore(embedding_model=mock_embedder, collection_name="test_collection")
        yield store
        store.clear_collection()


def _make_docs(n: int = 5) -> list[Document]:
    return [
        Document(
            page_content=f"Customer feedback document number {i}. "
                         f"The search feature is slow and unreliable.",
            metadata={"source": f"test_{i}.txt", "doc_type": "interview", "page": i},
        )
        for i in range(n)
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestVectorStore:

    def test_upsert_returns_correct_count(self, vector_store):
        docs = _make_docs(5)
        count = vector_store.upsert_documents(docs)
        assert count == 5

    def test_upsert_empty_list_returns_zero(self, vector_store):
        count = vector_store.upsert_documents([])
        assert count == 0

    def test_collection_count_increases_after_upsert(self, vector_store):
        before = vector_store.collection_count()
        docs = _make_docs(3)
        vector_store.upsert_documents(docs)
        after = vector_store.collection_count()
        assert after >= before + 1  # May be less than 3 if hashes collide

    def test_upsert_idempotent(self, vector_store):
        docs = _make_docs(3)
        vector_store.upsert_documents(docs)
        count_after_first = vector_store.collection_count()
        # Upsert same docs again
        vector_store.upsert_documents(docs)
        count_after_second = vector_store.collection_count()
        assert count_after_first == count_after_second

    def test_similarity_search_returns_documents(self, vector_store):
        docs = _make_docs(5)
        vector_store.upsert_documents(docs)
        results = vector_store.similarity_search("slow search feature", k=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        for doc in results:
            assert isinstance(doc, Document)
            assert doc.page_content

    def test_hybrid_search_returns_documents(self, vector_store):
        docs = _make_docs(5)
        vector_store.upsert_documents(docs)
        results = vector_store.hybrid_search("feedback search", k=3)
        assert isinstance(results, list)
        for doc in results:
            assert isinstance(doc, Document)

    def test_metadata_filter_applied(self, vector_store):
        docs_interview = [
            Document(
                page_content=f"Interview doc {i}",
                metadata={"source": f"int_{i}.txt", "doc_type": "interview"},
            )
            for i in range(3)
        ]
        docs_survey = [
            Document(
                page_content=f"Survey doc {i}",
                metadata={"source": f"sur_{i}.txt", "doc_type": "survey"},
            )
            for i in range(3)
        ]
        vector_store.upsert_documents(docs_interview + docs_survey)
        results = vector_store.similarity_search("doc", k=10, where={"doc_type": "survey"})
        for doc in results:
            assert doc.metadata.get("doc_type") == "survey"

    def test_clear_collection_empties_store(self, vector_store):
        docs = _make_docs(3)
        vector_store.upsert_documents(docs)
        assert vector_store.collection_count() > 0
        vector_store.clear_collection()
        assert vector_store.collection_count() == 0

    def test_doc_id_deterministic(self):
        doc = Document(page_content="hello world", metadata={})
        id1 = VectorStore._doc_id(doc)
        id2 = VectorStore._doc_id(doc)
        assert id1 == id2

    def test_doc_id_different_for_different_content(self):
        doc1 = Document(page_content="hello world", metadata={})
        doc2 = Document(page_content="goodbye world", metadata={})
        assert VectorStore._doc_id(doc1) != VectorStore._doc_id(doc2)

    def test_sanitise_metadata_converts_non_primitives(self):
        meta = {"key1": "string", "key2": 42, "key3": 3.14, "key4": True, "key5": [1, 2, 3]}
        clean = VectorStore._sanitise_metadata(meta)
        assert clean["key1"] == "string"
        assert clean["key2"] == 42
        assert clean["key5"] == "[1, 2, 3]"

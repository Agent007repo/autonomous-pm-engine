"""
src/knowledge/vector_store.py

ChromaDB wrapper that supports:
  - Upsert (add or update) document chunks
  - Dense semantic search (cosine similarity via BGE embeddings)
  - Sparse BM25 search (via ChromaDB's built-in keyword search)
  - Hybrid search: linear combination of dense + sparse scores
  - Metadata filtering (by doc_type, source)
"""

from __future__ import annotations

import hashlib
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings


class VectorStore:
    """
    Thin wrapper around ChromaDB that adds:
      1. Automatic BGE-M3 embedding
      2. Hybrid dense + sparse retrieval
      3. Convenience methods for the PM pipeline
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer | None = None,
        collection_name: str | None = None,
    ) -> None:
        cfg = get_settings()
        self._cfg = cfg
        self._collection_name = collection_name or cfg.chroma_collection

        # Initialise embedding model
        self._embedder: SentenceTransformer = embedding_model or SentenceTransformer(
            cfg.embedding_model, device=cfg.embedding_device
        )

        # Connect to ChromaDB
        if cfg.chroma_mode == "server":
            self._client = chromadb.HttpClient(
                host=cfg.chroma_host,
                port=cfg.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(f"Connected to ChromaDB server at {cfg.chroma_host}:{cfg.chroma_port}")
        else:
            self._client = chromadb.EphemeralClient(
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            logger.info("Using in-process ephemeral ChromaDB instance")

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{self._collection_name}' ready "
            f"(docs: {self._collection.count()})"
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_documents(self, documents: list[Document]) -> int:
        """
        Embed and upsert a list of Documents.
        Uses content hash as document ID to enable idempotent upserts.
        Returns the number of documents successfully upserted.
        """
        if not documents:
            return 0

        ids = [self._doc_id(doc) for doc in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [self._sanitise_metadata(doc.metadata) for doc in documents]

        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self._embedder.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        count_after = self._collection.count()
        logger.info(f"Upsert complete. Collection size: {count_after}")
        return len(ids)

    def clear_collection(self) -> None:
        """Delete all documents in the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Cleared collection '{self._collection_name}'")

    # ── Read ──────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Pure dense search using cosine similarity.
        `where` is a ChromaDB metadata filter dict, e.g. {"doc_type": "survey"}.
        """
        k = k or self._cfg.top_k_retrieval
        query_embedding = self._embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, max(1, self._collection.count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return self._results_to_documents(results)

    def keyword_search(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Sparse keyword search using ChromaDB's built-in full-text matching.
        Note: ChromaDB uses a simple substring/contains match under the hood.
        """
        k = k or self._cfg.top_k_retrieval
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, max(1, self._collection.count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._results_to_documents(results)

    def hybrid_search(
        self,
        query: str,
        k: int | None = None,
        alpha: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Hybrid retrieval: linear interpolation of dense and sparse scores.

        score = alpha * dense_score + (1 - alpha) * sparse_score

        alpha = 1.0 -> pure dense
        alpha = 0.0 -> pure sparse
        """
        k = k or self._cfg.top_k_retrieval
        alpha = alpha if alpha is not None else self._cfg.hybrid_alpha
        fetch_k = min(k * 3, 50)  # Over-fetch so fusion has enough candidates

        dense_docs = self.similarity_search(query, k=fetch_k, where=where)
        sparse_docs = self.keyword_search(query, k=fetch_k, where=where)

        # Build score maps keyed by document content hash
        dense_scores: dict[str, tuple[float, Document]] = {}
        for rank, doc in enumerate(dense_docs):
            score = 1.0 - (rank / max(len(dense_docs), 1))  # normalised rank score
            doc_id = self._doc_id(doc)
            dense_scores[doc_id] = (score, doc)

        sparse_scores: dict[str, tuple[float, Document]] = {}
        for rank, doc in enumerate(sparse_docs):
            score = 1.0 - (rank / max(len(sparse_docs), 1))
            doc_id = self._doc_id(doc)
            sparse_scores[doc_id] = (score, doc)

        all_ids = set(dense_scores) | set(sparse_scores)
        fused: list[tuple[float, Document]] = []

        for doc_id in all_ids:
            d_score = dense_scores[doc_id][0] if doc_id in dense_scores else 0.0
            s_score = sparse_scores[doc_id][0] if doc_id in sparse_scores else 0.0
            combined = alpha * d_score + (1.0 - alpha) * s_score
            doc = (dense_scores.get(doc_id) or sparse_scores.get(doc_id))[1]  # type: ignore[index]
            fused.append((combined, doc))

        fused.sort(key=lambda x: x[0], reverse=True)
        top_k = [doc for _, doc in fused[:k]]
        return top_k

    def collection_count(self) -> int:
        return self._collection.count()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _doc_id(doc: Document) -> str:
        content = doc.page_content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:16]

    @staticmethod
    def _sanitise_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        ChromaDB only accepts str, int, float, bool metadata values.
        Convert anything else to string.
        """
        clean: dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    @staticmethod
    def _results_to_documents(results: dict[str, Any]) -> list[Document]:
        docs: list[Document] = []
        if not results or not results.get("documents"):
            return docs
        for text_list, meta_list, dist_list in zip(
            results["documents"],
            results["metadatas"],
            results["distances"],
        ):
            for text, meta, dist in zip(text_list, meta_list, dist_list):
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={**(meta or {}), "_distance": dist},
                    )
                    docs.append(doc)
        return docs

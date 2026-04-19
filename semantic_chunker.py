"""
src/knowledge/semantic_chunker.py

Semantic chunking strategy:
  1. Split documents into individual sentences.
  2. Embed each sentence with a lightweight model.
  3. Compute rolling cosine similarity between adjacent sentence groups.
  4. Insert chunk boundaries where similarity drops below the threshold.
  5. Merge micro-chunks that are below the minimum token count.

This produces topically coherent chunks that respect natural topic shifts,
rather than blindly slicing at fixed token counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from langchain_core.documents import Document
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings

if TYPE_CHECKING:
    pass


@dataclass
class ChunkMetadata:
    source: str
    doc_type: str
    chunk_index: int
    sentence_start: int
    sentence_end: int
    token_count: int
    extra: dict = field(default_factory=dict)


class SemanticChunker:
    """
    Splits LangChain Documents into semantically coherent chunks.

    Algorithm
    ---------
    For each document:
      1. Sentence-tokenize the text.
      2. Embed every sentence (batched for speed).
      3. Slide a window of `window_size` sentences, compute the centroid
         embedding of each window.
      4. Where the cosine similarity between consecutive windows falls below
         `threshold`, insert a chunk boundary.
      5. Enforce max/min token constraints by splitting or merging.
    """

    def __init__(
        self,
        model: SentenceTransformer | None = None,
        threshold: float | None = None,
        chunk_size: int | None = None,
        chunk_overlap_sentences: int = 1,
        window_size: int = 3,
    ) -> None:
        cfg = get_settings()
        self.threshold = threshold or cfg.semantic_split_threshold
        self.max_chunk_tokens = chunk_size or cfg.chunk_size
        self.overlap_sentences = chunk_overlap_sentences
        self.window_size = window_size

        logger.info(f"Loading embedding model: {cfg.embedding_model}")
        self._model = model or SentenceTransformer(
            cfg.embedding_model, device=cfg.embedding_device
        )
        logger.info("SemanticChunker ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Chunk a list of Documents. Returns a new (larger) list of Documents
        where each item represents a single semantic chunk.
        """
        result: list[Document] = []
        for doc in documents:
            chunks = self._chunk_single(doc)
            result.extend(chunks)
            logger.debug(
                f"{doc.metadata.get('source', '?')} -> {len(chunks)} chunks"
            )
        logger.info(
            f"Chunking complete: {len(documents)} docs -> {len(result)} chunks"
        )
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _chunk_single(self, doc: Document) -> list[Document]:
        text = doc.page_content.strip()
        if not text:
            return []

        sentences = self._sentence_tokenize(text)
        if len(sentences) <= self.window_size:
            # Document too short to split meaningfully
            return [self._make_chunk(doc, sentences, 0, len(sentences) - 1, 0)]

        embeddings = self._embed_sentences(sentences)
        boundaries = self._find_boundaries(embeddings)
        raw_chunks = self._split_at_boundaries(sentences, boundaries)
        merged_chunks = self._enforce_size_constraints(raw_chunks)

        result: list[Document] = []
        sent_cursor = 0
        for i, chunk_sentences in enumerate(merged_chunks):
            chunk_doc = self._make_chunk(
                doc,
                chunk_sentences,
                sent_cursor,
                sent_cursor + len(chunk_sentences) - 1,
                i,
            )
            result.append(chunk_doc)
            # Overlap: next chunk starts `overlap_sentences` before this one ended
            sent_cursor += max(1, len(chunk_sentences) - self.overlap_sentences)

        return result

    def _sentence_tokenize(self, text: str) -> list[str]:
        """
        Simple regex-based sentence splitter.
        Splits on '. ', '! ', '? ', newlines.
        Filters empty strings and strips whitespace.
        """
        raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        return [s.strip() for s in raw if s.strip()]

    def _embed_sentences(self, sentences: list[str]) -> np.ndarray:
        """
        Returns a (N, D) float32 array of L2-normalised embeddings.
        Batched to avoid OOM on large documents.
        """
        embeddings = self._model.encode(
            sentences,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)

    def _find_boundaries(self, embeddings: np.ndarray) -> list[int]:
        """
        Returns a list of sentence indices where a chunk boundary should be inserted.
        A boundary is placed *after* index i when the similarity between
        window [i-w+1 : i+1] and window [i+1 : i+w+1] drops below threshold.
        """
        n = len(embeddings)
        w = self.window_size
        boundaries: list[int] = []

        for i in range(w - 1, n - w):
            left_centroid = embeddings[max(0, i - w + 1) : i + 1].mean(axis=0)
            right_centroid = embeddings[i + 1 : i + 1 + w].mean(axis=0)

            # Cosine similarity (embeddings are already L2-normalised)
            sim = float(np.dot(left_centroid, right_centroid))

            if sim < self.threshold:
                boundaries.append(i)

        return boundaries

    def _split_at_boundaries(
        self, sentences: list[str], boundaries: list[int]
    ) -> list[list[str]]:
        chunks: list[list[str]] = []
        start = 0
        for boundary in boundaries:
            chunks.append(sentences[start : boundary + 1])
            start = boundary + 1
        chunks.append(sentences[start:])
        return [c for c in chunks if c]

    def _enforce_size_constraints(
        self, chunks: list[list[str]]
    ) -> list[list[str]]:
        """
        Merge chunks that are too small into their neighbour,
        and split chunks that exceed max_chunk_tokens.
        Uses a simple word-count proxy (1 word ~ 1.3 tokens).
        """
        TOKEN_WORD_RATIO = 1.3
        min_tokens = 32

        # Merge too-small chunks forward
        merged: list[list[str]] = []
        buffer: list[str] = []
        for chunk in chunks:
            buffer.extend(chunk)
            word_count = sum(len(s.split()) for s in buffer)
            if word_count * TOKEN_WORD_RATIO >= min_tokens:
                merged.append(buffer)
                buffer = []
        if buffer:
            if merged:
                merged[-1].extend(buffer)
            else:
                merged.append(buffer)

        # Split chunks that are too long
        final: list[list[str]] = []
        for chunk in merged:
            word_count = sum(len(s.split()) for s in chunk)
            if word_count * TOKEN_WORD_RATIO <= self.max_chunk_tokens:
                final.append(chunk)
            else:
                # Simple bisect: split in half until within limit
                final.extend(self._bisect_chunk(chunk))

        return final

    def _bisect_chunk(self, sentences: list[str]) -> list[list[str]]:
        TOKEN_WORD_RATIO = 1.3
        word_count = sum(len(s.split()) for s in sentences)
        if word_count * TOKEN_WORD_RATIO <= self.max_chunk_tokens or len(sentences) <= 1:
            return [sentences]
        mid = len(sentences) // 2
        left = self._bisect_chunk(sentences[:mid])
        right = self._bisect_chunk(sentences[mid:])
        return left + right

    def _make_chunk(
        self,
        original_doc: Document,
        sentences: list[str],
        sent_start: int,
        sent_end: int,
        chunk_index: int,
    ) -> Document:
        text = " ".join(sentences)
        metadata = {
            **original_doc.metadata,
            "chunk_index": chunk_index,
            "sentence_start": sent_start,
            "sentence_end": sent_end,
            "chunk_char_len": len(text),
        }
        return Document(page_content=text, metadata=metadata)

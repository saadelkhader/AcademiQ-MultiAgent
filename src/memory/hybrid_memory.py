"""Hybrid memory module combining session, persistent, and semantic retrieval.

This module separates storage concerns (session and persistent stores) from
orchestration logic in HybridMemory.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

try:
    import chromadb
except Exception as exc:  # pragma: no cover - handled at runtime
    chromadb = None
    _CHROMA_IMPORT_ERROR: Exception | None = exc
else:
    _CHROMA_IMPORT_ERROR = None


LOGGER = logging.getLogger("hybrid_memory")


def _log_event(level: int, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    LOGGER.log(level, json.dumps(payload, ensure_ascii=True))


class HybridMemoryError(Exception):
    """Base error for hybrid memory operations."""


class StorageError(HybridMemoryError):
    """Raised when a storage operation fails."""


class EmbeddingError(HybridMemoryError):
    """Raised when embedding computation fails."""


@dataclass
class MemoryRecord:
    """Single memory unit shared across session and persistent stores."""

    id: str
    content: str
    created_at: float
    session_id: str
    agent_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    ttl_seconds: int | None = None
    token_count: int | None = None
    compressed: bool = False
    embedding: list[float] | None = None
    persisted: bool = False


@dataclass
class RetrievedMemory:
    """Memory retrieval result with similarity score and source."""

    record: MemoryRecord
    similarity: float
    source: str


class EmbeddingProvider(Protocol):
    """Embedding provider contract for local embedding models."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class SentenceTransformerEmbeddingProvider:
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - runtime guard
                raise EmbeddingError("sentence-transformers is required") from exc
            self._model = SentenceTransformer(self._model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        try:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as exc:
            raise EmbeddingError("Failed to compute document embeddings") from exc

    def embed_query(self, text: str) -> list[float]:
        self._ensure_model()
        try:
            embedding = self._model.encode([text], normalize_embeddings=True)[0]
            return embedding.tolist()
        except Exception as exc:
            raise EmbeddingError("Failed to compute query embedding") from exc


class TokenEstimator:
    """Estimate token usage with tiktoken when available."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = None
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoding = None

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is None:
            return max(1, len(text) // 4)
        return len(self._encoding.encode(text))


class TextCompressor:
    """Compress long contexts by trimming head and tail."""

    def __init__(self, max_chars: int = 2000) -> None:
        self._max_chars = max_chars

    def compress(self, text: str) -> tuple[str, bool]:
        if len(text) <= self._max_chars:
            return text, False
        head_len = int(self._max_chars * 0.6)
        tail_len = int(self._max_chars * 0.3)
        head = text[:head_len].rstrip()
        tail = text[-tail_len:].lstrip()
        compressed = f"{head}\n...[truncated]...\n{tail}"
        return compressed, True


class SessionMemoryStore:
    """Volatile in-memory store for recent session context."""

    def __init__(self, default_ttl_seconds: int | None = None, max_items: int = 200) -> None:
        self._default_ttl_seconds = default_ttl_seconds
        self._max_items = max_items
        self._sessions: dict[str, list[MemoryRecord]] = {}

    def add_record(self, record: MemoryRecord) -> None:
        self._sessions.setdefault(record.session_id, []).append(record)
        self._prune_session(record.session_id)

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def get_context(
        self,
        session_id: str,
        agent_id: str | None = None,
        max_items: int | None = None,
    ) -> list[MemoryRecord]:
        self._prune_session(session_id)
        records = self._sessions.get(session_id, [])
        if agent_id is not None:
            records = [record for record in records if record.agent_id == agent_id]
        if max_items is not None:
            records = records[-max_items:]
        return list(records)

    def iter_records(self, session_id: str | None = None) -> Iterable[MemoryRecord]:
        if session_id is None:
            for records in self._sessions.values():
                yield from records
        else:
            yield from self._sessions.get(session_id, [])

    def semantic_search(
        self,
        query_embedding: list[float],
        embedder: EmbeddingProvider,
        top_k: int,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[RetrievedMemory]:
        candidates: list[RetrievedMemory] = []
        for record in self.iter_records(session_id=session_id):
            if agent_id is not None and record.agent_id != agent_id:
                continue
            if _is_expired(record, time.time()):
                continue
            if record.embedding is None:
                record.embedding = embedder.embed_documents([record.content])[0]
            similarity = cosine_similarity(query_embedding, record.embedding)
            candidates.append(RetrievedMemory(record=record, similarity=similarity, source="session"))
        candidates.sort(key=lambda item: item.similarity, reverse=True)
        return candidates[:top_k]

    def _prune_session(self, session_id: str) -> None:
        records = self._sessions.get(session_id, [])
        now = time.time()
        records = [record for record in records if not _is_expired(record, now)]
        if self._max_items is not None and len(records) > self._max_items:
            records = records[-self._max_items :]
        self._sessions[session_id] = records


class PersistentMemoryStore:
    """Persistent store backed by SQLite (metadata) and ChromaDB (embeddings)."""

    def __init__(
        self,
        sqlite_path: Path | str,
        chroma_path: Path | str,
        collection_name: str = "hybrid_memory",
    ) -> None:
        if chromadb is None:  # pragma: no cover - runtime guard
            raise StorageError("chromadb is required") from _CHROMA_IMPORT_ERROR

        self._sqlite_path = Path(sqlite_path)
        self._chroma_path = Path(chroma_path)
        self._collection_name = collection_name

        try:
            self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._sqlite_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._initialize_schema()
        except sqlite3.Error as exc:
            raise StorageError("Failed to initialize SQLite storage") from exc

        try:
            self._chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._chroma_path))
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise StorageError("Failed to initialize ChromaDB storage") from exc

    def persist_record(self, record: MemoryRecord, embedding: list[float]) -> None:
        try:
            metadata_json = json.dumps(record.metadata, ensure_ascii=True)
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_metadata (
                        id, session_id, agent_id, created_at, ttl_seconds,
                        token_count, compressed, content, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.session_id,
                        record.agent_id,
                        record.created_at,
                        record.ttl_seconds,
                        record.token_count,
                        1 if record.compressed else 0,
                        record.content,
                        metadata_json,
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError("Failed to persist metadata to SQLite") from exc

        metadata = _build_chroma_metadata(record)
        try:
            if hasattr(self._collection, "upsert"):
                self._collection.upsert(
                    ids=[record.id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[record.content],
                )
            else:  # pragma: no cover - backward compatibility
                try:
                    self._collection.add(
                        ids=[record.id],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        documents=[record.content],
                    )
                except Exception:
                    self._collection.delete(ids=[record.id])
                    self._collection.add(
                        ids=[record.id],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        documents=[record.content],
                    )
        except Exception as exc:
            raise StorageError("Failed to persist embeddings to ChromaDB") from exc

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[RetrievedMemory]:
        try:
            where_filter = _build_filter(session_id=session_id, agent_id=agent_id)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "documents", "distances", "embeddings"],
            )
        except Exception as exc:
            raise StorageError("Failed to query ChromaDB") from exc

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        embeddings = results.get("embeddings", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        records_by_id = self._get_records_by_ids(ids)
        now = time.time()
        retrieved: list[RetrievedMemory] = []

        for index, record_id in enumerate(ids):
            record = records_by_id.get(record_id)
            if record is None:
                record = _record_from_chroma(
                    record_id=record_id,
                    document=documents[index] if index < len(documents) else "",
                    metadata=metadatas[index] if index < len(metadatas) else {},
                )
            if _is_expired(record, now):
                continue
            similarity = _similarity_from_distance_or_embedding(
                query_embedding,
                embeddings[index] if index < len(embeddings) else None,
                distances[index] if index < len(distances) else None,
            )
            retrieved.append(RetrievedMemory(record=record, similarity=similarity, source="persistent"))

        retrieved.sort(key=lambda item: item.similarity, reverse=True)
        return retrieved

    def purge_expired(self) -> int:
        now = time.time()
        try:
            rows = self._conn.execute(
                """
                SELECT id, created_at, ttl_seconds FROM memory_metadata
                WHERE ttl_seconds IS NOT NULL
                """
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError("Failed to load records for purge") from exc

        expired_ids = [
            row["id"]
            for row in rows
            if row["ttl_seconds"] is not None and now - row["created_at"] > row["ttl_seconds"]
        ]
        if not expired_ids:
            return 0

        placeholders = ",".join("?" for _ in expired_ids)
        try:
            with self._conn:
                self._conn.execute(
                    f"DELETE FROM memory_metadata WHERE id IN ({placeholders})",
                    expired_ids,
                )
        except sqlite3.Error as exc:
            raise StorageError("Failed to purge expired records from SQLite") from exc

        try:
            self._collection.delete(ids=expired_ids)
        except Exception as exc:
            raise StorageError("Failed to purge expired records from ChromaDB") from exc

        return len(expired_ids)

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def _initialize_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    agent_id TEXT,
                    created_at REAL,
                    ttl_seconds INTEGER,
                    token_count INTEGER,
                    compressed INTEGER,
                    content TEXT,
                    metadata_json TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_session
                ON memory_metadata(session_id)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_agent
                ON memory_metadata(agent_id)
                """
            )

    def _get_records_by_ids(self, record_ids: list[str]) -> dict[str, MemoryRecord]:
        if not record_ids:
            return {}
        placeholders = ",".join("?" for _ in record_ids)
        try:
            rows = self._conn.execute(
                f"SELECT * FROM memory_metadata WHERE id IN ({placeholders})",
                record_ids,
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError("Failed to load records from SQLite") from exc

        records: dict[str, MemoryRecord] = {}
        for row in rows:
            records[row["id"]] = _record_from_sqlite_row(row)
        return records


class HybridMemory:
    """Orchestrates session memory, persistent memory, and semantic retrieval."""

    def __init__(
        self,
        sqlite_path: Path | str,
        chroma_path: Path | str,
        collection_name: str = "hybrid_memory",
        embedding_provider: EmbeddingProvider | None = None,
        session_ttl_seconds: int | None = None,
        session_max_items: int = 200,
        compress_max_chars: int = 2000,
    ) -> None:
        self._session_store = SessionMemoryStore(
            default_ttl_seconds=session_ttl_seconds,
            max_items=session_max_items,
        )
        self._persistent_store = PersistentMemoryStore(
            sqlite_path=sqlite_path,
            chroma_path=chroma_path,
            collection_name=collection_name,
        )
        self._embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider()
        self._token_estimator = TokenEstimator()
        self._compressor = TextCompressor(max_chars=compress_max_chars)

    def save_context(
        self,
        session_id: str,
        agent_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        persist: bool = True,
        ttl_seconds: int | None = None,
        compress: bool | None = None,
    ) -> MemoryRecord:
        if not content:
            raise HybridMemoryError("content must not be empty")

        metadata = metadata or {}
        ttl_value = ttl_seconds if ttl_seconds is not None else self._session_store._default_ttl_seconds
        compressed_text, is_compressed = self._compressor.compress(content)
        if compress is False:
            compressed_text, is_compressed = content, False

        record = MemoryRecord(
            id=str(uuid.uuid4()),
            content=compressed_text,
            created_at=time.time(),
            session_id=session_id,
            agent_id=agent_id,
            metadata=metadata,
            ttl_seconds=ttl_value,
            token_count=self._token_estimator.estimate(compressed_text),
            compressed=is_compressed,
            persisted=False,
        )

        try:
            record.embedding = self._embedding_provider.embed_documents([record.content])[0]
        except EmbeddingError as exc:
            _log_event(logging.ERROR, "embedding_failed", session_id=session_id, agent_id=agent_id)
            raise exc

        self._session_store.add_record(record)
        _log_event(
            logging.INFO,
            "session_record_added",
            session_id=session_id,
            agent_id=agent_id,
            record_id=record.id,
        )

        if persist:
            self._persist_record(record)

        return record

    def retrieve_relevant_history(
        self,
        query: str,
        top_k: int = 5,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[RetrievedMemory]:
        if not query:
            return []

        query_embedding = self._embedding_provider.embed_query(query)
        session_matches = self._session_store.semantic_search(
            query_embedding=query_embedding,
            embedder=self._embedding_provider,
            top_k=top_k,
            session_id=session_id,
            agent_id=agent_id,
        )
        persistent_matches = self._persistent_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            session_id=session_id,
            agent_id=agent_id,
        )
        combined = session_matches + persistent_matches
        combined.sort(key=lambda item: item.similarity, reverse=True)
        return combined[:top_k]

    def clear_session(self, session_id: str) -> None:
        self._session_store.clear_session(session_id)
        _log_event(logging.INFO, "session_cleared", session_id=session_id)

    def get_session_context(
        self,
        session_id: str,
        agent_id: str | None = None,
        max_items: int | None = None,
    ) -> list[MemoryRecord]:
        return self._session_store.get_context(
            session_id=session_id,
            agent_id=agent_id,
            max_items=max_items,
        )

    def persist_memory(self, session_id: str | None = None) -> int:
        persisted = 0
        for record in self._session_store.iter_records(session_id=session_id):
            if record.persisted:
                continue
            self._persist_record(record)
            persisted += 1
        return persisted

    def close(self) -> None:
        self._persistent_store.close()

    def _persist_record(self, record: MemoryRecord) -> None:
        if record.embedding is None:
            record.embedding = self._embedding_provider.embed_documents([record.content])[0]
        self._persistent_store.persist_record(record, record.embedding)
        record.persisted = True
        _log_event(
            logging.INFO,
            "record_persisted",
            session_id=record.session_id,
            agent_id=record.agent_id,
            record_id=record.id,
        )


def _build_chroma_metadata(record: MemoryRecord) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "session_id": record.session_id,
        "agent_id": record.agent_id,
        "created_at": record.created_at,
    }
    if record.ttl_seconds is not None:
        metadata["ttl_seconds"] = record.ttl_seconds
    if record.token_count is not None:
        metadata["token_count"] = record.token_count
    if record.compressed:
        metadata["compressed"] = True
    return metadata


def _build_filter(session_id: str | None, agent_id: str | None) -> dict[str, Any] | None:
    if session_id and agent_id:
        return {"$and": [{"session_id": session_id}, {"agent_id": agent_id}]}
    if session_id:
        return {"session_id": session_id}
    if agent_id:
        return {"agent_id": agent_id}
    return None


def _record_from_sqlite_row(row: sqlite3.Row) -> MemoryRecord:
    metadata = {}
    if row["metadata_json"]:
        try:
            metadata = json.loads(row["metadata_json"])
        except json.JSONDecodeError:
            metadata = {}
    return MemoryRecord(
        id=row["id"],
        content=row["content"],
        created_at=row["created_at"],
        session_id=row["session_id"],
        agent_id=row["agent_id"],
        metadata=metadata,
        ttl_seconds=row["ttl_seconds"],
        token_count=row["token_count"],
        compressed=bool(row["compressed"]),
        persisted=True,
    )


def _record_from_chroma(record_id: str, document: str, metadata: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=document,
        created_at=float(metadata.get("created_at", time.time())),
        session_id=str(metadata.get("session_id", "")),
        agent_id=str(metadata.get("agent_id", "")),
        metadata={},
        ttl_seconds=metadata.get("ttl_seconds"),
        token_count=metadata.get("token_count"),
        compressed=bool(metadata.get("compressed", False)),
        persisted=True,
    )


def _similarity_from_distance_or_embedding(
    query_embedding: list[float],
    embedding: list[float] | None,
    distance: float | None,
) -> float:
    if distance is not None:
        return max(0.0, 1.0 - distance)
    if embedding is not None:
        return cosine_similarity(query_embedding, embedding)
    return 0.0


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _is_expired(record: MemoryRecord, now: float) -> bool:
    ttl = record.ttl_seconds
    return ttl is not None and now - record.created_at > ttl

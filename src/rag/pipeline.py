from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("rag")


@dataclass
class Chunk:
    chunk_id: str
    source: str
    content: str
    metadata: dict[str, Any]


class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_by_paragraph(self, text: str, source: str = "unknown") -> list[Chunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []
        buffer = ""
        index = 0
        for para in paragraphs:
            if len(buffer) + len(para) < self.chunk_size:
                buffer = (buffer + "\n\n" + para).strip()
            else:
                if buffer:
                    chunks.append(Chunk(str(uuid.uuid4()), source, buffer, {"source": source, "chunk_index": index}))
                    index += 1
                buffer = para
        if buffer:
            chunks.append(Chunk(str(uuid.uuid4()), source, buffer, {"source": source, "chunk_index": index}))
        return chunks


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def load(self, path: Path | str) -> tuple[str, str]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path), str(path)
        return path.read_text(encoding="utf-8", errors="replace"), str(path)

    def _load_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except ImportError as exc:
                raise ImportError("Install pypdf: pip install pypdf") from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)


class RAGPipeline:
    def __init__(self, memory: Any, chunker: TextChunker | None = None, loader: DocumentLoader | None = None, session_id: str = "rag-default", agent_id: str = "rag") -> None:
        self._memory = memory
        self._chunker = chunker or TextChunker()
        self._loader = loader or DocumentLoader()
        self._session_id = session_id
        self._agent_id = agent_id

    def ingest_text(self, text: str, source: str = "manual") -> int:
        chunks = self._chunker.chunk_by_paragraph(text, source=source)
        return self._index_chunks(chunks)

    def ingest_file(self, path: Path | str) -> int:
        text, source = self._loader.load(path)
        chunks = self._chunker.chunk_by_paragraph(text, source=source)
        LOGGER.info("Ingested %d chunks from %s", len(chunks), source)
        return self._index_chunks(chunks)

    def ingest_directory(self, directory: Path | str, recursive: bool = False) -> int:
        directory = Path(directory)
        total = 0
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
                try:
                    total += self.ingest_file(file_path)
                except Exception as exc:
                    LOGGER.warning("Skipping %s: %s", file_path, exc)
        return total

    def _index_chunks(self, chunks: list[Chunk]) -> int:
        stored = 0
        for chunk in chunks:
            try:
                self._memory.save_context(session_id=self._session_id, agent_id=self._agent_id, content=chunk.content, metadata=chunk.metadata, persist=True)
                stored += 1
            except Exception as exc:
                LOGGER.error("Failed to index chunk %s: %s", chunk.chunk_id, exc)
        return stored

    def retrieve(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> list[dict[str, Any]]:
        results = self._memory.retrieve_relevant_history(query=query, top_k=top_k, session_id=self._session_id, agent_id=self._agent_id)
        output = []
        for r in results:
            if r.similarity >= min_similarity:
                output.append({"content": r.record.content, "similarity": round(r.similarity, 4), "source": r.record.metadata.get("source", "unknown"), "source_type": r.source})
        return output

    def build_context_block(self, query: str, top_k: int = 5, min_similarity: float = 0.1) -> str:
        results = self.retrieve(query, top_k=top_k, min_similarity=min_similarity)
        if not results:
            return "(no relevant documents found)"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] (score={r['similarity']:.2f}, source={r['source']})\n{r['content']}")
        return "\n\n---\n\n".join(lines)

    def stats(self) -> dict[str, Any]:
        try:
            ctx = self._memory.get_session_context(session_id=self._session_id, agent_id=self._agent_id)
            return {"status": "ok", "session_id": self._session_id, "stored_items": len(ctx), "agent_id": self._agent_id}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

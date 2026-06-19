from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..registry import ToolSpec


LOGGER = logging.getLogger("mcp_file_access")


class FileAccessInput(BaseModel):
    relative_path: str
    max_chars: int = 20000


class FileAccessOutput(BaseModel):
    text: str
    metadata: dict[str, Any]
    trace_id: str | None = None


class FileAccessError(Exception):
    pass


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


def _resolve_path(relative_path: str) -> Path:
    data_dir = _data_dir()
    candidate = (data_dir / relative_path).resolve()
    try:
        candidate.relative_to(data_dir)
    except ValueError as exc:
        raise FileAccessError("path outside data directory") from exc
    return candidate


def _read_pdf(path: Path) -> str:
    reader = None
    try:
        from pypdf import PdfReader

        reader = PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader
        except Exception as exc:
            raise FileAccessError("PDF reader not installed") from exc
    pdf = reader(str(path))
    texts = []
    for page in pdf.pages:
        text = page.extract_text() or ""
        texts.append(text)
    return "\n".join(texts)


def _read_text(path: Path, max_chars: int) -> str:
    if path.suffix.lower() == ".pdf":
        content = _read_pdf(path)
    else:
        content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) > max_chars:
        content = content[:max_chars] + "...[truncated]"
    return content


def _handler(payload: FileAccessInput, trace_id: str | None) -> FileAccessOutput:
    path = _resolve_path(payload.relative_path)
    if not path.exists():
        raise FileAccessError("file not found")
    if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
        raise FileAccessError("unsupported file type")
    text = _read_text(path, payload.max_chars)
    output = FileAccessOutput(
        text=text,
        metadata={
            "path": str(path),
            "extension": path.suffix.lower(),
            "chars": len(text),
        },
        trace_id=trace_id,
    )
    LOGGER.info(
        json.dumps(
            {
                "event": "file_access",
                "trace_id": trace_id,
                "path": str(path),
                "chars": len(text),
            },
            ensure_ascii=True,
        )
    )
    return output


TOOL_SPEC = ToolSpec(
    name="file_access",
    description="Read data files from the data directory (pdf, txt, md).",
    input_model=FileAccessInput,
    output_model=FileAccessOutput,
    handler=_handler,
    capabilities={"data_dir": "data", "extensions": ["pdf", "txt", "md"]},
)

"""Project scaffolding script.

Creates the required folder structure and placeholder files in an idempotent way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent

TOP_LEVEL_DIRS = [
    "docs",
    "data",
    "tests",
    "scripts",
    "src",
]

SRC_SUBDIRS = [
    "orchestrator",
    "agents",
    "memory",
    "rag",
    "mcp_servers",
    "tools",
    "models",
    "utils",
    "api",
]

REQUIREMENTS = [
    "langchain",
    "langchain-community",
    "chromadb",
    "ollama",
    "pydantic",
    "fastapi",
    "uvicorn",
    "streamlit",
    "sentence-transformers",
    "sqlite-utils",
    "mcp",
    "tiktoken",
    "pytest",
    "python-dotenv",
]


class SetupError(Exception):
    """Raised when project setup fails."""


def ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SetupError(f"Failed to create directory: {path}") from exc


def ensure_file(path: Path, content: str | None = None) -> None:
    try:
        if not path.exists():
            path.write_text(content or "", encoding="utf-8")
    except OSError as exc:
        raise SetupError(f"Failed to create file: {path}") from exc


def ensure_init_files(paths: Iterable[Path]) -> None:
    for package_dir in paths:
        ensure_file(package_dir / "__init__.py")


def main() -> int:
    try:
        for dir_name in TOP_LEVEL_DIRS:
            ensure_dir(ROOT_DIR / dir_name)

        src_root = ROOT_DIR / "src"
        src_dirs = [src_root / name for name in SRC_SUBDIRS]
        for src_dir in src_dirs:
            ensure_dir(src_dir)

        ensure_init_files([src_root, *src_dirs])

        requirements_path = ROOT_DIR / "requirements.txt"
        ensure_file(requirements_path, "\n".join(REQUIREMENTS) + "\n")

        ensure_file(ROOT_DIR / "README.md", "# Project\n")
    except SetupError as exc:
        print(f"Setup failed: {exc}")
        return 1

    print("Project structure created successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import ast
import contextlib
import io
import math
import multiprocessing
import statistics
import time
from dataclasses import dataclass
from typing import Any

try:
    import resource
except Exception:
    resource = None


SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "str": str,
    "sum": sum,
}

SAFE_MODULES: dict[str, Any] = {
    "math": math,
    "statistics": statistics,
}

ALLOWED_IMPORTS = set(SAFE_MODULES.keys())
BLOCKED_CALLS = {"open", "exec", "eval", "compile", "input", "__import__"}


class SandboxError(Exception):
    pass


@dataclass
class SandboxResult:
    stdout: str
    result: Any
    error: str | None
    timed_out: bool
    duration_ms: float


def run_code(code: str, timeout_seconds: float, memory_mb: int, max_output_chars: int) -> SandboxResult:
    started = time.time()
    queue: multiprocessing.Queue[dict[str, Any]] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_worker,
        args=(code, memory_mb, max_output_chars, queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return SandboxResult(
            stdout="",
            result=None,
            error="timeout",
            timed_out=True,
            duration_ms=_ms(started),
        )
    payload = _get_queue_payload(queue)
    return SandboxResult(
        stdout=payload.get("stdout", ""),
        result=payload.get("result"),
        error=payload.get("error"),
        timed_out=False,
        duration_ms=_ms(started),
    )


def _worker(code: str, memory_mb: int, max_output_chars: int, queue: multiprocessing.Queue) -> None:
    if resource is not None:
        _apply_memory_limit(memory_mb)
    stdout = io.StringIO()
    result = None
    error = None
    try:
        tree = ast.parse(code, mode="exec")
        _validate_ast(tree)
        compiled = compile(tree, "<sandbox>", "exec")
        globals_dict = {"__builtins__": SAFE_BUILTINS, **SAFE_MODULES}
        locals_dict: dict[str, Any] = {}
        with contextlib.redirect_stdout(stdout):
            exec(compiled, globals_dict, locals_dict)
        result = locals_dict.get("result")
    except Exception as exc:
        error = str(exc)
    out = stdout.getvalue()
    if len(out) > max_output_chars:
        out = out[:max_output_chars] + "...[truncated]"
    queue.put({"stdout": out, "result": result, "error": error})


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base not in ALLOWED_IMPORTS:
                    raise SandboxError(f"import not allowed: {alias.name}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise SandboxError("dunder attribute not allowed")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise SandboxError("dunder name not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                raise SandboxError(f"call not allowed: {node.func.id}")


def _apply_memory_limit(memory_mb: int) -> None:
    if memory_mb <= 0:
        return
    limit = int(memory_mb) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        return


def _get_queue_payload(queue: multiprocessing.Queue) -> dict[str, Any]:
    try:
        return queue.get_nowait()
    except Exception:
        return {"stdout": "", "result": None, "error": "no output"}


def _ms(started: float) -> float:
    return round((time.time() - started) * 1000.0, 3)

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from ..registry import ToolSpec
from ..sandbox import SandboxError, run_code


LOGGER = logging.getLogger("mcp_python_executor")


class PythonExecutorInput(BaseModel):
    code: str
    timeout_seconds: float = 2.0
    memory_mb: int = 128
    max_output_chars: int = 4000


class PythonExecutorOutput(BaseModel):
    stdout: str
    result: Any = None
    error: str | None = None
    timed_out: bool = False
    duration_ms: float = 0.0
    trace_id: str | None = None


def _handler(payload: PythonExecutorInput, trace_id: str | None) -> PythonExecutorOutput:
    if not payload.code:
        raise SandboxError("code is empty")
    result = run_code(
        code=payload.code,
        timeout_seconds=payload.timeout_seconds,
        memory_mb=payload.memory_mb,
        max_output_chars=payload.max_output_chars,
    )
    output = PythonExecutorOutput(
        stdout=result.stdout,
        result=result.result,
        error=result.error,
        timed_out=result.timed_out,
        duration_ms=result.duration_ms,
        trace_id=trace_id,
    )
    LOGGER.info(
        json.dumps(
            {
                "event": "python_executor",
                "trace_id": trace_id,
                "timed_out": output.timed_out,
                "duration_ms": output.duration_ms,
            },
            ensure_ascii=True,
        )
    )
    return output


TOOL_SPEC = ToolSpec(
    name="python_executor",
    description="Execute safe Python code for math and data processing.",
    input_model=PythonExecutorInput,
    output_model=PythonExecutorOutput,
    handler=_handler,
    capabilities={
        "sandbox": True,
        "timeout": True,
        "memory_limit": True,
        "allowed_imports": ["math", "statistics"],
    },
)

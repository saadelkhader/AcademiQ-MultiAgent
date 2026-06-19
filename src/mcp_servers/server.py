from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException

from .registry import ToolRegistry
from .schemas import DiscoveryResponse, ToolInvocationRequest, ToolInvocationResponse
from .tools.file_access import TOOL_SPEC as FILE_ACCESS_TOOL
from .tools.python_executor import TOOL_SPEC as PYTHON_EXECUTOR_TOOL


LOGGER = logging.getLogger("mcp_server")

app = FastAPI(title="MCP Server", version="0.1.0")
registry = ToolRegistry(logger=LOGGER)
registry.register(PYTHON_EXECUTOR_TOOL)
registry.register(FILE_ACCESS_TOOL)


@app.get("/mcp/discover", response_model=DiscoveryResponse)
def discover() -> DiscoveryResponse:
    return DiscoveryResponse(
        name="agentic-mcp",
        version="0.1.0",
        description="MCP server exposing agent tools.",
        tools=registry.list_tools(),
        capabilities={"tools": True, "async": True},
    )


@app.post("/mcp/tools/{tool_name}/invoke", response_model=ToolInvocationResponse)
async def invoke(tool_name: str, request: ToolInvocationRequest) -> ToolInvocationResponse:
    started = time.time()
    trace_id = request.trace_id
    try:
        result = await registry.invoke(tool_name, request.input, trace_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="tool not found")
    except Exception as exc:
        return ToolInvocationResponse(
            ok=False,
            error=str(exc),
            trace_id=trace_id,
            duration_ms=_ms(started),
        )
    return ToolInvocationResponse(
        ok=True,
        data=_model_dump(result),
        trace_id=trace_id,
        duration_ms=_ms(started),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    payload = {"status": "ok"}
    LOGGER.info(json.dumps({"event": "health_check"}, ensure_ascii=True))
    return payload


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return {"value": model}


def _ms(started: float) -> float:
    return round((time.time() - started) * 1000.0, 3)

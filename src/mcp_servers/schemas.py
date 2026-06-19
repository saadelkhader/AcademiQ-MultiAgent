from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    capabilities: dict[str, Any] = Field(default_factory=dict)


class DiscoveryResponse(BaseModel):
    name: str
    version: str
    description: str
    tools: list[ToolSchema]
    capabilities: dict[str, Any] = Field(default_factory=dict)


class ToolInvocationRequest(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


class ToolInvocationResponse(BaseModel):
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    trace_id: str | None = None
    duration_ms: float | None = None

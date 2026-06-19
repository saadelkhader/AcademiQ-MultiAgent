from __future__ import annotations

import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Type

from pydantic import BaseModel

from .schemas import ToolSchema


ToolHandler = Callable[[BaseModel, str | None], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    handler: ToolHandler
    capabilities: dict[str, Any]

    def to_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=_model_schema(self.input_model),
            output_schema=_model_schema(self.output_model),
            capabilities=self.capabilities,
        )


class ToolRegistry:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("mcp_registry")
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec
        self._log_event("tool_registered", tool=spec.name)

    def list_tools(self) -> list[ToolSchema]:
        return [spec.to_schema() for spec in self._tools.values()]

    async def invoke(self, tool_name: str, payload: dict[str, Any], trace_id: str | None) -> BaseModel:
        spec = self._tools.get(tool_name)
        if spec is None:
            raise KeyError(tool_name)
        model = spec.input_model(**payload)
        started = time.time()
        try:
            result = spec.handler(model, trace_id)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            self._log_event(
                "tool_error",
                tool=tool_name,
                trace_id=trace_id,
                duration_ms=_ms(started),
                error=str(exc),
            )
            raise
        self._log_event(
            "tool_success",
            tool=tool_name,
            trace_id=trace_id,
            duration_ms=_ms(started),
        )
        return result

    def _log_event(self, event: str, **fields: Any) -> None:
        payload = {"event": event, **fields}
        self._logger.info(json.dumps(payload, ensure_ascii=True))


def _model_schema(model: Type[BaseModel]) -> dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


def _ms(started: float) -> float:
    return round((time.time() - started) * 1000.0, 3)

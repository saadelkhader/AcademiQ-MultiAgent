from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Protocol

from pydantic import BaseModel, Field

try:
    import ollama
except Exception as exc:  # pragma: no cover
    ollama = None
    _OLLAMA_IMPORT_ERROR: Exception | None = exc
else:
    _OLLAMA_IMPORT_ERROR = None


LOGGER = logging.getLogger("a2a")

A2A_SCHEMA = {
    "agent": {"id": "str", "role": "str"},
    "message": {
        "message_id": "str",
        "sender": "str",
        "receiver": "str",
        "task_type": "str",
        "content": "str",
        "metadata": "dict",
        "timestamp": "float",
        "priority": "int",
        "trace_id": "str",
    },
    "flow": ["sender -> bus -> receiver"],
}


class AgentError(Exception):
    pass


class MessageBusError(AgentError):
    pass


class LLMError(AgentError):
    pass


class AgentMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    task_type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    priority: int = 0
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class MessageBus(Protocol):
    def register(self, agent_id: str) -> None:
        ...

    async def send(self, message: AgentMessage) -> None:
        ...

    async def receive(self, receiver: str) -> AgentMessage:
        ...


class InMemoryMessageBus:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[AgentMessage]] = {}

    def register(self, agent_id: str) -> None:
        self._queues.setdefault(agent_id, asyncio.Queue())

    async def send(self, message: AgentMessage) -> None:
        queue = self._queues.get(message.receiver)
        if queue is None:
            raise MessageBusError(f"receiver not registered: {message.receiver}")
        await queue.put(message)

    async def receive(self, receiver: str) -> AgentMessage:
        queue = self._queues.get(receiver)
        if queue is None:
            raise MessageBusError(f"receiver not registered: {receiver}")
        return await queue.get()


class BaseAgent:
    def __init__(
        self,
        agent_id: str,
        role: str,
        bus: MessageBus,
        llm_model: str = "llama3",
        logger: logging.Logger | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.llm_model = llm_model
        self._bus = bus
        self._bus.register(agent_id)
        self._logger = logger or LOGGER

    async def send_message(
        self,
        receiver: str,
        task_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        priority: int = 0,
        trace_id: str | None = None,
    ) -> AgentMessage:
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            task_type=task_type,
            content=content,
            metadata=metadata or {},
            priority=priority,
            trace_id=trace_id or str(uuid.uuid4()),
        )
        started = time.time()
        try:
            await self._bus.send(message)
            status = "sent"
        except Exception as exc:
            status = "error"
            self.log_interaction(message, status=status, duration_ms=_ms(started), error=str(exc))
            raise
        self.log_interaction(message, status=status, duration_ms=_ms(started))
        return message

    async def receive_message(self, timeout: float | None = None) -> AgentMessage:
        started = time.time()
        try:
            if timeout is None:
                message = await self._bus.receive(self.agent_id)
            else:
                message = await asyncio.wait_for(self._bus.receive(self.agent_id), timeout)
        except asyncio.TimeoutError as exc:
            self._log_event("receive_timeout", duration_ms=_ms(started))
            raise MessageBusError("receive timeout") from exc
        except Exception as exc:
            self._log_event("receive_error", duration_ms=_ms(started), error=str(exc))
            raise
        self.log_interaction(message, status="received", duration_ms=_ms(started))
        return message

    async def process_task(self, message: AgentMessage) -> AgentMessage:
        started = time.time()
        try:
            response_content = await self.generate_response(message)
            response = AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                task_type=f"{message.task_type}.response",
                content=response_content,
                metadata={"in_reply_to": message.message_id},
                trace_id=message.trace_id,
            )
            await self._bus.send(response)
        except Exception as exc:
            self.log_interaction(message, status="error", duration_ms=_ms(started), error=str(exc))
            raise
        self.log_interaction(message, status="processed", duration_ms=_ms(started))
        return response

    async def generate_response(self, message: AgentMessage) -> str:
        if ollama is None:
            raise LLMError("ollama is required") from _OLLAMA_IMPORT_ERROR
        payload = [
            {"role": "system", "content": f"You are a {self.role} agent."},
            {"role": "user", "content": message.content},
        ]
        started = time.time()
        try:
            # run the blocking ollama.chat in a thread and bound with a longer timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(ollama.chat, model=self.llm_model, messages=payload),
                timeout=60.0,
            )
        except asyncio.TimeoutError as exc:
            self._log_event("llm_timeout", duration_ms=_ms(started), error=str(exc))
            raise LLMError("ollama chat timed out") from exc
        except Exception as exc:
            self._log_event("llm_error", duration_ms=_ms(started), error=str(exc))
            raise LLMError("ollama chat failed") from exc
        self._log_event("llm_call", duration_ms=_ms(started))
        content = result.get("message", {}).get("content", "")
        return content

    def log_interaction(
        self,
        message: AgentMessage,
        status: str,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "event": "agent_message",
            "agent_id": self.agent_id,
            "role": self.role,
            "message_id": message.message_id,
            "sender": message.sender,
            "receiver": message.receiver,
            "task_type": message.task_type,
            "trace_id": message.trace_id,
            "status": status,
            "priority": message.priority,
            "timestamp": message.timestamp,
        }
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if error:
            payload["error"] = error
        self._logger.info(json.dumps(payload, ensure_ascii=True))

    def _log_event(self, event: str, **fields: Any) -> None:
        payload = {"event": event, "agent_id": self.agent_id, "role": self.role, **fields}
        self._logger.info(json.dumps(payload, ensure_ascii=True))


def _ms(started: float) -> float:
    return round((time.time() - started) * 1000.0, 3)


async def _example() -> None:
    bus = InMemoryMessageBus()
    planner = BaseAgent("planner", "planning", bus, llm_model="llama3")
    verifier = BaseAgent("verifier", "verification", bus, llm_model="llama3")

    await planner.send_message("verifier", "review", "Check the plan", trace_id=str(uuid.uuid4()))
    incoming = await verifier.receive_message()
    await verifier.process_task(incoming)
    response = await planner.receive_message()
    print(response.content)


if __name__ == "__main__":
    asyncio.run(_example())

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from pydantic import BaseModel, Field

try:
    import ollama
except Exception as exc:  # pragma: no cover
    ollama = None
    _OLLAMA_IMPORT_ERROR: Exception | None = exc
else:
    _OLLAMA_IMPORT_ERROR = None


LOGGER = logging.getLogger("dynamic_router")
USE_ROUTER_LLM = os.getenv("A2A_ROUTER_LLM", "0") == "1"


class RouterError(Exception):
    pass


class LLMRouterError(RouterError):
    pass


class AgentState(BaseModel):
    agent_id: str
    role: str = ""
    capabilities: list[str] = Field(default_factory=list)
    context_tokens: int = 0
    max_context_tokens: int = 8192
    base_confidence: float = 0.5
    tool_ready: bool = True
    availability: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryState(BaseModel):
    session_id: str | None = None
    token_count: int = 0
    history_size: int = 0
    agent_load: dict[str, float] = Field(default_factory=dict)


class SystemMetrics(BaseModel):
    cpu_load: float = 0.0
    mem_load: float = 0.0
    latency_ms: float = 0.0


class RoutingDecision(BaseModel):
    selected_agent: str
    confidence_score: float
    uncertainty_score: float
    context_load: float
    reasoning: str
    fallback_agent: str | None = None
    timestamp: float = Field(default_factory=time.time)


class RouterConfig(BaseModel):
    uncertainty_weight: float = 0.4
    context_weight: float = 0.25
    confidence_weight: float = 0.35
    capability_weight: float = 0.2
    availability_weight: float = 0.1
    tool_weight: float = 0.1
    system_weight: float = 0.1
    fallback_threshold: float = 0.4
    max_context_tokens: int = 8192


class TokenEstimator:
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


class DynamicRouter:
    def __init__(
        self,
        llm_model: str = "llama3",
        config: RouterConfig | None = None,
        token_estimator: TokenEstimator | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._llm_model = llm_model
        self._config = config or RouterConfig()
        self._token_estimator = token_estimator or TokenEstimator()
        self._logger = logger or LOGGER
        self._stats: dict[str, dict[str, float]] = {}

    async def route(
        self,
        query: str,
        memory_state: MemoryState | None,
        agents_state: list[AgentState],
        history: list[Any] | None = None,
        system_metrics: SystemMetrics | None = None,
    ) -> RoutingDecision:
        if not agents_state:
            raise RouterError("agents_state must not be empty")
        started = time.time()
        uncertainty = await self._evaluate_uncertainty(query)
        history_tokens, history_size = self._history_stats(history or [])
        system_load = self._system_load(system_metrics)
        base_context = self._base_context_load(memory_state, history_tokens, history_size, system_load)
        scored = []
        for agent in agents_state:
            context_load = self._agent_context_load(agent, memory_state, base_context)
            score = self._score_agent(agent, uncertainty, context_load, system_load, query)
            scored.append((agent, score, context_load))
        scored.sort(key=lambda item: item[1], reverse=True)
        selected, score, context_load = scored[0]
        fallback = scored[1][0].agent_id if len(scored) > 1 else None
        if score < self._config.fallback_threshold and fallback:
            fallback_agent = selected.agent_id
            selected_agent = fallback
        else:
            fallback_agent = fallback
            selected_agent = selected.agent_id
        reasoning = (
            f"score={score:.3f} uncertainty={uncertainty:.3f} "
            f"context_load={context_load:.3f}"
        )
        decision = RoutingDecision(
            selected_agent=selected_agent,
            confidence_score=_clamp(score),
            uncertainty_score=_clamp(uncertainty),
            context_load=_clamp(context_load),
            reasoning=reasoning,
            fallback_agent=fallback_agent,
        )
        self._log_event(
            "routing_decision",
            selected_agent=decision.selected_agent,
            fallback_agent=decision.fallback_agent,
            confidence_score=decision.confidence_score,
            uncertainty_score=decision.uncertainty_score,
            context_load=decision.context_load,
            duration_ms=_ms(started),
        )
        return decision

    async def route_multi(
        self,
        query: str,
        memory_state: MemoryState | None,
        agents_state: list[AgentState],
        history: list[Any] | None = None,
        system_metrics: SystemMetrics | None = None,
        top_k: int = 2,
    ) -> list[RoutingDecision]:
        if top_k < 1:
            return []
        decision = await self.route(query, memory_state, agents_state, history, system_metrics)
        if top_k == 1:
            return [decision]
        remaining = [a for a in agents_state if a.agent_id != decision.selected_agent]
        if not remaining:
            return [decision]
        second = await self.route(query, memory_state, remaining, history, system_metrics)
        return [decision, second][:top_k]

    def register_feedback(self, agent_id: str, success: bool, reward: float = 1.0) -> None:
        stats = self._stats.setdefault(agent_id, {"count": 0.0, "success": 0.0, "reward": 0.0})
        stats["count"] += 1.0
        stats["success"] += 1.0 if success else 0.0
        stats["reward"] += reward

    async def _evaluate_uncertainty(self, query: str) -> float:
        if not USE_ROUTER_LLM:
            return self._heuristic_uncertainty(query)
        if ollama is None:
            self._log_event("llm_unavailable", error=str(_OLLAMA_IMPORT_ERROR))
            return self._heuristic_uncertainty(query)
        prompt = (
            "Return a single number between 0 and 1 for uncertainty. "
            "0 means clear. 1 means ambiguous or missing info. Query: "
            f"{query}"
        )
        started = time.time()
        try:
            # run blocking call in thread with timeout to avoid hanging router
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    ollama.chat,
                    model=self._llm_model,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            self._log_event("llm_timeout", duration_ms=_ms(started), error="uncertainty eval timeout")
            return self._heuristic_uncertainty(query)
        except Exception as exc:
            self._log_event("llm_error", duration_ms=_ms(started), error=str(exc))
            return self._heuristic_uncertainty(query)
        self._log_event("llm_uncertainty", duration_ms=_ms(started))
        content = result.get("message", {}).get("content", "")
        value = _parse_float(content)
        if value is None:
            return self._heuristic_uncertainty(query)
        return _clamp(value)

    def _heuristic_uncertainty(self, query: str) -> float:
        """Estimate query uncertainty from lexical features."""
        text = query.lower()
        score = 0.10
        if "?" in query:
            score += 0.05
        words = query.split()
        if len(words) < 4:
            score += 0.25  # very short = ambiguous
        elif len(words) < 8:
            score += 0.10
        if any(k in text for k in ["maybe", "unclear", "ambiguous", "not sure", "unknown", "perhaps"]):
            score += 0.30
        # Prefix hints (injected by orchestrator for step routing)
        if any(k in text for k in ["plan", "decompose", "organize", "structure", "task"]):
            score -= 0.05   # clear intent
        if any(k in text for k in ["retrieve", "search", "find", "document", "source"]):
            score -= 0.05
        if any(k in text for k in ["write", "synthesize", "explain", "answer", "generate"]):
            score -= 0.05
        if any(k in text for k in ["verify", "check", "validate", "review", "quality"]):
            score -= 0.05
        return _clamp(score)

    def _history_stats(self, history: list[Any]) -> tuple[int, int]:
        tokens = 0
        size = len(history)
        for item in history:
            if isinstance(item, str):
                tokens += self._token_estimator.estimate(item)
            elif isinstance(item, dict):
                content = item.get("content") or item.get("text") or ""
                if content:
                    tokens += self._token_estimator.estimate(str(content))
        return tokens, size

    def _system_load(self, system_metrics: SystemMetrics | None) -> float:
        if system_metrics is None:
            return 0.0
        return _clamp(max(system_metrics.cpu_load, system_metrics.mem_load))

    def _base_context_load(
        self,
        memory_state: MemoryState | None,
        history_tokens: int,
        history_size: int,
        system_load: float,
    ) -> float:
        token_count = history_tokens
        if memory_state is not None:
            token_count += memory_state.token_count
            history_size = max(history_size, memory_state.history_size)
        token_ratio = token_count / max(1, self._config.max_context_tokens)
        history_ratio = history_size / 50.0
        return _clamp(0.6 * token_ratio + 0.2 * history_ratio + 0.2 * system_load)

    def _agent_context_load(
        self,
        agent: AgentState,
        memory_state: MemoryState | None,
        base_context: float,
    ) -> float:
        saturation = agent.context_tokens / max(1, agent.max_context_tokens)
        extra = 0.0
        if memory_state is not None:
            extra = memory_state.agent_load.get(agent.agent_id, 0.0)
        return _clamp(base_context + 0.4 * saturation + 0.3 * extra)

    def _score_agent(
        self,
        agent: AgentState,
        uncertainty: float,
        context_load: float,
        system_load: float,
        query: str,
    ) -> float:
        cap_score = self._capability_score(agent.capabilities, query)
        tool_score = 1.0 if agent.tool_ready else 0.0
        score = (
            self._config.confidence_weight * agent.base_confidence
            + self._config.capability_weight * cap_score
            + self._config.availability_weight * agent.availability
            + self._config.tool_weight * tool_score
            - self._config.context_weight * context_load
            - self._config.uncertainty_weight * uncertainty
            - self._config.system_weight * system_load
        )
        self._log_event(
            "agent_score",
            agent_id=agent.agent_id,
            score=_clamp(score),
            capability=cap_score,
            context_load=context_load,
            uncertainty=uncertainty,
        )
        return _clamp(score)

    def _capability_score(self, capabilities: list[str], query: str) -> float:
        """Score agent capabilities against the query using keyword matching.

        Uses a broader match: capability keywords that appear anywhere in the
        query tokens (including partial word matches) receive a higher weight.
        """
        if not capabilities:
            return 0.0
        text = query.lower()
        tokens = set(text.split())
        matches = 0.0
        for cap in capabilities:
            cap_l = cap.lower()
            # exact token match
            if cap_l in tokens:
                matches += 1.0
            # substring match (e.g. "decompose" matches "decomposing")
            elif any(cap_l in tok or tok in cap_l for tok in tokens if len(tok) > 3):
                matches += 0.5
        return _clamp(matches / max(1, len(capabilities)))

    def _log_event(self, event: str, **fields: Any) -> None:
        payload = {"event": event, **fields}
        self._logger.info(json.dumps(payload, ensure_ascii=True))


def _parse_float(text: str) -> float | None:
    buffer = []
    for ch in text:
        if ch.isdigit() or ch in ".-":
            buffer.append(ch)
        elif buffer:
            break
    try:
        return float("".join(buffer)) if buffer else None
    except ValueError:
        return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ms(started: float) -> float:
    return round((time.time() - started) * 1000.0, 3)

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


try:
    from src.orchestrator.dynamic_router import (
        AgentState,
        DynamicRouter,
        MemoryState,
        SystemMetrics,
    )
except Exception:
    AgentState = None
    DynamicRouter = None
    MemoryState = None
    SystemMetrics = None


LOG = logging.getLogger("benchmark")


@dataclass(frozen=True)
class BenchmarkQuestion:
    qid: str
    category: str
    query: str
    expected_keywords: list[str]


@dataclass(frozen=True)
class AgentProfile:
    agent_id: str
    role: str
    capabilities: list[str]
    base_confidence: float
    max_context_tokens: int = 8192
    tool_ready: bool = True
    availability: float = 1.0


@dataclass
class BenchmarkContext:
    history: list[dict[str, Any]]
    agent_context_tokens: dict[str, int]
    memory_tokens: int


@dataclass(frozen=True)
class RouteSelection:
    selected_agents: list[str]
    confidence_score: float
    uncertainty_score: float
    context_load: float
    reasoning: str
    fallback_agent: str | None


@dataclass
class RunResult:
    config_name: str
    run_id: str
    question_id: str
    category: str
    latency_ms: float
    agent_calls: int
    failed: bool
    tokens: int
    quality_score: float
    context_cost: float
    selected_agents: list[str]
    fallback_used: bool


class TokenEstimator:
    def estimate(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)


class FixedHierarchicalOrchestrator:
    name = "fixed_hierarchical"

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    async def select_agents(self, question: BenchmarkQuestion, context: BenchmarkContext) -> RouteSelection:
        agent = self._mapping.get(question.category, "planner")
        uncertainty = 0.2 if question.category != "ambiguity" else 0.6
        return RouteSelection(
            selected_agents=[agent],
            confidence_score=0.6,
            uncertainty_score=uncertainty,
            context_load=_context_load(context),
            reasoning=f"fixed mapping for {question.category}",
            fallback_agent=None,
        )


class AdaptiveDynamicOrchestrator:
    name = "dynamic_adaptive"

    def __init__(self, agents: list[AgentProfile], multi_agent: bool = False) -> None:
        self._agents = agents
        self._multi_agent = multi_agent
        self._router = DynamicRouter() if DynamicRouter is not None else None

    async def select_agents(self, question: BenchmarkQuestion, context: BenchmarkContext) -> RouteSelection:
        if self._router is None or AgentState is None:
            return self._fallback_select(question, context)
        agents_state = [
            AgentState(
                agent_id=a.agent_id,
                role=a.role,
                capabilities=a.capabilities,
                context_tokens=context.agent_context_tokens.get(a.agent_id, 0),
                max_context_tokens=a.max_context_tokens,
                base_confidence=a.base_confidence,
                tool_ready=a.tool_ready,
                availability=a.availability,
            )
            for a in self._agents
        ]
        memory_state = None
        if MemoryState is not None:
            memory_state = MemoryState(
                session_id="bench",
                token_count=context.memory_tokens,
                history_size=len(context.history),
                agent_load={k: v / 10000.0 for k, v in context.agent_context_tokens.items()},
            )
        system_metrics = None
        if SystemMetrics is not None:
            system_metrics = SystemMetrics(cpu_load=0.3, mem_load=0.4, latency_ms=12.0)
        decision = await self._router.route(
            query=question.query,
            memory_state=memory_state,
            agents_state=agents_state,
            history=context.history,
            system_metrics=system_metrics,
        )
        selected = [decision.selected_agent]
        if self._multi_agent and decision.fallback_agent and decision.fallback_agent not in selected:
            selected.append(decision.fallback_agent)
        return RouteSelection(
            selected_agents=selected,
            confidence_score=decision.confidence_score,
            uncertainty_score=decision.uncertainty_score,
            context_load=decision.context_load,
            reasoning=decision.reasoning,
            fallback_agent=decision.fallback_agent,
        )

    def _fallback_select(self, question: BenchmarkQuestion, context: BenchmarkContext) -> RouteSelection:
        mapping = {
            "reasoning": "planner",
            "retrieval": "retriever",
            "calculation": "tool",
            "synthesis": "synth",
            "ambiguity": "clarifier",
        }
        agent = mapping.get(question.category, "planner")
        return RouteSelection(
            selected_agents=[agent],
            confidence_score=0.5,
            uncertainty_score=0.3,
            context_load=_context_load(context),
            reasoning="fallback heuristic",
            fallback_agent=None,
        )


def build_questions() -> list[BenchmarkQuestion]:
    return [
        BenchmarkQuestion(
            qid="q01",
            category="reasoning",
            query="Prove that the sum of the first n odd numbers equals n squared.",
            expected_keywords=["odd", "n squared", "induction"],
        ),
        BenchmarkQuestion(
            qid="q02",
            category="reasoning",
            query="Explain why BFS gives shortest paths in an unweighted graph.",
            expected_keywords=["bfs", "shortest", "unweighted"],
        ),
        BenchmarkQuestion(
            qid="q03",
            category="retrieval",
            query="List three key contributions of Alan Turing.",
            expected_keywords=["turing", "machine", "computing"],
        ),
        BenchmarkQuestion(
            qid="q04",
            category="retrieval",
            query="State two ideas from Shannon 1948 information theory paper.",
            expected_keywords=["entropy", "bit", "channel"],
        ),
        BenchmarkQuestion(
            qid="q05",
            category="calculation",
            query="Compute the integral of x^2 from 0 to 3.",
            expected_keywords=["9"],
        ),
        BenchmarkQuestion(
            qid="q06",
            category="calculation",
            query="Solve the system: 2x+3y=13 and x-y=1.",
            expected_keywords=["x=4", "y=3"],
        ),
        BenchmarkQuestion(
            qid="q07",
            category="synthesis",
            query="Summarize differences between supervised and unsupervised learning.",
            expected_keywords=["labels", "unlabeled", "supervised", "unsupervised"],
        ),
        BenchmarkQuestion(
            qid="q08",
            category="synthesis",
            query="Compare cloud computing and edge computing.",
            expected_keywords=["cloud", "edge", "latency", "bandwidth"],
        ),
        BenchmarkQuestion(
            qid="q09",
            category="ambiguity",
            query="Improve the model.",
            expected_keywords=["clarify", "details"],
        ),
        BenchmarkQuestion(
            qid="q10",
            category="ambiguity",
            query="Explain the results.",
            expected_keywords=["clarify", "context"],
        ),
        BenchmarkQuestion(
            qid="q11",
            category="reasoning",
            query="Show that sqrt(2) is irrational.",
            expected_keywords=["sqrt", "irrational", "contradiction"],
        ),
        BenchmarkQuestion(
            qid="q12",
            category="retrieval",
            query="Give main goals of the Bologna Process in higher education.",
            expected_keywords=["bologna", "mobility", "comparability"],
        ),
        BenchmarkQuestion(
            qid="q13",
            category="calculation",
            query="Probability of two heads in three fair coin flips.",
            expected_keywords=["3/8"],
        ),
        BenchmarkQuestion(
            qid="q14",
            category="synthesis",
            query="Write a short abstract about adaptive multi agent orchestration.",
            expected_keywords=["adaptive", "multi agent", "orchestration"],
        ),
        BenchmarkQuestion(
            qid="q15",
            category="ambiguity",
            query="Make it better.",
            expected_keywords=["clarify", "goal"],
        ),
    ]


def build_agents() -> list[AgentProfile]:
    return [
        AgentProfile(
            agent_id="planner",
            role="planning",
            capabilities=["reasoning", "ambiguity", "planning"],
            base_confidence=0.7,
        ),
        AgentProfile(
            agent_id="retriever",
            role="retrieval",
            capabilities=["retrieval", "document", "search"],
            base_confidence=0.75,
        ),
        AgentProfile(
            agent_id="tool",
            role="tool",
            capabilities=["calculation", "math", "tools"],
            base_confidence=0.8,
            tool_ready=True,
        ),
        AgentProfile(
            agent_id="synth",
            role="synthesis",
            capabilities=["synthesis", "summary", "writing"],
            base_confidence=0.72,
        ),
        AgentProfile(
            agent_id="clarifier",
            role="clarification",
            capabilities=["ambiguity", "clarify", "question"],
            base_confidence=0.65,
        ),
    ]


def simulate_response(
    agent: AgentProfile,
    question: BenchmarkQuestion,
    rng: random.Random,
) -> tuple[str, bool]:
    category = question.category
    match = category in agent.capabilities or category == agent.role
    base = 0.85 if match else 0.35
    if category == "ambiguity" and agent.agent_id in {"clarifier", "planner"}:
        base = 0.9
    ok = rng.random() < base
    if not ok:
        return "response not available", False
    if category == "ambiguity":
        return "clarify missing details and define the goal", True
    keywords = ", ".join(question.expected_keywords)
    return f"{agent.role} response: {keywords}", True


def estimate_quality(response: str, question: BenchmarkQuestion) -> float:
    if not question.expected_keywords:
        return 0.0
    text = response.lower()
    hits = sum(1 for kw in question.expected_keywords if kw in text)
    return hits / max(1, len(question.expected_keywords))


def simulate_latency(agent_id: str, rng: random.Random) -> float:
    base = {
        "planner": 120.0,
        "retriever": 180.0,
        "tool": 80.0,
        "synth": 150.0,
        "clarifier": 70.0,
    }.get(agent_id, 110.0)
    return base * (0.8 + rng.random() * 0.4)


def _context_load(context: BenchmarkContext) -> float:
    return min(1.0, (context.memory_tokens + len(context.history) * 50) / 10000.0)


def benchmark(
    configs: list[Any],
    questions: list[BenchmarkQuestion],
    agents: dict[str, AgentProfile],
    seed: int,
    runs: int,
) -> list[RunResult]:
    rng = random.Random(seed)
    results: list[RunResult] = []
    token_estimator = TokenEstimator()

    for run_index in range(runs):
        run_id = f"run-{run_index + 1}"
        for config in configs:
            context = BenchmarkContext(history=[], agent_context_tokens={}, memory_tokens=0)
            for question in questions:
                trace_id = str(uuid.uuid4())
                selection = _await(config.select_agents(question, context))
                selected_agents = selection.selected_agents
                routing_cost = 25.0 if config.name == "dynamic_adaptive" else 10.0
                latency_ms = routing_cost
                tokens = token_estimator.estimate(question.query)
                responses = []
                ok_flags = []
                for agent_id in selected_agents:
                    agent = agents[agent_id]
                    response, ok = simulate_response(agent, question, rng)
                    latency_ms += simulate_latency(agent_id, rng)
                    response_tokens = token_estimator.estimate(response)
                    tokens += response_tokens
                    context.agent_context_tokens[agent_id] = context.agent_context_tokens.get(agent_id, 0) + tokens
                    responses.append(response)
                    ok_flags.append(ok)
                combined = " ".join(responses)
                quality = estimate_quality(combined, question)
                failed = not any(ok_flags)
                context.memory_tokens += tokens
                context.history.append({"content": combined, "trace_id": trace_id})
                context_cost = tokens * (1.0 + selection.context_load)
                results.append(
                    RunResult(
                        config_name=config.name,
                        run_id=run_id,
                        question_id=question.qid,
                        category=question.category,
                        latency_ms=round(latency_ms, 3),
                        agent_calls=len(selected_agents),
                        failed=failed,
                        tokens=tokens,
                        quality_score=round(quality, 3),
                        context_cost=round(context_cost, 3),
                        selected_agents=list(selected_agents),
                        fallback_used=bool(
                            selection.fallback_agent
                            and selection.fallback_agent in selected_agents
                            and selection.fallback_agent != selected_agents[0]
                        ),
                    )
                )
                _log_event(
                    "benchmark_record",
                    trace_id=trace_id,
                    config=config.name,
                    question_id=question.qid,
                    latency_ms=latency_ms,
                    agent_calls=len(selected_agents),
                    failed=failed,
                    tokens=tokens,
                    quality_score=quality,
                    context_cost=context_cost,
                )
    return results


def summarize(results: list[RunResult]) -> dict[str, Any]:
    by_config: dict[str, list[RunResult]] = {}
    for item in results:
        by_config.setdefault(item.config_name, []).append(item)
    summary: dict[str, Any] = {}
    for name, items in by_config.items():
        summary[name] = {
            "count": len(items),
            "latency_ms": _stats([i.latency_ms for i in items]),
            "agent_calls": _stats([i.agent_calls for i in items]),
            "failure_rate": _failure_rate(items),
            "tokens": _stats([i.tokens for i in items]),
            "quality_score": _stats([i.quality_score for i in items]),
            "context_cost": _stats([i.context_cost for i in items]),
        }
    return summary


def visualization_payload(results: list[RunResult]) -> dict[str, Any]:
    by_config: dict[str, list[RunResult]] = {}
    for item in results:
        by_config.setdefault(item.config_name, []).append(item)
    hist: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for name, items in by_config.items():
        latencies = [i.latency_ms for i in items]
        qualities = [i.quality_score for i in items]
        tokens = [i.tokens for i in items]
        hist[name] = {
            "latency_ms": latencies,
            "quality_score": qualities,
            "tokens": tokens,
        }
        curves[name] = {
            "latency_sorted": sorted(latencies),
            "quality_sorted": sorted(qualities),
            "tokens_sorted": sorted(tokens),
        }
    radar = _radar_scores(by_config)
    return {"histograms": hist, "curves": curves, "radar": radar}


def _radar_scores(by_config: dict[str, list[RunResult]]) -> dict[str, Any]:
    metrics = {
        "latency_ms": [],
        "tokens": [],
        "context_cost": [],
        "failure_rate": [],
        "quality_score": [],
        "agent_calls": [],
    }
    for items in by_config.values():
        metrics["latency_ms"].extend([i.latency_ms for i in items])
        metrics["tokens"].extend([i.tokens for i in items])
        metrics["context_cost"].extend([i.context_cost for i in items])
        metrics["failure_rate"].append(_failure_rate(items))
        metrics["quality_score"].extend([i.quality_score for i in items])
        metrics["agent_calls"].extend([i.agent_calls for i in items])
    min_max = {k: (min(v), max(v)) if v else (0.0, 1.0) for k, v in metrics.items()}
    radar: dict[str, Any] = {}
    for name, items in by_config.items():
        latency = _normalize(_mean([i.latency_ms for i in items]), *min_max["latency_ms"], invert=True)
        tokens = _normalize(_mean([i.tokens for i in items]), *min_max["tokens"], invert=True)
        cost = _normalize(_mean([i.context_cost for i in items]), *min_max["context_cost"], invert=True)
        fail = _normalize(_failure_rate(items), *min_max["failure_rate"], invert=True)
        quality = _normalize(_mean([i.quality_score for i in items]), *min_max["quality_score"])
        calls = _normalize(_mean([i.agent_calls for i in items]), *min_max["agent_calls"], invert=True)
        radar[name] = {
            "latency": round(latency, 3),
            "tokens": round(tokens, 3),
            "context_cost": round(cost, 3),
            "reliability": round(fail, 3),
            "quality": round(quality, 3),
            "efficiency": round(calls, 3),
        }
    return radar


def write_csv(path: Path, results: list[RunResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "config",
                "run_id",
                "question_id",
                "category",
                "latency_ms",
                "agent_calls",
                "failed",
                "tokens",
                "quality_score",
                "context_cost",
                "selected_agents",
                "fallback_used",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.config_name,
                    item.run_id,
                    item.question_id,
                    item.category,
                    item.latency_ms,
                    item.agent_calls,
                    int(item.failed),
                    item.tokens,
                    item.quality_score,
                    item.context_cost,
                    "|".join(item.selected_agents),
                    int(item.fallback_used),
                ]
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _stats(values: Iterable[float]) -> dict[str, float]:
    values_list = list(values)
    if not values_list:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    values_list.sort()
    return {
        "mean": round(statistics.mean(values_list), 3),
        "p50": round(_percentile(values_list, 50), 3),
        "p90": round(_percentile(values_list, 90), 3),
        "min": round(values_list[0], 3),
        "max": round(values_list[-1], 3),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    k = (len(values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def _failure_rate(items: list[RunResult]) -> float:
    if not items:
        return 0.0
    return round(sum(1 for i in items if i.failed) / len(items), 3)


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return statistics.mean(values_list)


def _normalize(value: float, min_val: float, max_val: float, invert: bool = False) -> float:
    if max_val <= min_val:
        return 0.0
    score = (value - min_val) / (max_val - min_val)
    return 1.0 - score if invert else score


def _await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return _run(value)
    return value


def _run(awaitable: Any) -> Any:
    return _loop_run(awaitable)


def _loop_run(awaitable: Any) -> Any:
    try:
        import asyncio

        return asyncio.run(awaitable)
    except RuntimeError:
        return _run_in_loop(awaitable)


def _run_in_loop(awaitable: Any) -> Any:
    import asyncio

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(awaitable)


def _log_event(event: str, **fields: Any) -> None:
    LOG.info(json.dumps({"event": event, **fields}, ensure_ascii=True))


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    handler = logging.FileHandler(out_dir / "benchmark.log", encoding="utf-8")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    LOG.addHandler(handler)


# ---------------------------------------------------------------------------
# Pipeline benchmarks (real orchestrator, mocked Ollama — runs offline)
# ---------------------------------------------------------------------------
def _mock_ollama_response(text: str) -> dict:
    return {"message": {"content": text}}


def bench_full_pipeline_latency(iterations: int = 3) -> dict[str, Any]:
    """Mesure le temps réel de orchestrator.answer() avec LLM mocké."""
    import os

    os.environ.setdefault("A2A_PROFILE", "full")
    from unittest.mock import MagicMock, patch

    from src.orchestrator.academic_orchestrator import (
        AcademicOrchestrator,
        OrchestratorConfig,
    )

    # Réponse par défaut = vérification approuvée valide. Évite tout faux retry
    # si l'ordre des appels varie (ex: RAG sans retriever ne consomme pas d'appel).
    approved_json = json.dumps(
        {"factual": 8, "coherence": 9, "completeness": 8, "issues": [], "approved": True}
    )

    latencies: list[float] = []
    approvals = 0
    with patch("src.agents.specialized_agents.ollama") as mock_ol:
        for _ in range(iterations):
            mock_ol.chat = MagicMock(return_value=_mock_ollama_response(approved_json))
            orch = AcademicOrchestrator(config=OrchestratorConfig())
            result = _run(orch.answer("Qu'est-ce que la photosynthèse ?"))
            latencies.append(result.duration_ms)
            approvals += 1 if result.approved else 0
    payload = {
        "name": "full_pipeline_latency",
        "iterations": iterations,
        "avg_duration_ms": _mean(latencies),
        "min_duration_ms": min(latencies) if latencies else 0.0,
        "max_duration_ms": max(latencies) if latencies else 0.0,
        "approval_rate": approvals / max(1, iterations),
    }
    _log_event("bench_full_pipeline_latency", **payload)
    return payload


def bench_exam_generation() -> dict[str, Any]:
    """Teste la génération d'examen via l'orchestrateur (LLM mocké)."""
    import os

    os.environ.setdefault("A2A_PROFILE", "full")
    from unittest.mock import MagicMock, patch

    from src.orchestrator.academic_orchestrator import (
        AcademicOrchestrator,
        OrchestratorConfig,
    )

    exam_text = "EXAMEN\nQuestion 1 (5 pts)...\nCritères d'évaluation."
    approved_json = json.dumps(
        {"factual": 8, "coherence": 8, "completeness": 8, "issues": [], "approved": True}
    )

    def _smart_chat(*args, **kwargs):
        # Le système prompt du vérifier contient "verification"; on l'approuve.
        system = ""
        for m in kwargs.get("messages", []):
            if m.get("role") == "system":
                system = m.get("content", "")
                break
        if "verification" in system.lower():
            return _mock_ollama_response(approved_json)
        if "planification" in system.lower():
            return _mock_ollama_response(json.dumps({"tasks": ["Structurer", "Rédiger"]}))
        return _mock_ollama_response(exam_text)

    with patch("src.agents.specialized_agents.ollama") as mock_ol:
        mock_ol.chat = MagicMock(side_effect=_smart_chat)
        orch = AcademicOrchestrator(config=OrchestratorConfig())
        result = _run(orch.answer("Génère un examen de mathématiques niveau Licence 1."))
    payload = {
        "name": "exam_generation",
        "exam_len": len(result.final_answer),
        "approved": result.approved,
        "duration_ms": result.duration_ms,
        "non_empty": bool(result.final_answer.strip()),
    }
    _log_event("bench_exam_generation", **payload)
    return payload


def bench_routing_distribution(num_queries: int = 20) -> dict[str, Any]:
    """Vérifie que chaque agent est sélectionné au moins une fois sur 20 requêtes."""
    from src.orchestrator.dynamic_router import AgentState, DynamicRouter

    router = DynamicRouter()
    agents = [
        AgentState(agent_id="planner", role="planning",
                   capabilities=["plan", "decompose", "organize", "structure", "task"], base_confidence=0.85),
        AgentState(agent_id="rag", role="retrieval",
                   capabilities=["retrieve", "search", "find", "document", "context", "source"], base_confidence=0.80),
        AgentState(agent_id="synthesizer", role="synthesis",
                   capabilities=["write", "synthesize", "explain", "answer", "generate", "summarize"], base_confidence=0.90),
        AgentState(agent_id="verifier", role="verification",
                   capabilities=["verify", "check", "validate", "review", "quality", "correct"], base_confidence=0.75),
    ]
    prefixes = [
        "plan decompose organize structure task:",
        "retrieve search find document context source:",
        "write synthesize explain answer generate summarize:",
        "verify check validate review quality correct:",
    ]
    topics = ["la photosynthèse", "les bases de données", "la mécanique quantique",
              "le droit constitutionnel", "l'algorithmique"]
    counts: dict[str, int] = {a.agent_id: 0 for a in agents}
    n = 0
    for i in range(num_queries):
        prefix = prefixes[i % len(prefixes)]
        topic = topics[i % len(topics)]
        decision = _run(router.route(query=f"{prefix} {topic}", memory_state=None, agents_state=agents))
        counts[decision.selected_agent] = counts.get(decision.selected_agent, 0) + 1
        n += 1
    all_selected = all(v >= 1 for v in counts.values())
    payload = {
        "name": "routing_distribution",
        "num_queries": n,
        "counts": counts,
        "all_agents_selected": all_selected,
    }
    _log_event("bench_routing_distribution", **payload)
    return payload


def run_pipeline_benchmarks() -> dict[str, Any]:
    return {
        "full_pipeline_latency": bench_full_pipeline_latency(),
        "exam_generation": bench_exam_generation(),
        "routing_distribution": bench_routing_distribution(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--multi-agent", action="store_true")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run real-orchestrator pipeline benchmarks (mocked LLM)")
    args = parser.parse_args()

    if args.pipeline:
        out_dir = Path(args.out_dir)
        setup_logging(out_dir)
        payload = run_pipeline_benchmarks()
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    out_dir = Path(args.out_dir)
    setup_logging(out_dir)

    questions = build_questions()
    agents_list = build_agents()
    agents = {agent.agent_id: agent for agent in agents_list}

    fixed = FixedHierarchicalOrchestrator(
        {
            "reasoning": "planner",
            "retrieval": "retriever",
            "calculation": "tool",
            "synthesis": "synth",
            "ambiguity": "clarifier",
        }
    )
    dynamic = AdaptiveDynamicOrchestrator(agents_list, multi_agent=args.multi_agent)

    results = benchmark(
        configs=[fixed, dynamic],
        questions=questions,
        agents=agents,
        seed=args.seed,
        runs=args.runs,
    )

    summary = summarize(results)
    payload = {
        "seed": args.seed,
        "runs": args.runs,
        "results": [item.__dict__ for item in results],
        "summary": summary,
        "visualization": visualization_payload(results),
    }

    write_csv(out_dir / "benchmark_results.csv", results)
    write_json(out_dir / "benchmark_results.json", payload)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Academic Orchestrator — v3 (deadlock-free).

Fix principal : bypass complet du bus A2A pour les appels internes.
Chaque _step_xxx() appelle generate_response() directement sur l'agent,
éliminant le pattern send/receive/process_task/receive qui causait
un deadlock (le 2e receive_message attendait un message fantôme).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..agents.base_agent import AgentMessage, InMemoryMessageBus
from ..agents.specialized_agents import PlannerAgent, RAGAgent, SynthesizerAgent, VerifierAgent
from ..orchestrator.dynamic_router import AgentState, DynamicRouter, MemoryState, RouterConfig

LOGGER = logging.getLogger("orchestrator")


@dataclass
class OrchestratorConfig:
    llm_model: str = "llama3"
    max_verify_retries: int = 2
    verify_approval_threshold: float = 0.7
    router_config: RouterConfig = field(default_factory=RouterConfig)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class PipelineResult:
    query: str
    plan: list[str]
    retrieved_context: str
    rag_chunks: list[dict[str, Any]]
    draft: str
    verification: dict[str, Any]
    final_answer: str
    approved: bool
    retry_count: int
    trace_id: str
    duration_ms: float
    routing_decisions: list[dict[str, Any]]


def _make_msg(sender: str, receiver: str, task_type: str, content: str, trace_id: str) -> AgentMessage:
    """Crée un AgentMessage sans passer par le bus."""
    return AgentMessage(
        sender=sender,
        receiver=receiver,
        task_type=task_type,
        content=content,
        trace_id=trace_id,
    )


class AcademicOrchestrator:
    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        retriever: Any | None = None,
    ) -> None:
        self._config = config or OrchestratorConfig()
        self._bus = InMemoryMessageBus()
        self._router = DynamicRouter(
            llm_model=self._config.llm_model,
            config=self._config.router_config,
        )
        self._retriever = retriever
        self._planner = PlannerAgent(self._bus, llm_model=self._config.llm_model)
        self._rag = RAGAgent(self._bus, retriever=retriever, llm_model=self._config.llm_model)
        self._synthesizer = SynthesizerAgent(self._bus, llm_model=self._config.llm_model)
        self._verifier = VerifierAgent(self._bus, llm_model=self._config.llm_model)
        self._agent_states = self._build_agent_states()
        LOGGER.info(
            "AcademicOrchestrator ready — model=%s retriever=%s",
            self._config.llm_model,
            type(retriever).__name__ if retriever else "None",
        )

    @classmethod
    def build(cls, sqlite_path: str, chroma_path: str) -> "AcademicOrchestrator":
        from ..memory.hybrid_memory import HybridMemory
        from ..rag.pipeline import RAGPipeline
        memory = HybridMemory(sqlite_path=sqlite_path, chroma_path=chroma_path)
        return cls(retriever=RAGPipeline(memory))

    def _build_agent_states(self) -> list[AgentState]:
        return [
            AgentState(
                agent_id="planner", role="planning",
                capabilities=["plan", "decompose", "organize", "structure", "task"],
                base_confidence=0.85,
            ),
            AgentState(
                agent_id="rag", role="retrieval",
                capabilities=["retrieve", "search", "find", "document", "context", "source"],
                base_confidence=0.80,
            ),
            AgentState(
                agent_id="synthesizer", role="synthesis",
                capabilities=["write", "synthesize", "explain", "answer", "generate", "summarize"],
                base_confidence=0.90,
            ),
            AgentState(
                agent_id="verifier", role="verification",
                capabilities=["verify", "check", "validate", "review", "quality", "correct"],
                base_confidence=0.75,
            ),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def answer(self, query: str) -> PipelineResult:
        trace_id = str(uuid.uuid4())
        started = time.time()
        routing_decisions: list[dict[str, Any]] = []

        LOGGER.info(json.dumps({
            "event": "pipeline_start",
            "trace_id": trace_id,
            "query_len": len(query),
            "query_preview": query[:80],
        }))

        # Step 1 — Plan
        t = time.time()
        plan = await self._step_plan(query, trace_id, routing_decisions)
        LOGGER.info(json.dumps({
            "event": "step_plan_done", "trace_id": trace_id,
            "tasks": len(plan), "ms": round((time.time() - t) * 1000),
        }))

        # Step 2 — Retrieve
        t = time.time()
        context, rag_chunks = await self._step_retrieve(query, trace_id, routing_decisions)
        LOGGER.info(json.dumps({
            "event": "step_retrieve_done", "trace_id": trace_id,
            "chunks": len(rag_chunks),
            "top_sim": round(rag_chunks[0]["similarity"], 3) if rag_chunks else 0,
            "ms": round((time.time() - t) * 1000),
        }))

        # Step 3 — Synthesize (with real MemoryState)
        t = time.time()
        memory_state = MemoryState(
            token_count=max(0, len(context.split()) * 4 // 3),
            history_size=len(rag_chunks),
        )
        draft = await self._step_synthesize(
            query, plan, context, trace_id, routing_decisions, memory_state
        )
        LOGGER.info(json.dumps({
            "event": "step_synthesize_done", "trace_id": trace_id,
            "draft_len": len(draft), "ms": round((time.time() - t) * 1000),
        }))

        # Step 4 — Verify (with retry)
        t = time.time()
        verification, final_answer, retry_count = await self._step_verify_with_retry(
            query, draft, plan, context, trace_id, routing_decisions, memory_state
        )
        LOGGER.info(json.dumps({
            "event": "step_verify_done", "trace_id": trace_id,
            "approved": verification.get("approved"),
            "retries": retry_count,
            "ms": round((time.time() - t) * 1000),
        }))

        duration = round((time.time() - started) * 1000, 2)
        approved = bool(verification.get("approved", False))

        LOGGER.info(json.dumps({
            "event": "pipeline_complete", "trace_id": trace_id,
            "approved": approved, "retries": retry_count,
            "duration_ms": duration, "rag_chunks": len(rag_chunks),
        }))

        return PipelineResult(
            query=query,
            plan=plan,
            retrieved_context=context,
            rag_chunks=rag_chunks,
            draft=draft,
            verification=verification,
            final_answer=final_answer,
            approved=approved,
            retry_count=retry_count,
            trace_id=trace_id,
            duration_ms=duration,
            routing_decisions=routing_decisions,
        )

    # ------------------------------------------------------------------
    # Pipeline steps — direct generate_response() calls (no bus deadlock)
    # ------------------------------------------------------------------
    async def _step_plan(
        self,
        query: str,
        trace_id: str,
        routing_decisions: list[dict[str, Any]],
    ) -> list[str]:
        router_query = f"plan decompose organize structure task: {query}"
        decision = await self._router.route(
            query=router_query,
            memory_state=None,
            agents_state=self._agent_states,
        )
        routing_decisions.append({
            "step": "plan",
            "decision": decision.model_dump(),
            "actual_agent": "planner",
        })

        # ✅ Direct call — no bus, no deadlock
        msg = _make_msg("orchestrator", "planner", "plan", query, trace_id)
        try:
            response_content = await self._planner.generate_response(msg)
            self._planner.log_interaction(msg, status="processed")
        except Exception as exc:
            LOGGER.error("PlannerAgent failed: %s", exc)
            response_content = json.dumps({
                "tasks": [
                    "Retrieve relevant documents",
                    "Verify key facts",
                    "Synthesize a final answer",
                ]
            })

        try:
            parsed = json.loads(response_content)
            tasks = parsed.get("tasks", [])
            if isinstance(tasks, list) and tasks:
                return tasks
        except Exception:
            pass
        return ["Retrieve relevant documents", "Synthesize a final answer"]

    async def _step_retrieve(
        self,
        query: str,
        trace_id: str,
        routing_decisions: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Returns (context_string, raw_chunks_with_similarity)."""
        router_query = f"retrieve search find document context source: {query}"
        decision = await self._router.route(
            query=router_query,
            memory_state=None,
            agents_state=self._agent_states,
        )
        routing_decisions.append({
            "step": "retrieve",
            "decision": decision.model_dump(),
            "actual_agent": "rag",
        })

        # ── Collect real chunks with similarity scores
        raw_chunks: list[dict[str, Any]] = []
        if self._retriever is not None:
            try:
                raw_chunks = self._retriever.retrieve(
                    query=query, top_k=5, min_similarity=0.0
                )
                LOGGER.info(json.dumps({
                    "event": "rag_retrieved", "trace_id": trace_id,
                    "chunks": len(raw_chunks),
                    "similarities": [round(c["similarity"], 3) for c in raw_chunks],
                    "sources": list({c.get("source", "?") for c in raw_chunks}),
                }))
            except Exception as exc:
                LOGGER.error("Direct retrieval failed: %s", exc)

        # ── RAGAgent for LLM-assisted context summarization
        msg = _make_msg("orchestrator", "rag", "retrieve", query, trace_id)
        try:
            response_content = await self._rag.generate_response(msg)
            self._rag.log_interaction(msg, status="processed")
            parsed = json.loads(response_content)
            context = parsed.get("context", "")
        except Exception as exc:
            LOGGER.error("RAGAgent failed: %s", exc)
            context = ""

        # ── Fallback: build context from raw chunks if RAGAgent returned empty
        if not context or context.startswith("(no"):
            if raw_chunks:
                lines = [
                    f"[{i+1}] (score={c['similarity']:.2f}, source={c.get('source', '?')})\n{c['content']}"
                    for i, c in enumerate(raw_chunks)
                ]
                context = "\n\n---\n\n".join(lines)
            else:
                context = "(no documents found)"

        return context, raw_chunks

    async def _step_synthesize(
        self,
        query: str,
        plan: list[str],
        context: str,
        trace_id: str,
        routing_decisions: list[dict[str, Any]],
        memory_state: MemoryState,
    ) -> str:
        router_query = f"write synthesize explain answer generate summarize: {query}"
        decision = await self._router.route(
            query=router_query,
            memory_state=memory_state,
            agents_state=self._agent_states,
        )
        routing_decisions.append({
            "step": "synthesize",
            "decision": decision.model_dump(),
            "actual_agent": "synthesizer",
        })

        payload = json.dumps({"question": query, "context": context, "plan": plan})
        msg = _make_msg("orchestrator", "synthesizer", "synthesize", payload, trace_id)
        try:
            draft = await self._synthesizer.generate_response(msg)
            self._synthesizer.log_interaction(msg, status="processed")
        except Exception as exc:
            LOGGER.error("SynthesizerAgent failed: %s", exc)
            draft = f"(synthesis error: {exc})"
        return draft

    async def _step_verify_with_retry(
        self,
        query: str,
        draft: str,
        plan: list[str],
        context: str,
        trace_id: str,
        routing_decisions: list[dict[str, Any]],
        memory_state: MemoryState,
    ) -> tuple[dict[str, Any], str, int]:
        retry_count = 0
        current_draft = draft
        verification: dict[str, Any] = {}

        while retry_count <= self._config.max_verify_retries:
            router_query = f"verify check validate review quality correct: {query}"
            decision = await self._router.route(
                query=router_query,
                memory_state=memory_state,
                agents_state=self._agent_states,
            )
            routing_decisions.append({
                "step": f"verify_{retry_count}",
                "decision": decision.model_dump(),
                "actual_agent": "verifier",
            })

            payload = json.dumps({"question": query, "draft": current_draft})
            msg = _make_msg("orchestrator", "verifier", "verify", payload, trace_id)
            try:
                response_content = await self._verifier.generate_response(msg)
                self._verifier.log_interaction(msg, status="processed")
                verification = json.loads(response_content)
            except Exception as exc:
                LOGGER.error("VerifierAgent failed (retry=%d): %s", retry_count, exc)
                verification = {
                    "factual": 5, "coherence": 5, "completeness": 5,
                    "issues": [str(exc)], "approved": False,
                }

            if verification.get("approved", False):
                return verification, current_draft, retry_count

            retry_count += 1
            if retry_count <= self._config.max_verify_retries:
                issues = verification.get("issues", [])
                LOGGER.warning(json.dumps({
                    "event": "verify_retry", "trace_id": trace_id,
                    "retry": retry_count,
                    "issues": issues,
                }))
                # Re-synthèse réelle avec le feedback du vérifier (pas de placeholder).
                feedback = "; ".join(str(i) for i in issues) if issues else "Améliorer la qualité"
                context_with_feedback = f"{context}\n\n[Feedback vérification]: {feedback}"
                current_draft = await self._step_synthesize(
                    query, plan, context_with_feedback, trace_id,
                    routing_decisions, memory_state,
                )

        return verification, current_draft, retry_count

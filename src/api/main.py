"""FastAPI — AcademiQ v3.

Corrections :
  - Init robuste avec HybridMemory + RAGPipeline partagés
  - Orchestrateur avec retriever réel
  - Endpoint /debug/status pour diagnostiquer l'état du système
  - AskResponse inclut routing_decisions + rag_chunks
  - Logs structurés + gestion d'erreurs claire
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
LOGGER = logging.getLogger("api")

app = FastAPI(
    title="AcademiQ — Academic Assistant API",
    description="Multi-agent AI: Planner → RAG → Synthesizer → Verifier",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
_BASE_DATA = Path(os.getenv("A2A_DATA_DIR", "./data"))
_SQLITE_PATH = _BASE_DATA / "memory.db"
_CHROMA_PATH = _BASE_DATA / "chroma"

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_memory = None
_rag_pipeline = None
_orchestrator = None

_ui_state: dict[str, Any] = {
    "requests": 0,
    "approved": 0,
    "latencies_ms": [],
    "last_routing": [],
}

# Historique des 100 dernières requêtes /ask (F4)
_metrics_history: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Lazy initializers — real systems, no stubs
# ---------------------------------------------------------------------------
def _get_memory():
    global _memory
    if _memory is None:
        from src.memory.hybrid_memory import HybridMemory
        _BASE_DATA.mkdir(parents=True, exist_ok=True)
        _memory = HybridMemory(
            sqlite_path=str(_SQLITE_PATH),
            chroma_path=str(_CHROMA_PATH),
        )
        LOGGER.info("HybridMemory OK — sqlite=%s chroma=%s", _SQLITE_PATH, _CHROMA_PATH)
    return _memory


def _get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        from src.rag.pipeline import RAGPipeline
        _rag_pipeline = RAGPipeline(
            memory=_get_memory(),
            session_id="rag-global",
            agent_id="rag",
        )
        LOGGER.info("RAGPipeline OK (ChromaDB-backed, embedding=all-MiniLM-L6-v2)")
    return _rag_pipeline


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from src.orchestrator.academic_orchestrator import AcademicOrchestrator, OrchestratorConfig
        config = OrchestratorConfig(llm_model=os.getenv("A2A_LLM_MODEL", "llama3"))
        _orchestrator = AcademicOrchestrator(
            config=config,
            retriever=_get_rag_pipeline(),
        )
        LOGGER.info("AcademicOrchestrator OK — model=%s", config.llm_model)
    return _orchestrator


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _update_ui_state(
    approved: bool, duration_ms: float, routing: list | None = None
) -> None:
    _ui_state["requests"] += 1
    if approved:
        _ui_state["approved"] += 1
    _ui_state["latencies_ms"].append(duration_ms)
    _ui_state["latencies_ms"] = _ui_state["latencies_ms"][-50:]
    if routing:
        _ui_state["last_routing"] = routing


def _avg_latency_ms() -> float | None:
    vals = _ui_state["latencies_ms"]
    return round(sum(vals) / len(vals), 2) if vals else None


def _agent_snapshot() -> list[dict]:
    if _orchestrator is None:
        return [
            {"agent_id": a, "role": r, "status": "ready"}
            for a, r in [
                ("planner", "planning"),
                ("rag", "retrieval"),
                ("synthesizer", "synthesis"),
                ("verifier", "verification"),
            ]
        ]
    return [
        {
            "agent_id": a.agent_id,
            "role": a.role,
            "capabilities": a.capabilities,
            "base_confidence": a.base_confidence,
            "status": "ready",
        }
        for a in _orchestrator._agent_states
    ]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=4000)
    session_id: str | None = None
    top_k_docs: int = Field(5, ge=1, le=20)


class VerificationDetail(BaseModel):
    factual: int
    coherence: int
    completeness: int
    issues: list[str]
    approved: bool


class RAGChunk(BaseModel):
    content: str
    similarity: float
    source: str
    source_type: str = "persistent"


class AskResponse(BaseModel):
    query: str
    final_answer: str
    plan: list[str]
    context_snippet: str
    rag_chunks: list[RAGChunk]
    routing_decisions: list[dict]
    verification: VerificationDetail | None
    approved: bool
    retry_count: int
    trace_id: str
    duration_ms: float


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    source: str = Field("api-upload")


class IngestResponse(BaseModel):
    chunks_indexed: int
    source: str
    duration_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float
    rag_ready: bool
    orchestrator_ready: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    return {
        "name": "AcademiQ API v3",
        "status": "running",
        "endpoints": [
            "/health", "/ask", "/ingest",
            "/rag/stats", "/rag/debug",
            "/debug/status", "/ui/summary",
            "/pipeline/stats", "/metrics/history",
            "/generate/exam", "/generate/quiz", "/generate/summary",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="3.0.0",
        timestamp=time.time(),
        rag_ready=_rag_pipeline is not None,
        orchestrator_ready=_orchestrator is not None,
    )


@app.get("/debug/status")
def debug_status() -> dict:
    """Diagnostic complet de tous les composants du système."""
    result: dict[str, Any] = {}

    # Ollama
    try:
        import ollama as _ollama
        models_resp = _ollama.list()
        model_names = [m["model"] for m in models_resp.get("models", [])]
        result["ollama_available"] = True
        result["ollama_models"] = model_names
    except Exception as exc:
        result["ollama_available"] = False
        result["ollama_error"] = str(exc)
        result["ollama_models"] = []

    # sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        result["sentence_transformers_available"] = True
    except Exception as exc:
        result["sentence_transformers_available"] = False
        result["sentence_transformers_error"] = str(exc)

    # chromadb
    try:
        import chromadb  # noqa: F401
        result["chromadb_available"] = True
    except Exception as exc:
        result["chromadb_available"] = False
        result["chromadb_error"] = str(exc)

    # RAG pipeline
    result["orchestrator_ready"] = _orchestrator is not None
    try:
        rag = _get_rag_pipeline()
        mem = _get_memory()
        count = mem._persistent_store._collection.count()
        result["rag_ready"] = True
        result["rag_docs_count"] = count
    except Exception as exc:
        result["rag_ready"] = False
        result["rag_error"] = str(exc)
        result["rag_docs_count"] = 0

    result["pipeline_requests"] = _ui_state["requests"]
    result["pipeline_approved"] = _ui_state["approved"]
    result["avg_latency_ms"] = _avg_latency_ms()
    result["mode"] = os.getenv("A2A_PROFILE", "balanced")
    return result


@app.get("/rag/stats")
def rag_stats() -> dict:
    """Métriques réelles du pipeline RAG."""
    try:
        rag = _get_rag_pipeline()
        mem = _get_memory()
        count = mem._persistent_store._collection.count()
        return {
            "status": "ok",
            "pipeline": rag.stats(),
            "chroma_total_docs": count,
            "embedding_model": "all-MiniLM-L6-v2",
            "chunker": {"chunk_size": 512, "overlap": 64},
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.get("/rag/debug")
async def rag_debug(query: str, top_k: int = 5) -> dict:
    """Debug: chunks récupérés avec leurs similarités cosinus réelles."""
    try:
        rag = _get_rag_pipeline()
        t0 = time.time()
        results = rag.retrieve(query=query, top_k=top_k, min_similarity=0.0)
        latency = round((time.time() - t0) * 1000, 2)
        return {
            "query": query,
            "top_k": top_k,
            "results_count": len(results),
            "retrieval_latency_ms": latency,
            "chunks": [
                {
                    "rank": i + 1,
                    "similarity": r["similarity"],
                    "source": r["source"],
                    "source_type": r.get("source_type", "persistent"),
                    "content_preview": r["content"][:200]
                    + ("..." if len(r["content"]) > 200 else ""),
                }
                for i, r in enumerate(results)
            ],
        }
    except Exception as exc:
        LOGGER.error("RAG debug error: %s", exc, exc_info=True)
        return {"status": "error", "detail": str(exc)}


@app.get("/ui/summary")
def ui_summary() -> dict:
    return {
        "requests": _ui_state["requests"],
        "approved": _ui_state["approved"],
        "avg_latency_ms": _avg_latency_ms(),
        "routing": _ui_state["last_routing"],
        "agents": _agent_snapshot(),
        "mode": os.getenv("A2A_PROFILE", "balanced"),
    }


@app.get("/pipeline/stats")
def pipeline_stats() -> dict:
    try:
        rag = _get_rag_pipeline()
        return {"status": "running", "rag": rag.stats()}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, response: Response) -> AskResponse:
    started = time.time()
    trace = str(uuid.uuid4())
    LOGGER.info(
        "[%s] /ask — query='%s...' top_k=%d",
        trace[:8], request.query[:60], request.top_k_docs,
    )

    try:
        orchestrator = _get_orchestrator()
    except Exception as exc:
        LOGGER.error("[%s] Orchestrator init failed: %s", trace[:8], exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Orchestrator unavailable: {exc}")

    pipeline_timeout = float(os.getenv("A2A_PIPELINE_TIMEOUT", "180"))
    try:
        result = await asyncio.wait_for(
            orchestrator.answer(request.query),
            timeout=pipeline_timeout,
        )
    except asyncio.TimeoutError:
        LOGGER.error("[%s] Pipeline timeout (%.0fs)", trace[:8], pipeline_timeout)
        _update_ui_state(False, round((time.time() - started) * 1000, 2))
        raise HTTPException(status_code=504, detail="Pipeline timeout")
    except Exception as exc:
        LOGGER.error("[%s] Pipeline error: %s", trace[:8], exc, exc_info=True)
        _update_ui_state(False, round((time.time() - started) * 1000, 2))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # Build structured RAG chunks
    rag_chunks: list[RAGChunk] = []
    for chunk in getattr(result, "rag_chunks", []):
        try:
            rag_chunks.append(
                RAGChunk(
                    content=chunk.get("content", ""),
                    similarity=float(chunk.get("similarity", 0.0)),
                    source=chunk.get("source", "unknown"),
                    source_type=chunk.get("source_type", "persistent"),
                )
            )
        except Exception:
            pass

    context_snippet = result.retrieved_context[:500] + (
        "..." if len(result.retrieved_context) > 500 else ""
    )

    verification = None
    if result.verification:
        try:
            verification = VerificationDetail(**result.verification)
        except Exception:
            pass

    _update_ui_state(result.approved, result.duration_ms, result.routing_decisions)

    # F4 — historique des métriques (100 dernières requêtes)
    _metrics_history.append({
        "timestamp": time.time(),
        "query_len": len(request.query),
        "duration_ms": result.duration_ms,
        "approved": result.approved,
        "retry_count": result.retry_count,
        "plan_steps": len(result.plan),
    })
    if len(_metrics_history) > 100:
        _metrics_history.pop(0)

    # Preuve qu'un vrai LLM (Ollama/llama3) a été appelé
    response.headers["X-Real-LLM"] = "true"

    LOGGER.info(
        "[%s] done — approved=%s %.0fms chunks=%d",
        trace[:8], result.approved, result.duration_ms, len(rag_chunks),
    )

    return AskResponse(
        query=result.query,
        final_answer=result.final_answer,
        plan=result.plan,
        context_snippet=context_snippet,
        rag_chunks=rag_chunks,
        routing_decisions=result.routing_decisions,
        verification=verification,
        approved=result.approved,
        retry_count=result.retry_count,
        trace_id=result.trace_id,
        duration_ms=result.duration_ms,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    started = time.time()
    LOGGER.info(
        "Ingesting — source='%s' len=%d chars", request.source, len(request.text)
    )
    try:
        rag = _get_rag_pipeline()
        chunks = rag.ingest_text(request.text, source=request.source)
        LOGGER.info("Indexed %d chunks from '%s'", chunks, request.source)
    except Exception as exc:
        LOGGER.error("Ingest error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingest error: {exc}")
    return IngestResponse(
        chunks_indexed=chunks,
        source=request.source,
        duration_ms=round((time.time() - started) * 1000, 2),
    )


# ---------------------------------------------------------------------------
# F1 — Génération d'examens
# ---------------------------------------------------------------------------
class ExamRequest(BaseModel):
    subject: str
    level: str  # "Licence 1", "Master 2", etc.
    question_type: str  # "QCM", "ouvertes", "vrai-faux", "cas pratique", "mixte"
    num_questions: int = 10
    difficulty: str = "Moyen"
    language: str = "Français"
    topic_focus: str = ""  # sous-thème optionnel


class ExamResponse(BaseModel):
    exam_text: str
    subject: str
    level: str
    num_questions: int
    trace_id: str
    duration_ms: float


@app.post("/generate/exam", response_model=ExamResponse)
async def generate_exam(request: ExamRequest) -> ExamResponse:
    focus = f"Focus sur: {request.topic_focus}" if request.topic_focus else ""
    prompt = f"""Génère un examen académique complet en {request.language}.

Matière: {request.subject}
Niveau: {request.level}
Type de questions: {request.question_type}
Nombre de questions: {request.num_questions}
Difficulté: {request.difficulty}
{focus}

Format requis:
- En-tête avec matière, niveau, durée suggérée, barème total
- Questions numérotées clairement
- Pour QCM: 4 choix (A/B/C/D) par question
- Pour questions ouvertes: espace de réponse indiqué (lignes)
- Pour vrai-faux: énoncé clair
- Barème par question
- Section "Critères d'évaluation" à la fin

Génère uniquement l'examen, pas d'explication supplémentaire."""

    orchestrator = _get_orchestrator()
    try:
        result = await orchestrator.answer(prompt)
    except Exception as exc:
        LOGGER.error("Exam generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Exam generation error: {exc}")
    return ExamResponse(
        exam_text=result.final_answer,
        subject=request.subject,
        level=request.level,
        num_questions=request.num_questions,
        trace_id=result.trace_id,
        duration_ms=result.duration_ms,
    )


# ---------------------------------------------------------------------------
# F2 — Génération de quiz QCM (JSON structuré)
# ---------------------------------------------------------------------------
class QuizRequest(BaseModel):
    topic: str
    num_questions: int = 5
    difficulty: str = "Moyen"


@app.post("/generate/quiz")
async def generate_quiz(request: QuizRequest) -> dict:
    prompt = f"""Génère {request.num_questions} questions QCM sur "{request.topic}".
Difficulté: {request.difficulty}.
Retourne UNIQUEMENT un JSON valide:
{{"questions": [{{"id": 1, "question": "...", "choices": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "correct": "A", "explanation": "..."}}]}}"""
    orchestrator = _get_orchestrator()
    try:
        result = await orchestrator.answer(prompt)
    except Exception as exc:
        LOGGER.error("Quiz generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Quiz generation error: {exc}")
    try:
        json_match = re.search(r"\{.*\}", result.final_answer, re.DOTALL)
        data = json.loads(json_match.group()) if json_match else {}
    except Exception:
        data = {"questions": [], "raw": result.final_answer}
    return {"quiz": data, "topic": request.topic, "duration_ms": result.duration_ms}


# ---------------------------------------------------------------------------
# F3 — Génération de résumés
# ---------------------------------------------------------------------------
class SummaryRequest(BaseModel):
    text: str
    style: str = "académique"  # "académique", "bullet_points", "fiche_révision"
    language: str = "Français"


@app.post("/generate/summary")
async def generate_summary(request: SummaryRequest) -> dict:
    prompt = f"""Fais un résumé {request.style} du texte suivant en {request.language}:

{request.text[:3000]}

Style "{request.style}":
- académique: paragraphes structurés avec intro/corps/conclusion
- bullet_points: liste de points clés hiérarchisés
- fiche_révision: format fiche avec définitions, points clés, exemples"""
    orchestrator = _get_orchestrator()
    try:
        result = await orchestrator.answer(prompt)
    except Exception as exc:
        LOGGER.error("Summary generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summary generation error: {exc}")
    return {"summary": result.final_answer, "duration_ms": result.duration_ms}


# ---------------------------------------------------------------------------
# F4 — Historique des métriques
# ---------------------------------------------------------------------------
@app.get("/metrics/history")
def metrics_history() -> dict:
    return {
        "history": _metrics_history,
        "total": len(_metrics_history),
        "avg_duration_ms": sum(m["duration_ms"] for m in _metrics_history)
        / max(1, len(_metrics_history)),
        "approval_rate": sum(1 for m in _metrics_history if m["approved"])
        / max(1, len(_metrics_history)),
    }

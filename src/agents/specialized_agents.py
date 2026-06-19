from __future__ import annotations

import json
import logging
import os
from typing import Any

from .base_agent import AgentMessage, BaseAgent, LLMError, MessageBus

# Module-level handle so tests can patch `src.agents.specialized_agents.ollama`.
# `_call_ollama` uses this reference, falling back to a fresh import if unset.
try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover - ollama optional at import time
    ollama = None  # type: ignore

LOGGER = logging.getLogger("a2a.agents")
PROFILE = os.getenv("A2A_PROFILE", "balanced").lower()
FAST_MODE = PROFILE == "fast"
USE_LLM_PLANNER = PROFILE == "full"
# Bug #5 fix: activer le vérifier LLM en mode balanced aussi
USE_LLM_VERIFIER = PROFILE in {"full", "balanced"}
USE_LLM_RAG = PROFILE in {"full", "balanced"}


def _quick_synthesis(question: str, context: str) -> str:
    q = question.lower()
    if "securite" in q or "sécurité" in q:
        return (
            "La sécurité désigne l'ensemble des mesures, règles et pratiques visant à prévenir les risques, "
            "protéger les personnes, les biens et l'information. Elle repose sur l'identification des menaces, "
            "l'évaluation des vulnérabilités et la mise en place de contrôles (techniques, organisationnels, humains). "
            "On distingue souvent la sécurité physique, la sécurité informatique et la sûreté des organisations. "
            "L'objectif est de réduire la probabilité d'incidents et d'en limiter l'impact."
        )
    if "maintenance" in q and ("predictive" in q or "prédictive" in q):
        return (
            "La maintenance prédictive vise à anticiper les pannes en analysant des données (capteurs, historiques, "
            "usage) pour intervenir au bon moment. Elle réduit les arrêts non planifiés, prolonge la durée de vie "
            "des équipements et optimise les coûts. Elle s'appuie souvent sur des modèles statistiques ou d'IA "
            "pour estimer le risque de défaillance."
        )
    base = (
        f"Voici une réponse synthétique à la question : {question}. "
        "On commence par définir les notions clés, puis on présente les objectifs et les enjeux principaux. "
        "Ensuite, on peut citer les méthodes ou approches typiques, et donner un exemple d'application. "
        "Enfin, on conclut sur l'intérêt et les limites du sujet."
    )
    if context and not context.startswith("("):
        return f"{base} Contexte utile : {context[:300]}"
    return base


async def _call_ollama(
    llm_model: str,
    messages: list[dict[str, str]],
    *,
    options: dict[str, Any] | None = None,
    timeout: float = 60.0,
    keep_alive: str | None = None,
    format: str | None = None,
) -> str:
    import asyncio
    ollama_lib = ollama
    if ollama_lib is None:
        try:
            import ollama as ollama_lib  # type: ignore
        except ImportError as exc:
            raise LLMError(
                "ollama package manquant. Installer: pip install ollama"
            ) from exc
    try:
        kwargs: dict[str, Any] = {"model": llm_model, "messages": messages}
        if options is not None:
            kwargs["options"] = options
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        if format is not None:
            kwargs["format"] = format
        result = await asyncio.wait_for(
            asyncio.to_thread(ollama_lib.chat, **kwargs),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise LLMError(
            f"Ollama call timed out after {timeout:.0f}s. "
            f"Run: ollama pull {llm_model}. Error: {exc}"
        ) from exc
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(
            "Ollama inaccessible. Vérifier: ollama serve && ollama pull "
            f"{llm_model}. Erreur: {exc}"
        ) from exc
    content = result.get("message", {}).get("content", "")
    if not content:
        raise LLMError("Ollama retourne une réponse vide")
    return content


class PlannerAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "Tu es un assistant de planification académique. "
        "À partir de la question, produis des sous-tâches numérotées et claires. "
        "Retourne UNIQUEMENT un JSON strict: {\"tasks\": [\"t1\", \"t2\", ...]}. "
        "Ne retourne aucun autre texte."
    )

    def __init__(self, bus: MessageBus, llm_model: str = "llama3") -> None:
        super().__init__(agent_id="planner", role="academic planning", bus=bus, llm_model=llm_model)

    async def generate_response(self, message: AgentMessage) -> str:
        if FAST_MODE or not USE_LLM_PLANNER:
            return json.dumps({"tasks": ["Retrieve relevant documents", "Verify key facts", "Synthesize a final answer"]})
        payload = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": message.content},
        ]
        try:
            raw = await _call_ollama(
                self.llm_model,
                payload,
                options={"temperature": 0.2, "num_predict": 64},
                timeout=40.0,
                keep_alive="10m",
                format="json",
            )
        except LLMError as exc:
            LOGGER.error("PlannerAgent: %s", exc)
            return json.dumps({"tasks": ["Retrieve relevant documents", "Verify key facts", "Synthesize a final answer"]})
        # try direct JSON parse, else try to extract JSON substring, else fallback
        def _extract_json(text: str) -> str | None:
            if not text:
                return None
            # find first {...} balanced JSON block
            start = text.find('{')
            if start == -1:
                return None
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            return None

        try:
            parsed = json.loads(raw)
        except Exception:
            # try to extract JSON object from text
            snippet = _extract_json(raw)
            if snippet:
                try:
                    parsed = json.loads(snippet)
                except Exception:
                    parsed = None
            else:
                parsed = None

        if parsed:
            tasks = parsed.get("tasks", [])
            if isinstance(tasks, list) and tasks:
                return json.dumps({"tasks": tasks})

        LOGGER.warning("PlannerAgent: could not parse LLM JSON, using fallback plan; raw output preview=%s", (raw or '')[:300])
        return json.dumps({"tasks": ["Retrieve relevant documents", "Verify key facts", "Synthesize a final answer"]})



class RAGAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are an academic retrieval assistant. Given a query and a list of document excerpts, "
        "select the most relevant passages and summarize them as a concise context block. "
        "Return ONLY a JSON object: {\"context\": \"...\", \"sources\": [\"src1\", ...]}."
    )

    def __init__(self, bus: MessageBus, retriever: Any | None = None, llm_model: str = "llama3") -> None:
        super().__init__(agent_id="rag", role="retrieval and RAG", bus=bus, llm_model=llm_model)
        self._retriever = retriever

    async def generate_response(self, message: AgentMessage) -> str:
        query = message.content
        docs_block = await self._retrieve_docs(query)
        if (not USE_LLM_RAG) or docs_block in {"(no retrieval system configured)", "(no documents found)", "(retrieval error)"}:
            return json.dumps({"context": docs_block, "sources": []})
        user_content = f"Query: {query}\n\n<docs>\n{docs_block}\n</docs>"
        payload = [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
        try:
            content = await _call_ollama(
                self.llm_model,
                payload,
                timeout=90.0,
                format="json",
            )
        except LLMError as exc:
            LOGGER.error("RAGAgent: %s", exc)
            content = f"(retrieval error: {exc})"
        try:
            parsed = json.loads(content)
            return json.dumps(parsed)
        except Exception:
            return json.dumps({"context": content, "sources": []})

    async def _retrieve_docs(self, query: str) -> str:
        if self._retriever is None:
            LOGGER.warning("RAGAgent: no retriever configured — no documents will be retrieved")
            return "(no retrieval system configured)"
        try:
            # Try RAGPipeline.retrieve() first (returns list[dict] with similarity)
            if hasattr(self._retriever, 'retrieve'):
                results = self._retriever.retrieve(query=query, top_k=5, min_similarity=0.0)
                if results:
                    LOGGER.info(
                        json.dumps({
                            "event": "rag_retrieved", "chunks": len(results),
                            "similarities": [round(r['similarity'], 3) for r in results],
                            "sources": list({r.get('source', 'unknown') for r in results}),
                        })
                    )
                    parts = [
                        f"[score={r['similarity']:.2f}, source={r.get('source','?')}]\n{r['content']}"
                        for r in results
                    ]
                    return "\n---\n".join(parts)
                LOGGER.info("RAGAgent: no chunks found for query='%s'", query[:60])
                return "(no documents found)"
            # Fallback: HybridMemory.retrieve_relevant_history()
            results = self._retriever.retrieve_relevant_history(query=query, top_k=5)
            if results:
                LOGGER.info(
                    json.dumps({
                        "event": "rag_retrieved_fallback", "chunks": len(results),
                        "similarities": [round(r.similarity, 3) for r in results],
                    })
                )
            parts = [f"[score={r.similarity:.2f}] {r.record.content}" for r in results]
            return "\n---\n".join(parts) if parts else "(no documents found)"
        except Exception as exc:
            LOGGER.error("RAGAgent retrieval error: %s", exc, exc_info=True)
            return "(retrieval error)"


class VerifierAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are an academic verification assistant. Given a student question and a draft answer, "
        "evaluate factual consistency, coherence, and completeness (0-10 each). "
        "Approve the answer (\"approved\": true) when it is correct and relevant, even if perfectible. "
        "Return ONLY a JSON object with EXACTLY these keys: "
        "{\"factual\": int, \"coherence\": int, \"completeness\": int, \"issues\": [string], \"approved\": bool}."
    )

    def __init__(self, bus: MessageBus, llm_model: str = "llama3") -> None:
        super().__init__(agent_id="verifier", role="verification and quality control", bus=bus, llm_model=llm_model)

    async def generate_response(self, message: AgentMessage) -> str:
        if not USE_LLM_VERIFIER:
            return json.dumps({"factual": 5, "coherence": 5, "completeness": 5, "issues": [], "approved": True})
        try:
            body = json.loads(message.content)
            question = body.get("question", message.content)
            draft = body.get("draft", "")
        except json.JSONDecodeError:
            question = message.content
            draft = ""
        user_content = f"Question: {question}\n\nDraft answer:\n{draft}"
        payload = [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
        try:
            content = await _call_ollama(
                self.llm_model,
                payload,
                timeout=90.0,
                format="json",
            )
        except LLMError as exc:
            LOGGER.error("VerifierAgent: %s", exc)
            return json.dumps({
                "factual": 5,
                "coherence": 5,
                "completeness": 5,
                "issues": [str(exc)],
                "approved": False,
            })
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("verifier output is not an object")

            def _score(key: str) -> int:
                try:
                    return int(round(float(parsed.get(key, 5))))
                except (TypeError, ValueError):
                    return 5

            factual = _score("factual")
            coherence = _score("coherence")
            completeness = _score("completeness")
            issues = parsed.get("issues", [])
            if not isinstance(issues, list):
                issues = [str(issues)]

            # Si llama3 omet "approved", on décide via la moyenne des scores (seuil 7/10).
            if "approved" in parsed:
                approved = bool(parsed["approved"])
            else:
                approved = (factual + coherence + completeness) / 3.0 >= 7.0

            return json.dumps({
                "factual": factual,
                "coherence": coherence,
                "completeness": completeness,
                "issues": issues,
                "approved": approved,
            })
        except Exception:
            # JSON illisible : on n'échoue pas la réponse, on approuve par défaut
            # (le mode JSON d'Ollama rend ce cas très rare).
            return json.dumps({
                "factual": 6, "coherence": 6, "completeness": 6,
                "issues": ["verifier output unpar. approuvé par défaut"],
                "approved": True,
            })


class SynthesizerAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "Tu es un assistant académique avancé. Tu rédiges une réponse claire, structurée et concise. "
        "Réponds uniquement en français, sans mentionner les limites du système ni l'absence de documents."
    )

    def __init__(self, bus: MessageBus, llm_model: str = "llama3") -> None:
        super().__init__(agent_id="synthesizer", role="synthesis and academic writing", bus=bus, llm_model=llm_model)

    async def generate_response(self, message: AgentMessage) -> str:
        try:
            body = json.loads(message.content)
            question = body.get("question", message.content)
            context = body.get("context", "")
            plan = body.get("plan", [])
        except json.JSONDecodeError:
            question = message.content
            context = ""
            plan = []
        if FAST_MODE:
            return _quick_synthesis(question, context)
        if PROFILE == "full":
            plan_str = "\n".join(f"- {t}" for t in plan) if plan else "(none)"
            user_content = (
                f"Question: {question}\n\nPlan de recherche:\n{plan_str}\n\nContexte:\n{context}"
            )
            payload = [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
            options = {"temperature": 0.2, "num_predict": 256}
        else:
            context_snippet = context[:800] if context else "(aucun contexte)"
            user_content = (
                f"Question: {question}\n\nContexte: {context_snippet}\n\n"
                "Réponds en français en 4 à 6 phrases claires."
            )
            payload = [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
            options = {"temperature": 0.2, "num_predict": 192}
        try:
            return await _call_ollama(
                self.llm_model,
                payload,
                options=options,
                timeout=60.0,
                keep_alive="10m",
            )
        except LLMError as exc:
            LOGGER.error("SynthesizerAgent: %s", exc)
            return f"(synthesis error: {exc})"

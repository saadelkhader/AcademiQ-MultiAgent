"""Tests pipeline AcademiQ — vérifie l'absence de deadlock A2A.

Les agents lisent leurs flags d'activation LLM (USE_LLM_PLANNER, etc.) au
moment de l'import depuis A2A_PROFILE. Pour des tests déterministes on force
le profil "full" AVANT d'importer le module, puis on mocke ollama.chat.
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

# Forcer le profil "full" pour que chaque agent appelle réellement le LLM mocké,
# dans l'ordre plan -> retrieve -> synthesize -> verify.
os.environ.setdefault("A2A_PROFILE", "full")

from unittest.mock import MagicMock, patch


def mock_ollama_response(text):
    return {"message": {"content": text}}


@pytest.mark.asyncio
async def test_full_pipeline_no_bus_deadlock():
    """Test que le pipeline complet ne deadlock pas."""
    from src.orchestrator.academic_orchestrator import (
        AcademicOrchestrator,
        OrchestratorConfig,
    )

    with patch("src.agents.specialized_agents.ollama") as mock_ol:
        mock_ol.chat = MagicMock(
            side_effect=[
                mock_ollama_response(json.dumps({"tasks": ["Rechercher", "Analyser"]})),
                mock_ollama_response(
                    json.dumps({"context": "contexte test", "sources": []})
                ),
                mock_ollama_response(
                    "Voici une réponse académique complète sur le sujet."
                ),
                mock_ollama_response(
                    json.dumps(
                        {
                            "factual": 8,
                            "coherence": 9,
                            "completeness": 8,
                            "issues": [],
                            "approved": True,
                        }
                    )
                ),
            ]
        )
        orch = AcademicOrchestrator(config=OrchestratorConfig())
        result = await asyncio.wait_for(
            orch.answer("Qu'est-ce que la photosynthèse ?"),
            timeout=15.0,  # doit finir en moins de 15s, pas de deadlock
        )
        assert result.final_answer != ""
        assert result.approved is True
        assert len(result.plan) >= 1
        assert len(result.routing_decisions) >= 4
        print(f"OK Pipeline en {result.duration_ms:.0f}ms")


@pytest.mark.asyncio
async def test_planner_direct_call():
    """Test appel direct generate_response sans bus."""
    from src.agents.base_agent import AgentMessage, InMemoryMessageBus
    from src.agents.specialized_agents import PlannerAgent

    bus = InMemoryMessageBus()
    agent = PlannerAgent(bus)
    with patch("src.agents.specialized_agents.ollama") as mock_ol:
        mock_ol.chat = MagicMock(
            return_value=mock_ollama_response(
                json.dumps({"tasks": ["Étape 1", "Étape 2", "Étape 3"]})
            )
        )
        msg = AgentMessage(
            sender="test",
            receiver="planner",
            task_type="plan",
            content="Test question",
        )
        result = await agent.generate_response(msg)
        parsed = json.loads(result)
        assert "tasks" in parsed
        assert len(parsed["tasks"]) == 3
        print(f"OK PlannerAgent: {parsed['tasks']}")


@pytest.mark.asyncio
async def test_api_ask_endpoint():
    """Test endpoint /ask via httpx (skip si API down)."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            try:
                r = await client.get("http://localhost:8000/health", timeout=3)
            except Exception:
                pytest.skip("API non disponible")
            if r.status_code != 200:
                pytest.skip("API non disponible")
            # llama3 sur CPU peut prendre plusieurs minutes (plan+rag+synth+verify)
            r2 = await client.post(
                "http://localhost:8000/ask",
                json={"query": "Définir la photosynthèse en 2 phrases"},
                timeout=300,
            )
            assert r2.status_code == 200
            data = r2.json()
            assert "final_answer" in data
            assert data["final_answer"] != ""
            assert "routing_decisions" in data
            print(
                f"OK /ask: {data['duration_ms']:.0f}ms, approved={data['approved']}"
            )
    except ImportError:
        pytest.skip("httpx non installé")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline_no_bus_deadlock())
    asyncio.run(test_planner_direct_call())
    print("\nOK Tous les tests passent")

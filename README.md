# Orchestrateur Multi-Agents Adaptatif — Assistance Académique

> **Projet 1 — Architecture agentique** | Dominante scientifique

## Vue d'ensemble

Ce système implémente un assistant académique basé sur une architecture **multi-agents A2A** capable de décomposer une question complexe, récupérer des documents pertinents, produire une réponse argumentée, puis la vérifier automatiquement — le tout avec un routage dynamique adaptatif.

```
Requête utilisateur
       │
       ▼
 [DynamicRouter]      ← scoring multi-critères (uncertainty, context load, capabilities)
       │
       ├──► [PlannerAgent]      → décompose la tâche en sous-étapes
       │
       ├──► [RAGAgent]          → récupère le contexte documentaire (HybridMemory)
       │
       ├──► [SynthesizerAgent]  → produit un brouillon académique structuré
       │
       └──► [VerifierAgent]     → contrôle qualité + retry si score < seuil
                │
                ▼
          Réponse finale
```

---

## Architecture

### Agents spécialisés (`src/agents/`)

| Agent | Rôle | Output |
|-------|------|--------|
| `PlannerAgent` | Décompose la question en sous-tâches ordonnées | `{"tasks": [...]}` JSON |
| `RAGAgent` | Recherche et synthétise les passages pertinents | `{"context": "...", "sources": [...]}` |
| `SynthesizerAgent` | Rédige la réponse académique finale (intro + développement + conclusion) | Texte structuré |
| `VerifierAgent` | Évalue factuel/cohérence/complétude (0–10), approuve ou demande révision | `{"approved": bool, "issues": [...]}` |

Tous héritent de `BaseAgent` qui fournit : bus A2A, logging JSON structuré, intégration Ollama.

### Routeur dynamique (`src/orchestrator/dynamic_router.py`)

Score multi-critères pour sélectionner l'agent optimal :
- **Uncertainty** — ambiguïté de la requête (LLM ou heuristique)
- **Context load** — saturation du contexte de chaque agent
- **Capability match** — correspondance mots-clés capabilities ↔ requête
- **Availability** — charge système

### Mémoire hybride (`src/memory/hybrid_memory.py`)

- **Session** : in-memory, TTL configurable, pruning automatique
- **Persistante** : SQLite (métadonnées) + ChromaDB (embeddings)
- **Sémantique** : sentence-transformers local (all-MiniLM-L6-v2)

### Pipeline RAG (`src/rag/pipeline.py`)

- Ingestion : `.txt`, `.md`, `.pdf`
- Chunking par paragraphe avec overlap configurable
- Recherche par similarité cosinus via HybridMemory
- Construction du bloc contexte formaté pour le LLM

### Serveur MCP (`src/mcp_servers/`)

- `python_executor` : exécution Python sécurisée (sandbox multiprocess, timeout, memory limit)
- `file_access` : lecture de fichiers depuis le répertoire `data/`
- FastAPI sur `/mcp/discover` et `/mcp/tools/{name}/invoke`

---

## Installation

```bash
# 1. Cloner et créer l'environnement
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Installer Ollama (LLM local)
# https://ollama.com/download
ollama pull llama3

# 4. (Optionnel) Pré-télécharger le modèle d'embeddings
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

## Lancement

### API académique

```bash
uvicorn src.api.main:app --reload --port 8000
```

Endpoints :
- `POST /ask`          — soumettre une question académique
- `POST /ingest`       — indexer un document dans la RAG
- `GET  /health`       — état du service
- `GET  /pipeline/stats` — statistiques pipeline

Exemple :
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Expliquer le cycle de Krebs et son rôle dans la respiration cellulaire"}'
```

### Serveur MCP

```bash
uvicorn src.mcp_servers.server:app --port 8001
```

---

## Tests & Benchmarks

```bash
# Lancer tous les benchmarks avec pytest
pytest tests/benchmark.py -v

# Ou en standalone (génère benchmark_results.json et .csv)
python tests/benchmark.py
```

### Métriques mesurées

| Benchmark | Métrique |
|-----------|---------|
| Router accuracy | % correct agent selection |
| Planner agent | Valid JSON output, task count |
| Verifier agent | Approval decision accuracy |
| RAG chunking | Chunk count, retrieval recall |
| Message bus | Throughput (msg/sec) |

---

## Structure du projet

```
├── src/
│   ├── agents/
│   │   ├── base_agent.py           # BaseAgent, MessageBus, A2A protocol
│   │   └── specialized_agents.py  # Planner, RAG, Synthesizer, Verifier
│   ├── api/
│   │   └── main.py                # FastAPI — endpoint /ask, /ingest
│   ├── mcp_servers/
│   │   ├── server.py              # FastAPI MCP server
│   │   ├── registry.py            # Tool registry
│   │   ├── sandbox.py             # Python sandbox sécurisé
│   │   └── tools/                 # file_access, python_executor
│   ├── memory/
│   │   └── hybrid_memory.py       # Session + SQLite + ChromaDB
│   ├── orchestrator/
│   │   ├── dynamic_router.py      # Routeur multi-critères
│   │   └── academic_orchestrator.py # Pipeline complet
│   └── rag/
│       └── pipeline.py            # Ingestion + retrieval RAG
├── tests/
│   └── benchmark.py               # Suite de benchmarks
├── data/                          # Documents à indexer
├── requirements.txt
└── README.md
```

---

## Questions de recherche

1. **Architecture hiérarchique vs distribuée** — ce projet implémente les deux : pipeline séquentiel (hiérarchique) + routage dynamique (distribué). Le benchmark mesure laquelle produit des réponses plus robustes.

2. **Signaux de routage** — le `DynamicRouter` combine uncertainty, context load, et capability match. Les logs structurés permettent d'analyser quel signal domine par catégorie de question.

---

## Références

- A2A Protocol — Google DeepMind (2024)
- MCP (Model Context Protocol) — Anthropic (2024)
- RAG — Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- ChromaDB — Chroma (2023)
- Ollama — local LLM inference

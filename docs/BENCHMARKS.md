# AcademiQ — Rapport de Benchmarks

> Assistant Académique Multi-Agents (A2A · RAG · FastAPI · Ollama/llama3)
> Données mesurées et reproductibles. Dernière exécution : seed=42, runs=1.

Ce document regroupe **trois familles de mesures** :

1. **Expérience A — Comparaison de stratégies de routage** (statique hiérarchique
   vs dynamique adaptatif) — *étude de simulation* du routeur, LLM simulé.
2. **Expérience B — Benchmarks du pipeline réel** (orchestrateur réel, LLM mocké,
   exécution hors-ligne) — latence, génération d'examen, distribution du routage.
3. **Expérience C — Validation fonctionnelle bout-en-bout** (LLM llama3 réel via
   Ollama) — preuve d'appel LLM réel.

---

## Environnement de mesure

| Composant            | Valeur                                             |
|----------------------|----------------------------------------------------|
| Python               | 3.11+ (mesuré sous 3.14)                            |
| LLM local            | Ollama — llama3:latest (8B, quantization Q4_0)      |
| Embeddings           | sentence-transformers `all-MiniLM-L6-v2` (384 dim)  |
| Vector store         | ChromaDB 1.5.9 (persistant, cosinus)                |
| Chunking RAG         | chunk_size = 512, overlap = 64                      |
| Corpus indexé        | 111 chunks (documents académiques IA / Computer Vision) |
| Matériel LLM         | CPU (pas de GPU)                                     |
| Seed benchmark       | 42                                                  |

---

## Expérience A — Routage Statique vs Dynamique

**Protocole.** 15 requêtes variées par configuration (30 résultats au total).
Le LLM est **simulé** afin d'isoler l'effet de la *stratégie de routage* (et non
la latence du modèle). À lire comme une étude de simulation contrôlée.

### A.1 Synthèse des métriques

| Métrique             | Fixed Hierarchical | Dynamic Adaptive |
|----------------------|--------------------|------------------|
| Latence moyenne (ms) | **120.90**         | 205.12           |
| Latence p50 (ms)     | 116.71             | 208.75           |
| Latence p90 (ms)     | **164.20**         | 230.48           |
| Latence min (ms)     | 66.18              | 172.30           |
| Latence max (ms)     | 202.72             | 239.06           |
| Taux d'échec         | **0.133**          | 0.467            |
| Qualité moyenne      | **0.833**          | 0.533            |
| Tokens moyens        | 20.67              | **19.47**        |
| Coût contexte moyen  | **21.67**          | 21.82            |
| Appels d'agent moy.  | 1                  | 1                |

*(En gras = meilleure valeur sur la ligne.)*

### A.2 Profil radar (scores normalisés 0–1, plus haut = mieux)

| Dimension      | Fixed Hierarchical | Dynamic Adaptive |
|----------------|--------------------|------------------|
| latency        | 0.683              | 0.196            |
| tokens         | 0.492              | 0.549            |
| context_cost   | 0.558              | 0.552            |
| reliability    | 1.000              | 0.000            |
| quality        | 0.833              | 0.533            |
| efficiency     | 0.000              | 0.000            |

### A.3 Distribution des latences (ms, triées)

```
Fixed Hierarchical :
 66.2  75.0  82.5  82.9  84.8  90.2 107.2 116.7
135.8 139.5 141.9 160.3 162.7 165.2 202.7

Dynamic Adaptive :
172.3 184.1 185.8 189.0 189.8 195.3 196.3 208.7
210.6 212.9 213.5 221.5 221.5 236.4 239.1
```

### A.4 Interprétation (honnête)

Dans cette simulation, **le routage statique hiérarchique surpasse le routage
dynamique adaptatif** sur la latence (−41 %), la fiabilité (taux d'échec 13.3 %
vs 46.7 %) et la qualité (0.833 vs 0.533). Le routage dynamique introduit un
surcoût (évaluation d'incertitude + scoring multi-critères) qui n'est **pas**
amorti sur ce jeu de requêtes réduit et peu variable.

**À retenir :** le routage dynamique se justifie dans des contextes à **forte
variabilité** de requêtes, de charge ou de disponibilité des agents — pas sur un
flux homogène. C'est un axe de discussion / travaux futurs, pas une supériorité
établie ici.

---

## Expérience B — Benchmarks du pipeline réel (LLM mocké)

**Protocole.** Orchestrateur réel (`AcademicOrchestrator`), LLM Ollama **mocké**
pour une exécution déterministe et hors-ligne. Commande :
`python -m tests.benchmark --pipeline`

### B.1 Latence du pipeline complet

| Métrique           | Valeur     |
|--------------------|------------|
| Itérations         | 3          |
| Durée moyenne (ms) | 1.82       |
| Durée min (ms)     | 1.56       |
| Durée max (ms)     | 2.28       |
| Taux d'approbation | **1.0**    |

> Pipeline complet (plan → retrieve → synthesize → verify) en **~2 ms** sans
> deadlock avec LLM mocké → preuve que la correction du bus A2A élimine
> l'interblocage (le défaut original provoquait des timeouts de ~30 s/étape).

### B.2 Génération d'examen

| Métrique       | Valeur |
|----------------|--------|
| Réponse vide ? | non    |
| Approuvé ?     | oui    |
| Longueur (car.)| 51     |
| Durée (ms)     | 1.81   |

### B.3 Distribution du routage (couverture des agents)

Sur **20 requêtes variées**, chaque agent est sélectionné :

| Agent       | Sélections |
|-------------|------------|
| planner     | 5          |
| rag         | 5          |
| synthesizer | 5          |
| verifier    | 5          |

✅ `all_agents_selected = true` — les 4 agents sont effectivement atteignables
par le routeur (pas d'agent « mort »).

---

## Expérience C — Validation fonctionnelle (llama3 réel)

**Protocole.** Requête réelle via `POST /ask` avec Ollama/llama3 actif (CPU).

Requête : *« Définis la photosynthèse en 2 phrases »*

| Indicateur         | Valeur observé                                   |
|--------------------|--------------------------------------------------|
| Code HTTP          | 200                                              |
| Header `X-Real-LLM`| `true`                                           |
| Durée totale       | ≈ 178 939 ms (~3 min, llama3 8B sur CPU)         |
| `approved`         | true                                             |
| `retry_count`      | 0                                                |
| Étapes de routage  | plan, retrieve, synthesize, verify_0 (4)         |
| Scores Verifier    | factual=9, coherence=10, completeness=8          |

> La durée >> 500 ms confirme un **véritable appel au LLM** (pas de fallback ni
> de mode démo). `retry_count = 0` confirme que le mode JSON contraint (Ollama
> `format=json`) fiabilise le parsing du Verifier (avant correctif : 3 retries
> inutiles et ~162 s).

### Tests automatisés (pytest)

```
tests/test_pipeline.py::test_full_pipeline_no_bus_deadlock  PASSED
tests/test_pipeline.py::test_planner_direct_call            PASSED
tests/test_pipeline.py::test_api_ask_endpoint               PASSED
=> 3 passed
```

---

## Configuration du routeur (paramètres utilisés)

| Poids / paramètre        | Valeur |
|--------------------------|--------|
| uncertainty_weight       | 0.40   |
| confidence_weight        | 0.35   |
| context_weight           | 0.25   |
| capability_weight        | 0.20   |
| availability_weight      | 0.10   |
| tool_weight              | 0.10   |
| system_weight            | 0.10   |
| fallback_threshold       | 0.40   |
| max_context_tokens       | 8192   |

Confiances de base par agent : Planner 0.85 · RAG 0.80 · Synthesizer 0.90 ·
Verifier 0.75.

---

## Reproduire les mesures

```bash
# Expérience A — routage statique vs dynamique (LLM simulé)
python -m tests.benchmark --seed 42 --runs 1
#   -> tests/benchmark_results.json (résumé + viz) et benchmark_results.csv

# Expérience B — pipeline réel, LLM mocké, hors-ligne
python -m tests.benchmark --pipeline

# Expérience C — validation fonctionnelle (nécessite Ollama + llama3)
ollama serve            # terminal 1
python -m uvicorn src.api.main:app --port 8000   # terminal 2
pytest tests/test_pipeline.py -v
```

---

## Limites des mesures

- La latence de l'**Expérience A** est *simulée* : elle compare des stratégies de
  routage, **pas** la latence réelle du LLM (voir Expérience C pour le réel).
- Le routage dynamique n'a **pas** surpassé le statique sur ce jeu réduit (15
  requêtes/config) — résultat à ne pas surinterpréter.
- Corpus RAG limité (111 chunks) ; pas d'évaluation utilisateur formelle.
- LLM exécuté sur CPU (~2–3 min/requête) ; non optimisé GPU.
- La métrique de qualité de l'Expérience A est partiellement simulée.

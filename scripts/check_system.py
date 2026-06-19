"""AcademiQ System Diagnostic — scripts/check_system.py

Usage:
    python scripts/check_system.py

Vérifie tous les composants nécessaires au bon fonctionnement du pipeline.
"""
import sys
import time

# Windows: la console par défaut (cp1252) ne peut pas encoder les emoji.
# On reconfigure stdout/stderr en UTF-8 pour éviter un crash UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

print("=" * 55)
print("  AcademiQ — System Check")
print("=" * 55)
print()

# ── Ollama
print("[1/6] Ollama")
try:
    import ollama as _ollama
    models_resp = _ollama.list()
    model_names = [m.get("model", m.get("name", "?")) for m in models_resp.get("models", [])]
    print(f"  ✅ Ollama disponible — {len(model_names)} modèle(s)")
    for m in model_names:
        print(f"     • {m}")
    if not model_names:
        print("  ⚠️  Aucun modèle téléchargé — lancez: ollama pull llama3")
    elif "llama3" not in " ".join(model_names) and "llama3:latest" not in model_names:
        print("  ⚠️  llama3 non trouvé — lancez: ollama pull llama3")
except ConnectionError:
    print("  ❌ Ollama non démarré")
    print("     Fix: ollama serve")
except ImportError:
    print("  ❌ Package ollama absent")
    print("     Fix: pip install ollama")
except Exception as e:
    print(f"  ❌ Erreur Ollama: {e}")

print()

# ── sentence-transformers
print("[2/6] sentence-transformers (embeddings)")
try:
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(["test"], normalize_embeddings=True)
    ms = round((time.time() - t0) * 1000)
    print(f"  ✅ sentence-transformers OK — dim={len(emb[0])} — load={ms}ms")
except ImportError:
    print("  ❌ sentence-transformers absent")
    print("     Fix: pip install sentence-transformers")
except Exception as e:
    print(f"  ❌ Erreur: {e}")

print()

# ── ChromaDB
print("[3/6] ChromaDB (vector store)")
try:
    import chromadb
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("test")
    col.add(ids=["t1"], embeddings=[[0.1] * 384], documents=["test"])
    r = col.query(query_embeddings=[[0.1] * 384], n_results=1)
    print(f"  ✅ ChromaDB OK — version={chromadb.__version__}")
except ImportError:
    print("  ❌ chromadb absent")
    print("     Fix: pip install chromadb")
except Exception as e:
    print(f"  ❌ Erreur ChromaDB: {e}")

print()

# ── FastAPI + uvicorn
print("[4/6] FastAPI + uvicorn")
try:
    import fastapi, uvicorn
    print(f"  ✅ FastAPI {fastapi.__version__} | uvicorn {uvicorn.__version__}")
except ImportError as e:
    print(f"  ❌ {e}")
    print("     Fix: pip install fastapi uvicorn")

print()

# ── API /health
print("[5/6] API backend (http://localhost:8000)")
try:
    import urllib.request, json
    r = urllib.request.urlopen("http://localhost:8000/health", timeout=3)
    data = json.loads(r.read())
    print(f"  ✅ API OK — status={data.get('status')} version={data.get('version')}")
    print(f"     rag_ready={data.get('rag_ready')} orch_ready={data.get('orchestrator_ready')}")
except Exception as e:
    print(f"  ❌ API non accessible: {e}")
    print("     Fix: python -m uvicorn src.api.main:app --port 8000")

print()

# ── /debug/status
print("[6/6] Pipeline diagnostic (/debug/status)")
try:
    import urllib.request, json
    r = urllib.request.urlopen("http://localhost:8000/debug/status", timeout=5)
    data = json.loads(r.read())
    print(f"  ✅ /debug/status OK")
    print(f"     ollama={data.get('ollama_available')} models={data.get('ollama_models')}")
    print(f"     chromadb={data.get('chromadb_available')} sentence_transformers={data.get('sentence_transformers_available')}")
    print(f"     rag_ready={data.get('rag_ready')} docs={data.get('rag_docs_count')}")
    print(f"     requests={data.get('pipeline_requests')} approved={data.get('pipeline_approved')}")
except Exception as e:
    print(f"  ❌ /debug/status non accessible: {e}")

print()
print("=" * 55)
print()
print("Pour lancer l'API:")
print("  .venv\\Scripts\\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
print()
print("Pour tester /ask directement:")
print('  curl -X POST http://localhost:8000/ask \\')
print('    -H "Content-Type: application/json" \\')
print('    -d "{\"query\": \"Explique la photosynthese\"}"')
print()

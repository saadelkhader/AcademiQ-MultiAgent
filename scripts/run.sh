#!/bin/bash
echo "🚀 Démarrage AcademiQ..."

# Vérifier ollama
if ! pgrep -x "ollama" > /dev/null; then
    echo "📦 Démarrage Ollama..."
    ollama serve &
    sleep 3
fi

# Vérifier llama3
if ! ollama list | grep -q "llama3"; then
    echo "📥 Téléchargement llama3..."
    ollama pull llama3
fi

echo "✅ Ollama prêt"
echo "🌐 Démarrage API sur port 8000..."
uvicorn src.api.main:app --reload --port 8000 --host 0.0.0.0

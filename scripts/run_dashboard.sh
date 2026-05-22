#!/bin/bash
# Run the ICU Clinical Copilot Dashboard (FastAPI)
# Usage: bash scripts/run_dashboard.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "🏥 ICU Clinical Copilot — Starting FastAPI Dashboard"
echo "======================================================"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Set PYTHONPATH so src/ imports work (e.g. from api.main import ...)
export PYTHONPATH="$PROJECT_DIR/src"

# Ensure frontend/ directory exists
mkdir -p "$PROJECT_DIR/frontend"

echo "🚀 Launching FastAPI on http://localhost:8000"
echo "   API docs: http://localhost:8000/docs"
echo "   Press Ctrl+C to stop."
echo ""

# ── Knowledge Base mode ───────────────────────────────────────────────────────
# Set SKIP_PRIMEKG=1 to skip the heavy PrimeKG graph load.
export SKIP_PRIMEKG=1

echo "🚀 Launching FastAPI on http://localhost:8000"
echo "   API docs: http://localhost:8000/docs"
echo "   Press Ctrl+C to stop."
echo ""

uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir src

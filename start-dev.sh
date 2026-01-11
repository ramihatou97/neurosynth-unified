#!/bin/bash
# NeuroSynth - Start development servers

cd "$(dirname "$0")"

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
    echo "✓ Loaded environment from .env"
fi

echo "=========================================="
echo "  NeuroSynth Development Environment"
echo "=========================================="

# Stop any existing processes first
./stop-dev.sh

echo ""
echo "Starting services..."

# Start backend (uses env vars from .env)
echo "→ Starting backend on :8000..."
./venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "→ Starting frontend on :3000..."
cd /Users/ramihatoum/Desktop/frontend
npm run dev -- --port 3000 &
FRONTEND_PID=$!

cd - >/dev/null

sleep 2

echo ""
echo "=========================================="
echo "  Services Running:"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "  To stop: ./stop-dev.sh"
echo "=========================================="

# Keep script running (Ctrl+C to stop both)
wait

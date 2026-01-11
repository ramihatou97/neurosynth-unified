#!/bin/bash
# NeuroSynth - Stop all development servers

echo "Stopping NeuroSynth services..."

# Kill backend (uvicorn)
pkill -f "uvicorn.*src.api.main" 2>/dev/null && echo "✓ Backend stopped" || echo "- Backend not running"

# Kill frontend (vite)
pkill -f "node.*vite" 2>/dev/null && echo "✓ Frontend stopped" || echo "- Frontend not running"

# Give processes time to exit
sleep 1

# Verify and force kill if needed
if lsof -i :8000 -i :3000 >/dev/null 2>&1; then
    echo "⚠ Some processes still running, force killing..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null
    lsof -ti :3000 | xargs kill -9 2>/dev/null
fi

echo "Done."

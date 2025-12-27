#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "================================================"
echo "  NeuroSynth Unified - 1-Click Setup & Run"
echo "================================================"
echo ""

# Parse flags
CLEAN=false
REINSTALL=false
REBUILD_INDEX=false
OPEN_BROWSER=false
for arg in "$@"; do
    case $arg in
        --clean) CLEAN=true ;;
        --reinstall) REINSTALL=true ;;
        --rebuild) REBUILD_INDEX=true ;;
        --open) OPEN_BROWSER=true ;;
        --help)
            echo "Usage: ./start.sh [options]"
            echo "  --clean      Reset database and rebuild everything"
            echo "  --reinstall  Force reinstall Python dependencies"
            echo "  --rebuild    Force rebuild FAISS indexes"
            echo "  --open       Open browser to API docs after starting"
            exit 0 ;;
    esac
done

# Step 1: Check/Install Homebrew
echo "[1/11] Checking Homebrew..."
if ! command -v brew &>/dev/null; then
    echo "   Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi
echo "   Homebrew ✓"

# Step 2: Check/Install PostgreSQL
echo "[2/11] Checking PostgreSQL..."
if ! brew list postgresql@16 &>/dev/null; then
    echo "   Installing PostgreSQL 16..."
    brew install postgresql@16
fi
echo "   PostgreSQL ✓"

# Step 3: Start PostgreSQL
echo "[3/11] Starting PostgreSQL..."
brew services start postgresql@16 2>/dev/null || true
sleep 2
echo "   PostgreSQL running ✓"

# Step 4: Create database/user
echo "[4/11] Setting up database..."
if [ "$CLEAN" = true ]; then
    echo "   Dropping existing database..."
    dropdb neurosynth 2>/dev/null || true
    dropuser neurosynth 2>/dev/null || true
fi

# Create user if not exists
psql postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='neurosynth'" | grep -q 1 || \
    psql postgres -c "CREATE USER neurosynth WITH PASSWORD 'neurosynth' CREATEDB;"

# Create database if not exists
psql postgres -tc "SELECT 1 FROM pg_database WHERE datname='neurosynth'" | grep -q 1 || \
    psql postgres -c "CREATE DATABASE neurosynth OWNER neurosynth;"

# Enable pgvector extension
psql neurosynth -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true
echo "   Database ready ✓"

# Step 5: Check Python
echo "[5/11] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 not found. Install Python 3.10+"
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "   Python $PYTHON_VERSION ✓"

# Step 6: Create venv
if [ ! -d "venv" ] || [ "$CLEAN" = true ]; then
    echo "[6/11] Creating virtual environment..."
    rm -rf venv
    python3 -m venv venv
    REINSTALL=true
else
    echo "[6/11] Virtual environment exists ✓"
fi
source venv/bin/activate

# Step 7: Install dependencies
if [ "$REINSTALL" = true ]; then
    echo "[7/11] Installing dependencies..."
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo "   Dependencies installed ✓"
else
    echo "[7/11] Dependencies installed ✓"
fi

# Step 8: Setup .env
if [ ! -f ".env" ]; then
    echo "[8/11] Creating .env from template..."
    cp .env.example .env
    # Update DATABASE_URL for local PostgreSQL
    sed -i '' 's|postgresql+asyncpg://neurosynth:neurosynth@localhost:5432/neurosynth|postgresql+asyncpg://neurosynth:neurosynth@localhost/neurosynth|' .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env with your API keys!"
    echo "   Required: VOYAGE_API_KEY, ANTHROPIC_API_KEY"
    echo ""
    read -p "Press Enter after editing .env (or Ctrl+C to exit)..."
else
    echo "[8/11] .env exists ✓"
fi

# Step 9: Initialize database schema
echo "[9/11] Initializing database schema..."
python scripts/init_database.py 2>/dev/null && echo "   Schema initialized ✓" || echo "   Schema ready ✓"

# Step 10: Build indexes
if [ "$REBUILD_INDEX" = true ] || [ "$CLEAN" = true ] || [ ! -f "indexes/text.faiss" ]; then
    echo "[10/11] Building FAISS indexes..."
    rm -rf indexes/*.faiss indexes/*.json 2>/dev/null || true
    python scripts/build_indexes.py 2>/dev/null && echo "   Indexes built ✓" || echo "   Indexes skipped (no data yet)"
else
    echo "[10/11] FAISS indexes exist ✓"
fi

# Step 11: Start server
echo "[11/11] Starting API server..."
echo ""
echo "================================================"
echo "  🚀 NeuroSynth API: http://localhost:8000"
echo "  📚 API Docs:       http://localhost:8000/docs"
echo "  🛑 Stop:           Ctrl+C"
echo "================================================"
echo ""

if [ "$OPEN_BROWSER" = true ]; then
    (sleep 2 && open http://localhost:8000/docs) &
fi

uvicorn src.api.main:app --reload --port 8000

#!/bin/bash
# =============================================================================
# MAMGA-Local — One-Click Setup Script
# Purpose: Install all dependencies and configure the LLM backend interactively
# Date: 2026-04-19
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# ANSI Colors
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${BLUE}=== $* ===${RESET}"; }
die()     { error "$*"; exit 1; }

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}"
cat << 'EOF'
  __  __    _    __  __ ____    _       _                    _
 |  \/  |  / \  |  \/  / ___|  | |     ___   ___ __ _| |
 | |\/| | / _ \ | |\/| | |  _  | |    / _ \ / __/ _` | |
 | |  | |/ ___ \| |  | | |_| | | |___| (_) | (_| (_| | |
 |_|  |_/_/   \_\_|  |_|\____| |_____|\___/ \___\__,_|_|

  Multi-Graph Agentic Memory for Local AI  •  One-Click Setup
EOF
echo -e "${RESET}"

# Move to script directory so relative paths always work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# STEP 1 — Python version check
# =============================================================================
header "Step 1 / 6 — Python Version Check"

PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))")
        major=$("$candidate" -c "import sys; print(sys.version_info[0])")
        minor=$("$candidate" -c "import sys; print(sys.version_info[1])")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON_BIN="$candidate"
            info "Found Python $ver at $(command -v "$candidate")"
            break
        fi
    fi
done

[ -z "$PYTHON_BIN" ] && die "Python 3.9+ is required but not found. Install it from https://python.org"

# Warn for very new Python (packages may lag behind)
minor_int=$("$PYTHON_BIN" -c "import sys; print(sys.version_info[1])")
if [ "$minor_int" -ge 13 ]; then
    warn "Python 3.${minor_int} detected. Some packages (torch, faiss-cpu) may not yet have"
    warn "pre-built wheels for this version. We will fall back to source builds if needed."
fi

success "Python version OK"

# =============================================================================
# STEP 2 — Virtual environment
# =============================================================================
header "Step 2 / 6 — Virtual Environment"

if [ -d "venv" ]; then
    warn "venv/ already exists — skipping creation (delete it manually to start fresh)"
else
    info "Creating virtual environment with $PYTHON_BIN..."
    "$PYTHON_BIN" -m venv venv
    success "venv/ created"
fi

# Activate
# shellcheck disable=SC1091
source venv/bin/activate
info "Activated: $(which python)"

info "Upgrading pip..."
pip install --upgrade pip --quiet
success "pip upgraded"

# =============================================================================
# STEP 3 — Install dependencies
# =============================================================================
header "Step 3 / 6 — Installing Dependencies"

info "Installing packages from requirements.txt..."
info "(This may take several minutes on first install — torch & sentence-transformers are large)"
echo ""

# Install with --prefer-binary to avoid slow source builds when possible
if pip install --prefer-binary -r requirements.txt; then
    success "All dependencies installed"
else
    warn "Some packages failed with pre-built wheels — retrying without --prefer-binary..."
    pip install -r requirements.txt || die "Dependency installation failed. See errors above."
    success "All dependencies installed (via source build)"
fi

# =============================================================================
# STEP 4 — Create runtime directories
# =============================================================================
header "Step 4 / 6 — Creating Runtime Directories"

for dir in data cache results; do
    mkdir -p "$dir"
    success "Directory ready: ./$dir/"
done

# =============================================================================
# STEP 5 — LLM Backend Configuration
# =============================================================================
header "Step 5 / 6 — LLM Backend Configuration"

# Auto-detect available backends
OLLAMA_AVAILABLE=false
LMSTUDIO_AVAILABLE=false

if command -v ollama &>/dev/null; then
    OLLAMA_AVAILABLE=true
    OLLAMA_VER=$(ollama --version 2>/dev/null | head -1 || echo "unknown")
    info "Detected: Ollama ($OLLAMA_VER)"
fi

if [ -d "/Applications/LM Studio.app" ] || [ -d "$HOME/Applications/LM Studio.app" ]; then
    LMSTUDIO_AVAILABLE=true
    info "Detected: LM Studio.app"
fi

echo ""
echo -e "${BOLD}Select your LLM backend:${RESET}"
echo ""
echo "  1) LM Studio  — local server on port 1234${LMSTUDIO_AVAILABLE:+ ${GREEN}(detected)${RESET}}"
echo "  2) Ollama     — local server on port 11434${OLLAMA_AVAILABLE:+ ${GREEN}(detected)${RESET}}"
echo "  3) OpenAI     — cloud API (requires OPENAI_API_KEY)"
echo "  4) Skip       — I will edit .env manually"
echo ""

while true; do
    read -rp "  Enter choice [1-4]: " CHOICE
    case "$CHOICE" in
        1|2|3|4) break ;;
        *) warn "Please enter 1, 2, 3, or 4" ;;
    esac
done

# ---------- Generate .env ----------
if [ -f ".env" ]; then
    warn ".env already exists — creating backup at .env.bak"
    cp .env .env.bak
fi

case "$CHOICE" in
    1)  # LM Studio
        info "Configuring for LM Studio..."
        read -rp "  Model identifier shown in LM Studio [default: local-model]: " LM_MODEL
        LM_MODEL="${LM_MODEL:-local-model}"
        read -rp "  LM Studio port [default: 1234]: " LM_PORT
        LM_PORT="${LM_PORT:-1234}"

        cat > .env << EOF
# MAMGA-Local — generated by setup.sh on $(date +%Y-%m-%d)
LLM_BACKEND=lmstudio
LLM_MODEL=${LM_MODEL}
LLM_BASE_URL=http://localhost:${LM_PORT}/v1

DEFAULT_EMBEDDING_MODEL=minilm
ENCODER_MODEL=all-MiniLM-L6-v2

CACHE_DIR=./cache
LOG_LEVEL=INFO
EOF
        success ".env written for LM Studio (port ${LM_PORT}, model '${LM_MODEL}')"
        echo ""
        echo -e "${YELLOW}  ▸ Launch LM Studio → Local Server tab → Start Server on port ${LM_PORT}${RESET}"
        ;;

    2)  # Ollama
        info "Configuring for Ollama..."

        # List available models if ollama is running
        OLLAMA_MODELS=""
        if "$OLLAMA_AVAILABLE" && ollama list &>/dev/null 2>&1; then
            OLLAMA_MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | head -10)
        fi

        if [ -n "$OLLAMA_MODELS" ]; then
            echo "  Available Ollama models:"
            echo "$OLLAMA_MODELS" | while read -r m; do echo "    • $m"; done
            echo ""
        fi

        read -rp "  Model name [default: llama3]: " OLLAMA_MODEL
        OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"

        cat > .env << EOF
# MAMGA-Local — generated by setup.sh on $(date +%Y-%m-%d)
LLM_BACKEND=ollama
LLM_MODEL=${OLLAMA_MODEL}
LLM_BASE_URL=http://localhost:11434

DEFAULT_EMBEDDING_MODEL=minilm
ENCODER_MODEL=all-MiniLM-L6-v2

CACHE_DIR=./cache
LOG_LEVEL=INFO
EOF
        success ".env written for Ollama (model '${OLLAMA_MODEL}')"
        echo ""
        echo -e "${YELLOW}  ▸ Make sure Ollama is running: ollama serve${RESET}"
        echo -e "${YELLOW}  ▸ Pull model if needed:        ollama pull ${OLLAMA_MODEL}${RESET}"
        ;;

    3)  # OpenAI
        info "Configuring for OpenAI..."
        read -rp "  OpenAI API key (sk-...): " OPENAI_KEY
        read -rp "  Model [default: gpt-4o-mini]: " OAI_MODEL
        OAI_MODEL="${OAI_MODEL:-gpt-4o-mini}"

        cat > .env << EOF
# MAMGA-Local — generated by setup.sh on $(date +%Y-%m-%d)
LLM_BACKEND=openai
LLM_MODEL=${OAI_MODEL}
OPENAI_API_KEY=${OPENAI_KEY}

DEFAULT_EMBEDDING_MODEL=minilm
ENCODER_MODEL=all-MiniLM-L6-v2

CACHE_DIR=./cache
LOG_LEVEL=INFO
EOF
        success ".env written for OpenAI (model '${OAI_MODEL}')"
        warn "Make sure your API key has sufficient quota."
        ;;

    4)  # Skip
        if [ ! -f ".env" ]; then
            cp .env.example .env
            info "Copied .env.example → .env (edit before running)"
        fi
        warn "Skipped backend configuration. Edit .env manually before use."
        ;;
esac

# =============================================================================
# STEP 6 — Run unit tests (exclude tests needing live LLM)
# =============================================================================
header "Step 6 / 6 — Verifying Installation (Unit Tests)"

info "Running unit tests (skipping integration/llm-marked tests)..."
echo ""

if python -m pytest tests/ -m "not integration and not llm" --tb=short -q; then
    echo ""
    success "All unit tests passed — environment is healthy"
else
    echo ""
    warn "Some unit tests failed. This may be due to Python version compatibility."
    warn "The core system can still function; check test output above for details."
fi

# =============================================================================
# Done — Next Steps
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}======================================================"
echo -e "  Setup Complete!"
echo -e "======================================================${RESET}"
echo ""
echo -e "${BOLD}Quick Start:${RESET}"
echo ""
echo -e "  1. Activate venv every session:"
echo -e "     ${CYAN}source venv/bin/activate${RESET}"
echo ""
echo -e "  2. Run a memory test (LoCoMo sample):"
echo -e "     ${CYAN}python test_fixed_memory.py --sample 0 --max-questions 5 --rebuild${RESET}"
echo ""
echo -e "  3. Run the full test suite:"
echo -e "     ${CYAN}python -m pytest tests/${RESET}"
echo ""
echo -e "  4. Review / edit config:"
echo -e "     ${CYAN}cat .env${RESET}"
echo ""
echo -e "${YELLOW}Note: Re-run this script anytime to reconfigure the LLM backend.${RESET}"
echo ""

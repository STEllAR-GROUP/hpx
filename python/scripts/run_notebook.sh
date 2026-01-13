#!/bin/bash
# HPXPy Notebook Runner
#
# SPDX-License-Identifier: BSL-1.0
#
# Run Jupyter notebooks in Docker container or locally.
#
# Usage:
#   ./scripts/run_notebook.sh                     # Start Jupyter server
#   ./scripts/run_notebook.sh --execute FILE.ipynb  # Execute notebook
#   ./scripts/run_notebook.sh --docker            # Run in Docker
#   ./scripts/run_notebook.sh --docker --execute tutorials/01_getting_started.ipynb

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
IMAGE_NAME="${HPXPY_IMAGE_NAME:-hpxpy}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
USE_DOCKER=false
EXECUTE_NOTEBOOK=""
NOTEBOOK_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker|-d)
            USE_DOCKER=true
            shift
            ;;
        --execute|-e)
            EXECUTE_NOTEBOOK="$2"
            shift 2
            ;;
        --port|-p)
            JUPYTER_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [NOTEBOOK]"
            echo ""
            echo "Options:"
            echo "  --docker, -d        Run in Docker container"
            echo "  --execute, -e FILE  Execute notebook non-interactively"
            echo "  --port, -p PORT     Jupyter server port (default: 8888)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Start Jupyter server locally"
            echo "  $0 --docker                           # Start Jupyter in Docker"
            echo "  $0 --execute tutorials/01_getting_started.ipynb"
            echo "  $0 --docker --execute benchmarks/07_benchmarks.ipynb"
            exit 0
            ;;
        *)
            if [[ -z "$NOTEBOOK_PATH" ]]; then
                NOTEBOOK_PATH="$1"
            fi
            shift
            ;;
    esac
done

cd "$PROJECT_DIR"

# Install Jupyter if needed (local mode)
install_jupyter_local() {
    if ! command -v jupyter &> /dev/null; then
        log_info "Installing Jupyter..."
        pip install jupyter nbconvert
    fi
}

# Execute notebook
execute_notebook() {
    local notebook="$1"
    local output_dir="$(dirname "$notebook")"
    local output_name="$(basename "$notebook" .ipynb)_executed.ipynb"

    log_info "Executing notebook: $notebook"

    if $USE_DOCKER; then
        docker run --rm \
            -v "$PROJECT_DIR:/opt/hpxpy" \
            -w /opt/hpxpy \
            "${IMAGE_NAME}:latest" \
            bash -c "
                pip install -q jupyter nbconvert &&
                jupyter nbconvert --to notebook --execute \
                    --output '$output_name' \
                    --ExecutePreprocessor.timeout=600 \
                    '$notebook'
            "
    else
        source .venv/bin/activate 2>/dev/null || true
        install_jupyter_local
        PYTHONPATH=build:. jupyter nbconvert \
            --to notebook \
            --execute \
            --output "$output_name" \
            --ExecutePreprocessor.timeout=600 \
            "$notebook"
    fi

    log_info "Output saved to: $output_dir/$output_name"
}

# Start Jupyter server
start_jupyter_server() {
    log_info "Starting Jupyter server on port $JUPYTER_PORT..."

    if $USE_DOCKER; then
        log_info "Running in Docker container..."
        docker run --rm -it \
            -p "${JUPYTER_PORT}:${JUPYTER_PORT}" \
            -v "$PROJECT_DIR:/opt/hpxpy" \
            -w /opt/hpxpy \
            "${IMAGE_NAME}:latest" \
            bash -c "
                pip install -q jupyter &&
                jupyter notebook \
                    --ip=0.0.0.0 \
                    --port=${JUPYTER_PORT} \
                    --no-browser \
                    --allow-root \
                    --NotebookApp.token=''
            "
    else
        source .venv/bin/activate 2>/dev/null || true
        install_jupyter_local
        log_info "Open http://localhost:${JUPYTER_PORT} in your browser"
        PYTHONPATH=build:. jupyter notebook \
            --port="${JUPYTER_PORT}" \
            --NotebookApp.token=''
    fi
}

# Main
if [[ -n "$EXECUTE_NOTEBOOK" ]]; then
    execute_notebook "$EXECUTE_NOTEBOOK"
elif [[ -n "$NOTEBOOK_PATH" ]]; then
    execute_notebook "$NOTEBOOK_PATH"
else
    start_jupyter_server
fi

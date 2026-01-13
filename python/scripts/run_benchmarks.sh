#!/bin/bash
# HPXPy Benchmark Runner
#
# SPDX-License-Identifier: BSL-1.0
#
# Run benchmarks locally or in Docker with result collection.
#
# Usage:
#   ./scripts/run_benchmarks.sh              # Run locally
#   ./scripts/run_benchmarks.sh --docker     # Run in Docker
#   ./scripts/run_benchmarks.sh --save       # Save results to JSON

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
IMAGE_NAME="${HPXPY_IMAGE_NAME:-hpxpy}"
RESULTS_DIR="${PROJECT_DIR}/benchmark_results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${BLUE}=== $1 ===${NC}\n"; }

# Parse arguments
USE_DOCKER=false
SAVE_RESULTS=false
BENCHMARK_FILE="benchmarks/benchmark_arrays.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker|-d)
            USE_DOCKER=true
            shift
            ;;
        --save|-s)
            SAVE_RESULTS=true
            shift
            ;;
        --file|-f)
            BENCHMARK_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --docker, -d      Run benchmarks in Docker container"
            echo "  --save, -s        Save results to JSON file"
            echo "  --file, -f FILE   Benchmark file to run (default: benchmarks/benchmark_arrays.py)"
            echo ""
            echo "Environment variables:"
            echo "  HPXPY_IMAGE_NAME  Docker image name (default: hpxpy)"
            echo "  HPX_THREADS       Number of HPX threads to use"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

# Get system info
get_system_info() {
    echo "{"
    echo "  \"hostname\": \"$(hostname)\","
    echo "  \"date\": \"$(date -Iseconds)\","
    echo "  \"cpu\": \"$(uname -p)\","
    echo "  \"os\": \"$(uname -s) $(uname -r)\","
    if command -v nproc &> /dev/null; then
        echo "  \"cores\": $(nproc),"
    else
        echo "  \"cores\": $(sysctl -n hw.ncpu 2>/dev/null || echo 0),"
    fi
    if command -v free &> /dev/null; then
        echo "  \"memory_gb\": $(free -g | awk '/^Mem:/{print $2}')"
    else
        echo "  \"memory_gb\": $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))"
    fi
    echo "}"
}

# Run benchmarks
run_benchmarks() {
    log_header "Running HPXPy Benchmarks"

    if $USE_DOCKER; then
        log_info "Running in Docker container: ${IMAGE_NAME}"
        docker run --rm \
            -v "$PROJECT_DIR:/opt/hpxpy" \
            -w /opt/hpxpy \
            "${IMAGE_NAME}:latest" \
            python3 "$BENCHMARK_FILE"
    else
        log_info "Running locally..."
        source .venv/bin/activate 2>/dev/null || true
        source build/setup_env.sh 2>/dev/null || true
        PYTHONPATH=build:. python3 "$BENCHMARK_FILE"
    fi
}

# Save results
save_results() {
    mkdir -p "$RESULTS_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.json"

    log_info "Saving system info to: $RESULT_FILE"

    get_system_info > "$RESULT_FILE"

    log_info "Results saved to: $RESULT_FILE"
    log_info "View all results: ls -la $RESULTS_DIR/"
}

# Main
log_header "System Information"
get_system_info

run_benchmarks

if $SAVE_RESULTS; then
    save_results
fi

log_info "Benchmarks complete!"

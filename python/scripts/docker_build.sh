#!/bin/bash
# HPXPy Docker Build Script
#
# SPDX-License-Identifier: BSL-1.0
#
# Build and optionally push the HPXPy Docker image.
#
# Usage:
#   ./scripts/docker_build.sh              # Build only
#   ./scripts/docker_build.sh --push       # Build and push to Docker Hub
#   ./scripts/docker_build.sh --ecr        # Build and push to AWS ECR
#   ./scripts/docker_build.sh --gcr        # Build and push to Google GCR

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
IMAGE_NAME="${HPXPY_IMAGE_NAME:-hpxpy}"
IMAGE_TAG="${HPXPY_IMAGE_TAG:-latest}"
HPX_VERSION="${HPX_VERSION:-1.10.0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
PUSH_DOCKERHUB=false
PUSH_ECR=false
PUSH_GCR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_DOCKERHUB=true
            shift
            ;;
        --ecr)
            PUSH_ECR=true
            shift
            ;;
        --gcr)
            PUSH_GCR=true
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --push      Push to Docker Hub"
            echo "  --ecr       Push to AWS ECR"
            echo "  --gcr       Push to Google GCR"
            echo "  --tag TAG   Set image tag (default: latest)"
            echo ""
            echo "Environment variables:"
            echo "  HPXPY_IMAGE_NAME    Image name (default: hpxpy)"
            echo "  HPXPY_IMAGE_TAG     Image tag (default: latest)"
            echo "  HPX_VERSION         HPX version to build (default: 1.10.0)"
            echo "  AWS_ACCOUNT_ID      AWS account ID for ECR"
            echo "  AWS_REGION          AWS region for ECR (default: us-east-1)"
            echo "  GCP_PROJECT_ID      Google Cloud project ID for GCR"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

# Build the image
log_info "Building HPXPy Docker image..."
log_info "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
log_info "  HPX version: ${HPX_VERSION}"

docker build \
    --build-arg HPX_VERSION="${HPX_VERSION}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile \
    .

log_info "Build complete!"

# Run tests
log_info "Running tests in container..."
docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" pytest tests/unit/ -v --tb=short || {
    log_error "Tests failed!"
    exit 1
}
log_info "Tests passed!"

# Push to Docker Hub
if $PUSH_DOCKERHUB; then
    log_info "Pushing to Docker Hub..."
    docker push "${IMAGE_NAME}:${IMAGE_TAG}"
    log_info "Pushed to Docker Hub: ${IMAGE_NAME}:${IMAGE_TAG}"
fi

# Push to AWS ECR
if $PUSH_ECR; then
    AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID must be set}"
    AWS_REGION="${AWS_REGION:-us-east-1}"
    ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"

    log_info "Authenticating with AWS ECR..."
    aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    log_info "Tagging for ECR..."
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${ECR_REPO}:${IMAGE_TAG}"

    log_info "Pushing to ECR..."
    docker push "${ECR_REPO}:${IMAGE_TAG}"
    log_info "Pushed to ECR: ${ECR_REPO}:${IMAGE_TAG}"
fi

# Push to Google GCR
if $PUSH_GCR; then
    GCP_PROJECT_ID="${GCP_PROJECT_ID:?GCP_PROJECT_ID must be set}"
    GCR_REPO="gcr.io/${GCP_PROJECT_ID}/${IMAGE_NAME}"

    log_info "Authenticating with GCR..."
    gcloud auth configure-docker --quiet

    log_info "Tagging for GCR..."
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${GCR_REPO}:${IMAGE_TAG}"

    log_info "Pushing to GCR..."
    docker push "${GCR_REPO}:${IMAGE_TAG}"
    log_info "Pushed to GCR: ${GCR_REPO}:${IMAGE_TAG}"
fi

log_info "Done!"

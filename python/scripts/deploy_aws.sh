#!/bin/bash
# HPXPy AWS Deployment Script
#
# SPDX-License-Identifier: BSL-1.0
#
# Deploy HPXPy to AWS ECS for cloud testing.
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - Docker installed locally
#   - ECR repository created
#
# Usage:
#   ./scripts/deploy_aws.sh setup      # Create ECS infrastructure
#   ./scripts/deploy_aws.sh deploy     # Build and deploy
#   ./scripts/deploy_aws.sh run        # Run benchmark task
#   ./scripts/deploy_aws.sh cleanup    # Remove all resources

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
CLUSTER_NAME="${CLUSTER_NAME:-hpxpy-cluster}"
TASK_FAMILY="${TASK_FAMILY:-hpxpy-benchmark}"
SERVICE_NAME="${SERVICE_NAME:-hpxpy-service}"
IMAGE_NAME="${HPXPY_IMAGE_NAME:-hpxpy}"
AWS_REGION="${AWS_REGION:-us-east-1}"

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

# Get AWS account ID
get_account_id() {
    aws sts get-caller-identity --query Account --output text
}

# Setup ECS infrastructure
setup_infrastructure() {
    log_header "Setting up AWS Infrastructure"

    ACCOUNT_ID=$(get_account_id)
    ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"

    # Create ECR repository
    log_info "Creating ECR repository..."
    aws ecr create-repository \
        --repository-name "${IMAGE_NAME}" \
        --region "${AWS_REGION}" \
        2>/dev/null || log_warn "ECR repository may already exist"

    # Create ECS cluster
    log_info "Creating ECS cluster..."
    aws ecs create-cluster \
        --cluster-name "${CLUSTER_NAME}" \
        --region "${AWS_REGION}" \
        2>/dev/null || log_warn "ECS cluster may already exist"

    # Create task execution role
    log_info "Creating IAM role..."
    cat > /tmp/trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name ecsTaskExecutionRole \
        --assume-role-policy-document file:///tmp/trust-policy.json \
        2>/dev/null || log_warn "IAM role may already exist"

    aws iam attach-role-policy \
        --role-name ecsTaskExecutionRole \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
        2>/dev/null || true

    log_info "Infrastructure setup complete!"
    log_info "ECR Repository: ${ECR_REPO}"
    log_info "ECS Cluster: ${CLUSTER_NAME}"
}

# Build and push Docker image
deploy_image() {
    log_header "Building and Deploying Docker Image"

    ACCOUNT_ID=$(get_account_id)
    ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"

    cd "$PROJECT_DIR"

    # Build image
    log_info "Building Docker image..."
    docker build -t "${IMAGE_NAME}:latest" -f Dockerfile .

    # Authenticate with ECR
    log_info "Authenticating with ECR..."
    aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${ECR_REPO%/*}"

    # Tag and push
    log_info "Pushing to ECR..."
    docker tag "${IMAGE_NAME}:latest" "${ECR_REPO}:latest"
    docker push "${ECR_REPO}:latest"

    log_info "Image deployed: ${ECR_REPO}:latest"

    # Register task definition
    log_info "Registering task definition..."

    cat > /tmp/task-definition.json << EOF
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "hpxpy",
      "image": "${ECR_REPO}:latest",
      "essential": true,
      "command": ["python3", "benchmarks/benchmark_arrays.py"],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${TASK_FAMILY}",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs",
          "awslogs-create-group": "true"
        }
      }
    }
  ]
}
EOF

    aws ecs register-task-definition \
        --cli-input-json file:///tmp/task-definition.json \
        --region "${AWS_REGION}"

    log_info "Deployment complete!"
}

# Run benchmark task
run_benchmark_task() {
    log_header "Running Benchmark Task"

    # Get default VPC and subnet
    VPC_ID=$(aws ec2 describe-vpcs \
        --filters "Name=isDefault,Values=true" \
        --query "Vpcs[0].VpcId" \
        --output text \
        --region "${AWS_REGION}")

    SUBNET_ID=$(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=${VPC_ID}" \
        --query "Subnets[0].SubnetId" \
        --output text \
        --region "${AWS_REGION}")

    SG_ID=$(aws ec2 describe-security-groups \
        --filters "Name=vpc-id,Values=${VPC_ID}" "Name=group-name,Values=default" \
        --query "SecurityGroups[0].GroupId" \
        --output text \
        --region "${AWS_REGION}")

    log_info "Running task in VPC: ${VPC_ID}"

    TASK_ARN=$(aws ecs run-task \
        --cluster "${CLUSTER_NAME}" \
        --task-definition "${TASK_FAMILY}" \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_ID}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}" \
        --region "${AWS_REGION}" \
        --query "tasks[0].taskArn" \
        --output text)

    log_info "Task started: ${TASK_ARN}"
    log_info "View logs: aws logs tail /ecs/${TASK_FAMILY} --follow"

    # Wait for task to complete
    log_info "Waiting for task to complete..."
    aws ecs wait tasks-stopped \
        --cluster "${CLUSTER_NAME}" \
        --tasks "${TASK_ARN}" \
        --region "${AWS_REGION}"

    log_info "Task completed!"

    # Get exit code
    EXIT_CODE=$(aws ecs describe-tasks \
        --cluster "${CLUSTER_NAME}" \
        --tasks "${TASK_ARN}" \
        --region "${AWS_REGION}" \
        --query "tasks[0].containers[0].exitCode" \
        --output text)

    if [ "$EXIT_CODE" = "0" ]; then
        log_info "Benchmark succeeded!"
    else
        log_error "Benchmark failed with exit code: ${EXIT_CODE}"
    fi

    # Show logs
    log_header "Task Logs"
    aws logs tail "/ecs/${TASK_FAMILY}" \
        --region "${AWS_REGION}" \
        --since 10m
}

# Cleanup all resources
cleanup_resources() {
    log_header "Cleaning up AWS Resources"

    ACCOUNT_ID=$(get_account_id)

    log_warn "This will delete all HPXPy AWS resources!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled."
        exit 0
    fi

    # Delete ECS cluster
    log_info "Deleting ECS cluster..."
    aws ecs delete-cluster \
        --cluster "${CLUSTER_NAME}" \
        --region "${AWS_REGION}" \
        2>/dev/null || true

    # Delete ECR repository
    log_info "Deleting ECR repository..."
    aws ecr delete-repository \
        --repository-name "${IMAGE_NAME}" \
        --region "${AWS_REGION}" \
        --force \
        2>/dev/null || true

    # Delete CloudWatch log group
    log_info "Deleting log group..."
    aws logs delete-log-group \
        --log-group-name "/ecs/${TASK_FAMILY}" \
        --region "${AWS_REGION}" \
        2>/dev/null || true

    log_info "Cleanup complete!"
}

# Main
case "${1:-help}" in
    setup)
        setup_infrastructure
        ;;
    deploy)
        deploy_image
        ;;
    run)
        run_benchmark_task
        ;;
    cleanup)
        cleanup_resources
        ;;
    help|--help|-h)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  setup    Create ECS infrastructure (cluster, ECR repo, IAM role)"
        echo "  deploy   Build Docker image and push to ECR"
        echo "  run      Run benchmark task on ECS Fargate"
        echo "  cleanup  Delete all AWS resources"
        echo ""
        echo "Environment variables:"
        echo "  AWS_REGION     AWS region (default: us-east-1)"
        echo "  CLUSTER_NAME   ECS cluster name (default: hpxpy-cluster)"
        echo "  TASK_FAMILY    Task definition family (default: hpxpy-benchmark)"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Run '$0 help' for usage."
        exit 1
        ;;
esac

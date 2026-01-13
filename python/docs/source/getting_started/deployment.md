# Cloud Deployment

This guide covers deploying HPXPy to cloud environments for testing and production use.

## Docker

HPXPy provides a multi-stage Dockerfile for reproducible builds.

### Building the Image

```bash
cd python
docker build -t hpxpy .
```

### Running Tests

```bash
docker run hpxpy
# Or explicitly:
docker run hpxpy pytest tests/unit/ -v
```

### Interactive Python

```bash
docker run -it hpxpy python3
>>> import hpxpy as hpx
>>> hpx.init()
>>> arr = hpx.arange(100)
>>> print(hpx.sum(arr))
>>> hpx.finalize()
```

### Running Jupyter Notebooks

```bash
# Start Jupyter server in Docker
docker run -it -p 8888:8888 hpxpy bash -c "
    pip install jupyter &&
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
"
```

Then open `http://localhost:8888` in your browser.

## Scripts

HPXPy includes helper scripts for common deployment tasks:

### Docker Build Script

```bash
# Build only
./scripts/docker_build.sh

# Build and push to Docker Hub
./scripts/docker_build.sh --push

# Build and push to AWS ECR
AWS_ACCOUNT_ID=123456789 AWS_REGION=us-east-1 ./scripts/docker_build.sh --ecr

# Build and push to Google GCR
GCP_PROJECT_ID=my-project ./scripts/docker_build.sh --gcr
```

### Notebook Runner

```bash
# Start Jupyter server locally
./scripts/run_notebook.sh

# Start Jupyter in Docker
./scripts/run_notebook.sh --docker

# Execute a notebook non-interactively
./scripts/run_notebook.sh --execute tutorials/01_getting_started.ipynb

# Execute in Docker
./scripts/run_notebook.sh --docker --execute tutorials/07_benchmarks.ipynb
```

### Benchmark Runner

```bash
# Run benchmarks locally
./scripts/run_benchmarks.sh

# Run in Docker
./scripts/run_benchmarks.sh --docker

# Save results to JSON
./scripts/run_benchmarks.sh --save
```

## AWS Deployment

### Prerequisites

- AWS CLI configured with appropriate credentials
- Docker installed locally

### Setup Infrastructure

```bash
./scripts/deploy_aws.sh setup
```

This creates:
- ECR repository for Docker images
- ECS cluster for running containers
- IAM role for task execution

### Deploy and Run

```bash
# Build and push Docker image to ECR
./scripts/deploy_aws.sh deploy

# Run benchmark task on ECS Fargate
./scripts/deploy_aws.sh run
```

### View Logs

```bash
aws logs tail /ecs/hpxpy-benchmark --follow
```

### Cleanup

```bash
./scripts/deploy_aws.sh cleanup
```

## Google Cloud Platform

### Cloud Run (Serverless)

```bash
# Build and push to GCR
GCP_PROJECT_ID=my-project ./scripts/docker_build.sh --gcr

# Deploy to Cloud Run
gcloud run deploy hpxpy \
    --image gcr.io/my-project/hpxpy:latest \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --cpu 4
```

### GKE (Kubernetes)

```bash
# Create cluster
gcloud container clusters create hpxpy-cluster \
    --num-nodes=3 \
    --machine-type=n1-standard-4

# Deploy
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hpxpy-benchmark
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hpxpy
  template:
    metadata:
      labels:
        app: hpxpy
    spec:
      containers:
      - name: hpxpy
        image: gcr.io/my-project/hpxpy:latest
        command: ["python3", "benchmarks/benchmark_arrays.py"]
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
EOF
```

## GitHub Actions CI

The repository includes a GitHub Actions workflow for automated testing:

```yaml
# .github/workflows/hpxpy.yml
name: HPXPy CI

on:
  push:
    branches: [master]
    paths:
      - 'python/**'
  pull_request:
    paths:
      - 'python/**'
```

The workflow:
1. Builds HPX from source
2. Builds HPXPy
3. Runs unit tests
4. Runs benchmarks
5. Verifies Docker build

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HPX_NUM_THREADS` | Number of HPX worker threads | Auto-detect |
| `HPXPY_IMAGE_NAME` | Docker image name | `hpxpy` |
| `HPXPY_IMAGE_TAG` | Docker image tag | `latest` |
| `AWS_REGION` | AWS region for deployment | `us-east-1` |
| `GCP_PROJECT_ID` | Google Cloud project ID | Required for GCP |

## Best Practices

### Container Resources

For production workloads, allocate appropriate resources:

```yaml
# Kubernetes example
resources:
  requests:
    cpu: "4"
    memory: "8Gi"
  limits:
    cpu: "8"
    memory: "16Gi"
```

### Multi-Node HPX

For distributed HPX across multiple containers:

```bash
# Master node
docker run -e HPX_LOCALITIES=4 -e HPX_LOCALITY_ID=0 hpxpy python my_distributed_script.py

# Worker nodes
docker run -e HPX_LOCALITIES=4 -e HPX_LOCALITY_ID=1 hpxpy python my_distributed_script.py
# ... etc
```

### Performance Monitoring

Monitor HPX performance using environment variables:

```bash
docker run \
    -e HPX_PRINT_COUNTER='/threadqueue/length' \
    -e HPX_PRINT_COUNTER='/threads/count/cumulative' \
    hpxpy python benchmarks/benchmark_arrays.py
```

## Troubleshooting

### Container Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Python/HPXPy error |
| 137 | OOM killed (increase memory) |
| 139 | Segmentation fault |

### Common Issues

**HPX initialization fails:**
```
RuntimeError: this function can be called from an HPX thread only
```
Solution: Ensure `hpx.init()` is called before any HPX operations.

**Out of memory:**
```
std::bad_alloc
```
Solution: Increase container memory or reduce array sizes.

**Slow performance:**
- Check `HPX_NUM_THREADS` is set appropriately
- Verify container has access to all CPU cores
- Use `--cpus` flag with Docker to allocate cores

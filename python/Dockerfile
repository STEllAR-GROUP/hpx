# HPXPy Docker Image
#
# SPDX-License-Identifier: BSL-1.0
#
# Multi-stage build for HPXPy with HPX runtime
#
# Build: docker build -t hpxpy .
# Run:   docker run -it hpxpy python3
# Test:  docker run hpxpy pytest tests/unit/ -v

ARG HPX_VERSION=1.10.0
ARG UBUNTU_VERSION=22.04

# =============================================================================
# Stage 1: Build HPX
# =============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS hpx-builder

ARG HPX_VERSION
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libhwloc-dev \
    libasio-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone and build HPX
WORKDIR /opt
RUN git clone --depth 1 --branch ${HPX_VERSION} \
    https://github.com/STEllAR-GROUP/hpx.git hpx-src

WORKDIR /opt/hpx-src/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/hpx \
    -DHPX_WITH_FETCH_ASIO=ON \
    -DHPX_WITH_EXAMPLES=OFF \
    -DHPX_WITH_TESTS=OFF \
    -DHPX_WITH_DOCUMENTATION=OFF \
    && make -j$(nproc) \
    && make install

# =============================================================================
# Stage 2: Build HPXPy
# =============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS hpxpy-builder

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libboost-all-dev \
    libhwloc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy HPX installation from builder
COPY --from=hpx-builder /opt/hpx /opt/hpx

# Set HPX environment
ENV HPX_DIR=/opt/hpx/lib/cmake/HPX
ENV LD_LIBRARY_PATH=/opt/hpx/lib:$LD_LIBRARY_PATH

# Copy HPXPy source
WORKDIR /opt/hpxpy
COPY . .

# Create virtual environment and install Python dependencies
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy pybind11 pytest pytest-cov

# Build HPXPy
RUN . .venv/bin/activate && \
    mkdir -p build && cd build && \
    cmake .. -DHPX_DIR=${HPX_DIR} -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# =============================================================================
# Stage 3: Runtime image
# =============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    libboost-filesystem1.74.0 \
    libboost-program-options1.74.0 \
    libboost-system1.74.0 \
    libhwloc15 \
    && rm -rf /var/lib/apt/lists/*

# Copy HPX runtime libraries
COPY --from=hpx-builder /opt/hpx/lib /opt/hpx/lib

# Copy HPXPy with built module
COPY --from=hpxpy-builder /opt/hpxpy /opt/hpxpy

# Set environment
ENV LD_LIBRARY_PATH=/opt/hpx/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/hpxpy/build:/opt/hpxpy
ENV PATH=/opt/hpxpy/.venv/bin:$PATH
ENV VIRTUAL_ENV=/opt/hpxpy/.venv

WORKDIR /opt/hpxpy

# Default command: run tests
CMD ["pytest", "tests/unit/", "-v", "--tb=short"]

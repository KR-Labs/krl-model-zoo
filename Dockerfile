# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

LABEL maintainer="KR-Labs <contact@kr-labs.com>"
LABEL description="KRL Model Zoo Core - Production-ready model orchestration"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml setup.py ./
COPY krl_core/ ./krl_core/
COPY examples/ ./examples/
COPY tests/ ./tests/

# Install package with all dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=krl_core", "--cov-report=term-missing"]

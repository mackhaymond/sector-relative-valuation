# syntax=docker/dockerfile:1

# Build stage
FROM python:3.12-slim AS builder

# Install uv from the official distroless image (smallest footprint).
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app

# System build deps; cleanup in the same layer to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first to leverage Docker cache.
COPY pyproject.toml uv.lock ./

# uv sync writes the venv to /app/.venv by default. --frozen requires the
# lockfile to be up-to-date; --no-dev skips the dev dependency group;
# --no-install-project skips installing this project as a package since
# it runs as a script via `python src/dashboard.py`.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-dev --no-install-project

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

# Copy the resolved venv from the builder. PATH puts the venv's python
# first so `python src/dashboard.py` picks up the installed packages.
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser sector_analysis.csv sector_analysis_full.csv weights.csv ./

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8050

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "src/dashboard.py"]

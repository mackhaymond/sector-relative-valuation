# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies and poetry, then cleanup in the same layer
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --no-ansi --only main

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

# Copy only the installed packages and application files
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
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

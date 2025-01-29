# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies and poetry, then cleanup in the same layer
RUN apt-get update && apt-get install -y \
    gcc \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --no-ansi

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Copy only the installed packages and application files
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY src/ ./src/
COPY sector_analysis.csv sector_analysis_full.csv weights.csv ./

# Expose the port the app runs on
EXPOSE 8050

# Command to run the application
CMD ["python", "src/dashboard.py"]

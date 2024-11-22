FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency files first
COPY pyproject.toml poetry.lock ./

# Install poetry and dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Copy the rest of the application
COPY src/ ./src/
COPY sector_analysis.csv ./

# Expose the port the app runs on
EXPOSE 8050

# Command to run the application
CMD ["python", "src/dashboard.py"]

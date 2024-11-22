# Use python:3.12-slim as the base image
FROM python:3.12-slim

# Install Poetry
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 -

# Set the working directory
WORKDIR /app

# Copy the entire repository into the Docker image
COPY . .

# Install dependencies using Poetry
RUN poetry install

# Expose port 8050
EXPOSE 8050

# Set the default command to run the Dash app
CMD ["poetry", "run", "python", "src/dashboard.py"]

# Default recipe
default: dash

# Download and process data
data:
    python src/data_processing.py

# Generate portfolio weights
weights:
    python src/portfolio_optimization.py

# Run the dashboard
dash:
    python src/dashboard.py

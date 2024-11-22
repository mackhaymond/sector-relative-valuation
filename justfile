# Default recipe
default: dash

# Download data
data:
    python3 src/data.py

# Generate weights
weights:
    python3 src/generate_weights.py

# Run the dashboard
dash:
    python3 src/dashboard.py

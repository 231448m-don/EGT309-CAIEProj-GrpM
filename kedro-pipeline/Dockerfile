FROM python:3.11-slim

# Workdir inside the container
WORKDIR /app

# System packages for builds (xgboost, sklearn, etc.)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install deps
# Make sure you actually have 'requirements.txt' in this folder
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy Kedro project metadata
COPY pyproject.toml ./pyproject.toml
# (optional but nice)
# COPY README.md ./README.md

# Copy Kedro project code and config
COPY src ./src
COPY conf ./conf
COPY data ./data

# Let Python see your package (kedro_pipeline)
ENV PYTHONPATH=/app/src

# Default: run your ML pipeline (change name if needed)
CMD ["kedro", "run", "--pipeline", "ml_pipeline"]

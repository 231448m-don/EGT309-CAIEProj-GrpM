FROM python:3.10-slim

WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y build-essential && apt-get clean

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy ENTIRE Kedro project into container
COPY kedro-pipeline/ .

# The working directory inside container is the Kedro project
WORKDIR /app

# Default command: run Kedro pipeline
CMD ["kedro", "run", "--pipeline", "modeling"]

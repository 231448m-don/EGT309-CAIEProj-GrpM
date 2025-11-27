# Use a lightweight Python image
FROM python:3.12-slim

# Avoid buffered output
ENV PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /app

# Install OS deps if needed (uncomment if xgboost complains)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Ensure run.sh is executable inside the container
RUN chmod +x run.sh

# Default command: run the pipeline
CMD ["bash", "run.sh"]

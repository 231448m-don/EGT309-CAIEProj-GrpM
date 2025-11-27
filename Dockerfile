FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy *your Git repo contents* into the image
COPY . .

RUN chmod +x run.sh

CMD ["bash", "run.sh"]

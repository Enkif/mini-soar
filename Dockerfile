FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🟢 Add a non-root user so artifacts aren’t owned by root
RUN useradd -m appuser
USER appuser

EXPOSE 8501

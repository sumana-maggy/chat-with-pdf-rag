FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ libffi-dev libssl-dev curl && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Model ready')"

COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}

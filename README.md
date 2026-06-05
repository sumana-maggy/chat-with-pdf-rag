# RAG Chat with PDF + RAGAS Evaluation

> **Production-grade Retrieval-Augmented Generation pipeline with automated self-evaluation.**  
> Built by [Sumana Yerremsetti](https://github.com/sumana-maggy) — AI Engineer at Deloitte USI

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange?style=flat-square)](https://trychroma.com)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-blue?style=flat-square&logo=google-gemini)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**[🚀 Live Demo](https://chat-with-pdf-rag.onrender.com)** &nbsp;·&nbsp; **[📂 GitHub](https://github.com/sumana-maggy/chat-with-pdf-rag)**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER UPLOADS PDF                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  PyMuPDF    │  Extract text page by page
                    │  Parser     │
                    └──────┬──────┘
                           │  raw text per page
                    ┌──────▼──────┐
                    │   Sliding   │  chunk_size=500
                    │   Window    │  overlap=50
                    │   Chunker   │  → N chunks with page metadata
                    └──────┬──────┘
                           │  list of chunks
                    ┌──────▼──────┐
                    │  MiniLM-    │  all-MiniLM-L6-v2
                    │  L6-v2      │  384-dimensional vectors
                    │  Embedder   │  normalize_embeddings=True
                    └──────┬──────┘
                           │  embeddings
                    ┌──────▼──────┐
                    │  ChromaDB   │  cosine similarity index
                    │  Vector     │  per-session collection
                    │  Store      │  HNSW index
                    └─────────────┘

                    ── AT QUERY TIME ──

User Question ──► Embed Query (MiniLM) ──► Cosine Similarity Search
                                                      │
                                              top-K chunks
                                                      │
                                    ┌─────────────────▼──────────────────┐
                                    │         Gemini 1.5 Flash           │
                                    │         (Streaming SSE)            │
                                    │  system: retrieved chunks as ctx   │
                                    │  → token-by-token streaming answer │
                                    └─────────────────┬──────────────────┘
                                                      │ full answer
                                    ┌─────────────────▼──────────────────┐
                                    │       RAGAS Evaluation             │
                                    │       (Gemini 1.5 Flash)           │
                                    │                                    │
                                    │  ① Faithfulness       0.0–1.0     │
                                    │  ② Answer Relevance   0.0–1.0     │
                                    │  ③ Context Precision  0.0–1.0     │
                                    │  ④ Context Recall     0.0–1.0     │
                                    │  ─────────────────────────────     │
                                    │     Overall RAG Score  0.0–1.0    │
                                    └────────────────────────────────────┘
```

---

## RAGAS Metrics Explained

| Metric | What it measures | Formula idea |
|---|---|---|
| **Faithfulness** | Are all claims in the answer grounded in the retrieved context? Detects hallucination. | claims supported by context / total claims |
| **Answer Relevance** | Does the answer directly address the question asked? | semantic similarity of answer to question |
| **Context Precision** | What fraction of retrieved chunks were actually useful? | useful chunks / total retrieved chunks |
| **Context Recall** | Does the context contain all information needed to answer fully? | info covered by context / info in answer |

All metrics scored 0.0–1.0. **Gemini 1.5 Flash** is used as the judge — fast and free/cheap for evaluation.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI + uvicorn (async, production-grade) |
| **PDF Parsing** | PyMuPDF (fast, accurate text extraction) |
| **Chunking** | Sliding window with configurable size + overlap |
| **Embeddings** | `all-MiniLM-L6-v2` via sentence-transformers (384-dim) |
| **Vector Store** | ChromaDB (in-memory, cosine similarity, HNSW index) |
| **Generation** | Google Gemini 1.5 Flash (streaming SSE) |
| **Evaluation** | Gemini 1.5 Flash as RAGAS judge |
| **Frontend** | Vanilla HTML/CSS/JS (no framework, fast load) |
| **Deployment** | Docker → Render |

---

## Running Locally

### Prerequisites
- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/sumana-maggy/chat-with-pdf-rag.git
cd chat-with-pdf-rag

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run the backend
cd backend
export GEMINI_API_KEY="your-api-key-here"
uvicorn main:app --reload --port 8000

# 4. Open the frontend
# Go to http://localhost:8000
# OR open frontend/index.html directly in browser
```

### With Docker

```bash
docker build -t chat-with-pdf-rag .
docker run -p 8000:8000 chat-with-pdf-rag
# Visit http://localhost:8000
```

---

## Deploy to Render (Free)

1. Fork this repo
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. Done! Your app is live at `https://chat-with-pdf-rag.onrender.com`

> **Note:** Free Render tier spins down after inactivity. First request after sleep takes ~30s (model reloads). Upgrade to Starter ($7/mo) for always-on.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health + model status |
| `POST` | `/upload` | Upload PDF → returns `session_id` |
| `POST` | `/query` | Ask question → streaming SSE response |

### Upload Example
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf" \
  -F "chunk_size=500" \
  -F "chunk_overlap=50"
```

### Query Example
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "question": "What is the main conclusion?",
    "top_k": 4,
    "min_score": 0.2
  }'
```

### SSE Event Types
```
data: {"type": "chunks", "chunks": [...]}       ← retrieved context
data: {"type": "token", "token": "Hello"}       ← streaming answer
data: {"type": "answer_done", "answer": "..."}  ← full answer
data: {"type": "ragas_start"}                   ← evaluation beginning
data: {"type": "ragas_done", "scores": {...}}   ← RAGAS results
data: [DONE]
```

---

## Project Structure

```
chat-with-pdf-rag/
├── backend/
│   ├── main.py              ← FastAPI app, routes, SSE streaming
│   ├── requirements.txt
│   └── rag/
│       ├── __init__.py
│       ├── chunker.py       ← PyMuPDF parser + sliding window chunker
│       ├── embedder.py      ← sentence-transformers wrapper
│       ├── vector_store.py  ← ChromaDB per-session collection
│       ├── retriever.py     ← cosine similarity retrieval
│       ├── generator.py     ← Claude streaming answer generation
│       └── ragas_eval.py    ← 4-metric RAGAS evaluation via Claude
├── frontend/
│   └── index.html           ← Single-page app, streaming SSE client
├── Dockerfile               ← Production Docker image
├── render.yaml              ← One-click Render deployment
├── .gitignore
└── README.md
```

---

## Key Design Decisions

**Why in-memory ChromaDB?**  
Sessions are per-upload and ephemeral by design — no need for persistence. Keeps deployment simple (no external DB) while still being a proper vector store with HNSW indexing.

**Why MiniLM-L6-v2?**  
Best speed/quality tradeoff for document retrieval at 384 dims. Runs on CPU in ~50ms per chunk. Model is baked into the Docker image so no download on first request.

**Why Gemini 1.5 Flash for RAGAS?**  
Gemini 1.5 Flash is highly efficient, provides structured JSON output, and offers a generous free tier for developers. It handles both generation and evaluation at high speed.

**Why streaming SSE?**  
Improves perceived latency dramatically. Users see the answer forming in real time rather than waiting for the full response.

---

## Author

**Yerremsetti Sumana** — AI Engineer at Deloitte USI  
[GitHub](https://github.com/sumana-maggy) · [LinkedIn](https://linkedin.com/in/sumana-yerremsetti-534921169) · [Portfolio](https://sumana-maggy.github.io)

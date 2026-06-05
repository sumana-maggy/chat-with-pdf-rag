import os, uuid, json, asyncio
os.environ["CHROMA_TELEMETRY_OFF"] = "True"
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from rag.chunker import chunk_pdf_bytes
from rag.embedder import get_embedder
from rag.vector_store import VectorStore
from rag.retriever import retrieve_chunks
from rag.generator import stream_answer
from rag.ragas_eval import evaluate_ragas

# ── In-memory session store: session_id → VectorStore ──
sessions: dict[str, VectorStore] = {}
embedder = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    print("Loading embedding model...")
    embedder = get_embedder()
    print("Embedding model ready.")
    yield
    sessions.clear()

app = FastAPI(title="RAG PDF API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def root():
    idx = os.path.join(frontend_path, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"status": "RAG PDF API running"}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "all-MiniLM-L6-v2", "sessions": len(sessions)}

# ── UPLOAD PDF ──
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 20MB).")

    # Chunk the PDF
    chunks = chunk_pdf_bytes(pdf_bytes, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise HTTPException(400, "Could not extract text from PDF.")

    # Embed + store in ChromaDB
    session_id = str(uuid.uuid4())
    vs = VectorStore(session_id=session_id, embedder=embedder)
    vs.add_chunks(chunks)
    sessions[session_id] = vs

    return {
        "session_id": session_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "pages": max(c["page"] for c in chunks),
        "dim": 384,
    }

# ── QUERY — STREAMING ──
class QueryRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 4
    min_score: float = 0.2
    chat_history: list = []

@app.post("/query")
async def query(req: QueryRequest):
    vs = sessions.get(req.session_id)
    if not vs:
        raise HTTPException(404, "Session not found. Please re-upload your PDF.")

    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        raise HTTPException(500, "Gemini API key not configured on server.")

    # Retrieve
    retrieved = retrieve_chunks(
        vs=vs,
        embedder=embedder,
        question=req.question,
        top_k=req.top_k,
        min_score=req.min_score,
    )

    if not retrieved:
        retrieved = retrieve_chunks(vs=vs, embedder=embedder, question=req.question, top_k=req.top_k, min_score=0.0)

    async def event_stream():
        # Send retrieved chunks first
        yield f"data: {json.dumps({'type': 'chunks', 'chunks': retrieved})}\n\n"

        # Stream the answer
        full_answer = ""
        async for token in stream_answer(
            question=req.question,
            retrieved=retrieved,
            chat_history=req.chat_history,
            api_key=GEMINI_API_KEY,
        ):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        yield f"data: {json.dumps({'type': 'answer_done', 'answer': full_answer})}\n\n"

        # Run RAGAS evaluation
        yield f"data: {json.dumps({'type': 'ragas_start'})}\n\n"
        try:
            ragas_scores = await evaluate_ragas(
                question=req.question,
                answer=full_answer,
                retrieved=retrieved,
                api_key=GEMINI_API_KEY,
            )
            yield f"data: {json.dumps({'type': 'ragas_done', 'scores': ragas_scores})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'ragas_error', 'error': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

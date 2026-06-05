import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rag.embedder import embed_texts

class VectorStore:
    """
    Per-session ChromaDB collection.
    Uses in-memory Chroma so no disk setup needed on Render.
    """

    def __init__(self, session_id: str, embedder: SentenceTransformer):
        self.session_id = session_id
        self.embedder = embedder
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.create_collection(
            name=f"session_{session_id[:8]}",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict]):
        """Embed and store chunks in ChromaDB."""
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(self.embedder, texts)

        self.collection.add(
            ids=[str(c["chunkId"]) for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"page": c["page"], "chunkId": c["chunkId"]} for c in chunks],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Return top_k most similar chunks with scores."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance → similarity: score = 1 - distance
            score = round(1.0 - distance, 4)
            meta = results["metadatas"][0][i]
            chunks.append({
                "chunkId": meta["chunkId"],
                "text": doc,
                "page": meta["page"],
                "score": score,
            })
        return chunks

    def count(self) -> int:
        return self.collection.count()

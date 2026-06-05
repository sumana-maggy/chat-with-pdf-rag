from sentence_transformers import SentenceTransformer
from rag.embedder import embed_query
from rag.vector_store import VectorStore

def retrieve_chunks(
    vs: VectorStore,
    embedder: SentenceTransformer,
    question: str,
    top_k: int = 4,
    min_score: float = 0.2,
) -> list[dict]:
    """
    Embed the query, retrieve top_k chunks from ChromaDB,
    filter by min_score, return sorted by score descending.
    """
    q_vec = embed_query(embedder, question)
    results = vs.query(query_embedding=q_vec, top_k=top_k)
    filtered = [c for c in results if c["score"] >= min_score]
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered

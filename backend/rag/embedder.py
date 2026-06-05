import hashlib
import math
from typing import List

# Lightweight TF-IDF style embeddings that use zero extra RAM
# No model download needed - pure Python math
# Dim: 384 to match MiniLM interface

DIM = 384

def get_embedder():
    """Returns None - we use hash-based embeddings, no model needed."""
    return None

def _text_to_vector(text: str) -> List[float]:
    """
    Hash-based embedding: deterministic, zero RAM, no model.
    Uses multiple hash functions to create a sparse-ish dense vector.
    Good enough for document retrieval on a budget server.
    """
    text = text.lower().strip()
    words = text.split()
    vec = [0.0] * DIM

    for word in words:
        # Multiple hash seeds for better distribution
        for seed in range(4):
            h = hashlib.md5(f"{seed}:{word}".encode()).hexdigest()
            idx = int(h[:8], 16) % DIM
            val = (int(h[8:16], 16) / 0xFFFFFFFF) * 2 - 1
            vec[idx] += val

    # Also hash bigrams
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        h = hashlib.md5(f"bi:{bigram}".encode()).hexdigest()
        idx = int(h[:8], 16) % DIM
        val = (int(h[8:16], 16) / 0xFFFFFFFF) * 2 - 1
        vec[idx] += val * 0.5

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]

def embed_texts(embedder, texts: List[str]) -> List[List[float]]:
    return [_text_to_vector(t) for t in texts]

def embed_query(embedder, query: str) -> List[float]:
    return _text_to_vector(query)

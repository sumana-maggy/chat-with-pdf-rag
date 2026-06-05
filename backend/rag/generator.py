import anthropic
from typing import AsyncGenerator

async def stream_answer(
    question: str,
    retrieved: list[dict],
    chat_history: list[dict],
    api_key: str,
) -> AsyncGenerator[str, None]:
    """
    Stream Claude's answer token-by-token using retrieved chunks as context.
    """
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} | Page {c['page']} | Similarity: {c['score']:.3f}]\n{c['text']}"
        for i, c in enumerate(retrieved)
    )

    system_prompt = f"""You are a precise document QA assistant using Retrieval-Augmented Generation (RAG).

You are given the most semantically relevant chunks from a PDF document, retrieved via cosine similarity.

Rules:
- Answer ONLY from the provided context chunks.
- Cite page numbers inline e.g. (Page 3).
- If the answer is not in the chunks, say so clearly — do not hallucinate.
- Be concise, accurate, and helpful.

RETRIEVED CONTEXT:
{context}"""

    messages = [*chat_history[-6:], {"role": "user", "content": question}]

    client = anthropic.Anthropic(api_key=api_key)

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text

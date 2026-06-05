import google.generativeai as genai
from typing import AsyncGenerator

async def stream_answer(
    question: str,
    retrieved: list[dict],
    chat_history: list[dict],
    api_key: str,
) -> AsyncGenerator[str, None]:
    genai.configure(api_key=api_key)
    
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} | Page {c['page']} | Similarity: {c['score']:.3f}]\n{c['text']}"
        for i, c in enumerate(retrieved)
    )

    system_instruction = f"""You are a precise document QA assistant using RAG.
Answer ONLY from the provided context chunks. Cite page numbers inline e.g. (Page 3).
If the answer is not in the chunks, say so clearly.

RETRIEVED CONTEXT:
{context}"""

    # Gemini 1.5 Flash is fast and has a large context window
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_instruction
    )

    # Convert chat history to Gemini format: {"role": "user"/"model", "parts": ["text"]}
    history = []
    for msg in chat_history[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)
    
    # We use a synchronous generator wrapper since google-generativeai's stream is sync
    # but we are in an async context. For FastAPI SSE, this works fine if we iterate.
    response = chat.send_message(question, stream=True)
    
    for chunk in response:
        if chunk.text:
            yield chunk.text

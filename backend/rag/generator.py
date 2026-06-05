import json
import httpx
from typing import AsyncGenerator

async def stream_answer(
    question: str,
    retrieved: list[dict],
    chat_history: list[dict],
    api_key: str,
) -> AsyncGenerator[str, None]:
    # Use v1beta and explicit alt=sse for better streaming compatibility
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:streamGenerateContent?alt=sse&key={api_key}"
    
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} | Page {c['page']} | Similarity: {c['score']:.3f}]\n{c['text']}"
        for i, c in enumerate(retrieved)
    )

    system_instruction = f"""You are a precise document QA assistant using RAG.
Answer ONLY from the provided context chunks. Cite page numbers inline e.g. (Page 3).
If the answer is not in the chunks, say so clearly.

RETRIEVED CONTEXT:
{context}"""

    contents = []
    for msg in chat_history[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    
    contents.append({"role": "user", "parts": [{"text": question}]})

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "topK": 40
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("POST", url, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    yield f"⚠️ API Error {response.status_code}: {err_body.decode()}"
                    return
                
                async for line in response.aiter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        raw_json = line[6:].strip()
                        try:
                            chunk = json.loads(raw_json)
                            if "candidates" in chunk:
                                parts = chunk["candidates"][0].get("content", {}).get("parts", [])
                                if parts:
                                    yield parts[0]["text"]
                        except:
                            continue
        except Exception as e:
            yield f"⚠️ Connection Error: {str(e)}"

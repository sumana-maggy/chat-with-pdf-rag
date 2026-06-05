import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import AsyncGenerator

async def stream_answer(
    question: str,
    retrieved: list[dict],
    chat_history: list[dict],
    api_key: str,
) -> AsyncGenerator[str, None]:
    # Use 'rest' transport for stability on cloud platforms like Render
    genai.configure(api_key=api_key, transport='rest')
    
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} | Page {c['page']} | Similarity: {c['score']:.3f}]\n{c['text']}"
        for i, c in enumerate(retrieved)
    )

    system_instruction = f"""You are a precise document QA assistant using RAG.
Answer ONLY from the provided context chunks. Cite page numbers inline e.g. (Page 3).
If the answer is not in the chunks, say so clearly.

RETRIEVED CONTEXT:
{context}"""

    # Create model. We avoid system_instruction in constructor for max compatibility with REST transport
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # Build history with explicit part structure and prepend system instruction
    history = [
        {"role": "user", "parts": [{"text": f"SYSTEM INSTRUCTION: {system_instruction}\n\nPlease acknowledge and wait for my question."}]},
        {"role": "model", "parts": [{"text": "Understood. I am ready to answer questions based only on the provided document context."}]}
    ]

    for msg in chat_history[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        # Ensure parts is a list of dicts with 'text' key
        history.append({"role": role, "parts": [{"text": msg["content"]}]})

    chat = model.start_chat(history=history)
    
    # Disable safety filters to prevent '400 BadRequest' when content is falsely flagged
    safety = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    try:
        response = await chat.send_message_async(
            question, 
            stream=True,
            safety_settings=safety
        )
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        print(f"Streaming error: {e}")
        yield f"⚠️ Error: {str(e)}"

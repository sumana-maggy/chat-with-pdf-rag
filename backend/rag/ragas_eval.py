import json
import httpx

async def evaluate_ragas(
    question: str,
    answer: str,
    retrieved: list[dict],
    api_key: str,
) -> dict:
    # Use the model alias verified by the user
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
    
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} | Page {c['page']} | Similarity: {c['score']:.3f}]\n{c['text']}"
        for i, c in enumerate(retrieved)
    )

    prompt = f"""You are a RAG evaluation expert. Score the following RAG output on 4 RAGAS metrics. Return ONLY valid JSON, no markdown.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER: {answer}

Return exactly this JSON structure:
{{
  "faithfulness": {{"score": 0.0, "reason": "one sentence"}},
  "answer_relevance": {{"score": 0.0, "reason": "one sentence"}},
  "context_precision": {{"score": 0.0, "reason": "one sentence"}},
  "context_recall": {{"score": 0.0, "reason": "one sentence"}}
}}"""

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0
        }
    }

    # Using X-goog-api-key header
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            if response.status_code != 200:
                raise Exception(f"API Error {response.status_code}: {response.text}")
            
            resp_json = response.json()
            raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
            
            # Clean possible markdown
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            
            scores = json.loads(raw_text)
            for metric in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]:
                if metric not in scores:
                    scores[metric] = {"score": 0.5, "reason": "Could not evaluate."}
                scores[metric]["score"] = max(0.0, min(1.0, float(scores[metric]["score"])))
            
            overall = sum(scores[m]["score"] for m in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]) / 4
            scores["overall"] = round(overall, 4)
            return scores
        except Exception as e:
            print(f"RAGAS evaluation error: {e}")
            return {
                "faithfulness": {"score": 0.5, "reason": "API/Parse error."},
                "answer_relevance": {"score": 0.5, "reason": "API/Parse error."},
                "context_precision": {"score": 0.5, "reason": "API/Parse error."},
                "context_recall": {"score": 0.5, "reason": "API/Parse error."},
                "overall": 0.5,
            }

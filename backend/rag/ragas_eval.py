import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json

async def evaluate_ragas(
    question: str,
    answer: str,
    retrieved: list[dict],
    api_key: str,
) -> dict:
    # Use 'rest' transport for stability
    genai.configure(api_key=api_key, transport='rest')
    
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

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={"response_mime_type": "application/json"}
    )

    safety = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    try:
        # Use asynchronous generation
        response = await model.generate_content_async(
            prompt,
            safety_settings=safety
        )
        
        scores = json.loads(response.text)
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
            "faithfulness": {"score": 0.5, "reason": "Parse error."},
            "answer_relevance": {"score": 0.5, "reason": "Parse error."},
            "context_precision": {"score": 0.5, "reason": "Parse error."},
            "context_recall": {"score": 0.5, "reason": "Parse error."},
            "overall": 0.5,
        }

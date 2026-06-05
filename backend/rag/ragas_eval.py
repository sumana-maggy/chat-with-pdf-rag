import anthropic
import json

async def evaluate_ragas(
    question: str,
    answer: str,
    retrieved: list[dict],
    api_key: str,
) -> dict:
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} | Page {c['page']} | Similarity: {c['score']:.3f}]\n{c['text']}"
        for i, c in enumerate(retrieved)
    )

    prompt = f"""You are a RAG evaluation expert. Score the following RAG output on 4 RAGAS metrics. Return ONLY valid JSON, no markdown.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Return exactly this JSON:
{{
  "faithfulness": {{"score": 0.0, "reason": "one sentence"}},
  "answer_relevance": {{"score": 0.0, "reason": "one sentence"}},
  "context_precision": {{"score": 0.0, "reason": "one sentence"}},
  "context_recall": {{"score": 0.0, "reason": "one sentence"}}
}}"""

    client = anthropic.Anthropic(api_key=api_key)

    # Get first available model dynamically
    models = client.models.list()
    model_id = models.data[0].id

    response = client.messages.create(
        model=model_id,
        max_tokens=512,
        system="You are a RAG evaluation expert. Return only valid JSON, no markdown.",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(clean)
        for metric in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]:
            if metric not in scores:
                scores[metric] = {"score": 0.5, "reason": "Could not evaluate."}
            scores[metric]["score"] = max(0.0, min(1.0, float(scores[metric]["score"])))
        overall = sum(scores[m]["score"] for m in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]) / 4
        scores["overall"] = round(overall, 4)
        return scores
    except Exception as e:
        return {
            "faithfulness": {"score": 0.5, "reason": "Parse error."},
            "answer_relevance": {"score": 0.5, "reason": "Parse error."},
            "context_precision": {"score": 0.5, "reason": "Parse error."},
            "context_recall": {"score": 0.5, "reason": "Parse error."},
            "overall": 0.5,
        }

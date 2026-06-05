import anthropic
import json

async def evaluate_ragas(
    question: str,
    answer: str,
    retrieved: list[dict],
    api_key: str,
) -> dict:
    """
    Evaluate RAG output using 4 RAGAS metrics via Claude as judge.

    Metrics:
    - Faithfulness: Are all claims in the answer grounded in the context?
    - Answer Relevance: Does the answer actually address the question?
    - Context Precision: Are the retrieved chunks relevant to the question?
    - Context Recall: Does the context contain all info needed to answer?

    Returns scores 0.0–1.0 per metric with reasoning.
    """
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} | Page {c['page']} | Similarity: {c['score']:.3f}]\n{c['text']}"
        for i, c in enumerate(retrieved)
    )

    prompt = f"""You are a RAG evaluation expert implementing the RAGAS framework. 
Evaluate the following RAG system output across 4 metrics. Return ONLY valid JSON, no markdown fences.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Score each metric 0.0 to 1.0:

1. FAITHFULNESS (0-1): Are ALL claims in the answer directly supported by the retrieved context? 
   - 1.0 = every claim is grounded in context
   - 0.0 = answer contains hallucinations not in context

2. ANSWER_RELEVANCE (0-1): Does the answer directly and completely address the question asked?
   - 1.0 = perfectly relevant and complete
   - 0.0 = completely off-topic or avoids the question

3. CONTEXT_PRECISION (0-1): What fraction of the retrieved chunks were actually useful for answering?
   - 1.0 = all retrieved chunks were relevant
   - 0.0 = all retrieved chunks were noise

4. CONTEXT_RECALL (0-1): Does the retrieved context contain sufficient information to answer the question fully?
   - 1.0 = context fully covers what's needed
   - 0.0 = critical information missing from context

Return exactly this JSON:
{{
  "faithfulness": {{"score": 0.0, "reason": "one concise sentence"}},
  "answer_relevance": {{"score": 0.0, "reason": "one concise sentence"}},
  "context_precision": {{"score": 0.0, "reason": "one concise sentence"}},
  "context_recall": {{"score": 0.0, "reason": "one concise sentence"}}
}}"""

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # use Haiku for eval — faster + cheaper
        max_tokens=512,
        system="You are a RAG evaluation expert. Return only valid JSON, no markdown, no explanation outside JSON.",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    try:
        # Strip markdown fences if any
        clean = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(clean)

        # Validate and clamp scores
        for metric in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]:
            if metric not in scores:
                scores[metric] = {"score": 0.5, "reason": "Could not evaluate."}
            scores[metric]["score"] = max(0.0, min(1.0, float(scores[metric]["score"])))

        # Compute overall
        overall = sum(scores[m]["score"] for m in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]) / 4
        scores["overall"] = round(overall, 4)

        return scores

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return {
            "faithfulness": {"score": 0.5, "reason": "Parse error during evaluation."},
            "answer_relevance": {"score": 0.5, "reason": "Parse error during evaluation."},
            "context_precision": {"score": 0.5, "reason": "Parse error during evaluation."},
            "context_recall": {"score": 0.5, "reason": "Parse error during evaluation."},
            "overall": 0.5,
            "error": str(e),
        }

import fitz  # PyMuPDF

def chunk_pdf_bytes(
    pdf_bytes: bytes,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Extract text from PDF bytes and split into overlapping chunks.
    Returns list of {chunkId, text, page} dicts.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").replace("\n", " ").strip()
        text = " ".join(text.split())  # normalize whitespace
        if text:
            pages.append({"pageNum": page_num + 1, "text": text})
    doc.close()

    chunks = []
    chunk_id = 0
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 20:
                chunks.append({
                    "chunkId": chunk_id,
                    "text": chunk_text,
                    "page": page["pageNum"],
                })
                chunk_id += 1
            if end == len(text):
                break
            start += chunk_size - chunk_overlap

    return chunks

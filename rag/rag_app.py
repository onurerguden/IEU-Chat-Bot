import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types


# ==================================================
# CONFIG
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

FAISS_DIR = BASE_DIR / "vector_db" / "faiss_store"
INDEX_PATH = FAISS_DIR / "courses.index"
META_PATH = FAISS_DIR / "metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 6


SYSTEM_PROMPT = """
You are an academic assistant answering questions about university curricula.

Rules:
- Use ONLY the provided context.
- Do NOT use external knowledge.
- If the answer is not supported by the context, say:
  "The curriculum data does not support this information."
- Do not speculate or guess.
- Base comparisons strictly on retrieved evidence.
"""

USER_PROMPT_TEMPLATE = """
Context:
{context}

Question:
{question}

Answer (grounded strictly in the context):
"""


# ==================================================
# HELPERS
# ==================================================
def format_context(docs: list[dict]) -> str:
    """
    Formats retrieved chunks into a readable context block
    """
    blocks = []

    for d in docs:
        meta = d["metadata"]
        header = (
            f"[Department: {meta.get('department')} | "
            f"Course: {meta.get('course_code')} | "
            f"Type: {meta.get('type')} | "
            f"Chunk: {meta.get('chunk_type')}]"
        )
        blocks.append(f"{header}\n{d['text']}")

    return "\n\n".join(blocks)


# ==================================================
# RAG ENGINE (FAISS + GEMINI)
# ==================================================
class FAISSRAG:

    def __init__(self):
        # --------------------
        # Embedding model
        # --------------------
        self.embedder = SentenceTransformer(EMBED_MODEL)

        # --------------------
        # FAISS index
        # --------------------
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise RuntimeError(
                "FAISS data not found. Run build_faiss.py first."
            )

        self.index = faiss.read_index(str(INDEX_PATH))

        with open(META_PATH, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # --------------------
        # Gemini (NEW SDK)
        # --------------------
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    # --------------------
    # RETRIEVAL
    # --------------------
    def retrieve(self, query: str) -> list[dict]:
        q_emb = self.embedder.encode(
            query,
            normalize_embeddings=True
        ).astype("float32")

        _, indices = self.index.search(
            np.array([q_emb]), TOP_K
        )

        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.data[idx])

        return results

    # --------------------
    # GENERATION
    # --------------------
    def answer(self, question: str) -> str:
        retrieved = self.retrieve(question)

        if not retrieved:
            return "No relevant curriculum information was found."

        context = format_context(retrieved)

        prompt = f"""{SYSTEM_PROMPT}

{USER_PROMPT_TEMPLATE.format(
    context=context,
    question=question
)}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512
            )
        )

        return response.text.strip()


# ==================================================
# CLI APP
# ==================================================
def main():
    print("IEU Engineering Curriculum Chatbot (FAISS)")
    print("Type 'exit' to quit\n")

    rag = FAISSRAG()

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        ans = rag.answer(q)
        print("\nBot:", ans, "\n")


if __name__ == "__main__":
    main()
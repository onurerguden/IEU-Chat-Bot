import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import ollama  # Ollama kütüphanesini ekledik

# ==================================================
# CONFIG
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

FAISS_DIR = BASE_DIR / "vector_db" / "faiss_store"
INDEX_PATH = FAISS_DIR / "courses.index"
META_PATH = FAISS_DIR / "metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 15  # Daha geniş kapsamlı cevaplar için 15 idealdir

SYSTEM_PROMPT = """
You are an academic assistant for Izmir University of Economics.
- Answer the user's questions based ONLY on the provided context.
- If the question is a greeting (like "Hello", "Hi", "Merhaba"), respond politely as an AI assistant.
- For academic questions, if the information is NOT in the context, say: "The curriculum data does not support this information."
- Do not speculate or hallucinate.
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
    Retrieved chunk'ları okunabilir bir blok haline getirir.
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
# RAG ENGINE (FAISS + OLLAMA)
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
        # Ollama Modeli
        # --------------------
        # Terminalde 'ollama run llama3.1:8b' ile indirdiğin model ismi
        self.model = "llama3.1:8b"

    # --------------------
    # RETRIEVAL (GETİRME)
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
            if idx != -1 and idx < len(self.data):
                results.append(self.data[idx])

        return results

    # --------------------
    # GENERATION (ÜRETME)
    # --------------------
    def answer(self, question: str) -> str:
        # Selamlaşma kontrolü
        if question.lower() in ["hello", "hi", "merhaba"]:
            return "Hello! I am the IEU Engineering Curriculum Assistant. How can I help you today?"

        retrieved = self.retrieve(question)

        if not retrieved:
            return "No relevant curriculum information was found."

        context = format_context(retrieved)

        # Ollama Chat API Çağrısı
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': USER_PROMPT_TEMPLATE.format(
                        context=context,
                        question=question
                    )}
                ],
                options={
                    'temperature': 0.0  # Kesin bilgi için 0.0
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}. Please ensure Ollama app is running."

# ==================================================
# CLI APP
# ==================================================
def main():
    print("IEU Engineering Curriculum Chatbot (FAISS + Ollama)")
    print("Type 'exit' to quit\n")

    try:
        rag = FAISSRAG()
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        ans = rag.answer(q)
        print("\nBot:", ans, "\n")

if __name__ == "__main__":
    main()
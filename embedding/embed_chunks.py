import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "course_chunks.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "course_chunks_embedded.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Chunk file not found: {INPUT_PATH}")

    print("Loading chunks...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Total chunks: {len(chunks)}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embedded_chunks = []

    for chunk, emb in zip(chunks, embeddings):
        embedded_chunks.append({
            "id": chunk["id"],
            "embedding": emb.tolist(),
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=4)

    print(f"\nDONE.")
    print(f"Embedded chunks saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parents[1]

EMBED_PATH = BASE_DIR / "data" / "processed" / "course_chunks_embedded.json"
FAISS_DIR = BASE_DIR / "vector_db" / "faiss_store"
FAISS_INDEX_PATH = FAISS_DIR / "courses.index"
META_PATH = FAISS_DIR / "metadata.json"


def main():
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embedded chunks...")
    with open(EMBED_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Total chunks: {len(chunks)}")

    embeddings = np.array(
        [c["embedding"] for c in chunks],
        dtype="float32"
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (normalized)

    print("Building FAISS index...")
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print("FAISS index saved to:", FAISS_INDEX_PATH)
    print("Metadata saved to:", META_PATH)
    print("Total vectors:", index.ntotal)


if __name__ == "__main__":
    main()
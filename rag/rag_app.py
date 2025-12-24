import os
import json
import faiss
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import ollama

# ==================================================
# CONFIG
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

FAISS_DIR = BASE_DIR / "vector_db" / "faiss_store"
INDEX_PATH = FAISS_DIR / "courses.index"
META_PATH = FAISS_DIR / "metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 30  # İlk aşamada daha fazla al, sonra filtrele

SYSTEM_PROMPT = """
You are an academic assistant for Izmir University of Economics Engineering Faculty.

Key Facts:
- Engineering programs: Software Engineering, Computer Engineering, Industrial Engineering, Electrical and Electronics Engineering
- All programs are 4 years (8 semesters)
- Fall semesters: 1, 3, 5, 7 | Spring semesters: 2, 4, 6, 8
- Each year requires 60 ECTS (30 per semester)

ELECTIVE SYSTEM:
 ELEC 001, ELEC 002, ELEC 003, etc.* (Course codes starting with "ELEC")
   - These are ELECTIVE COURSE SLOTS in the curriculum
   - Type: mandatory (students MUST fill these slots)
   - When asked "How many elective courses?", COUNT THESE
   - Example: ELEC 001, ELEC 002, ELEC 003 → 3 elective courses required

 ELECTIVE OPTIONS* (Type=elective in metadata)
   - Actual courses students can choose from
   - Categorized into pools (technical, general culture, etc.)
   - Examples:
     * Technical Pool: CE 304, CE 315, SE 360 (department courses)
     * General Culture Pool: GEHU 205, GEEC 201 ,
   - Students select from these to fill their ELEC slots

CRITICAL INSTRUCTIONS:
- Answer ONLY based on the provided context chunks
- If context contains the answer, provide it clearly and completely
- If context does NOT contain the answer, say: "The curriculum data does not support this information."
- For greetings, respond politely
- NEVER speculate or make up information
- When listing courses, include course codes and names
- When asked about "year X" or "semester Y", focus on metadata-summary chunks
"""

USER_PROMPT_TEMPLATE = """
Context (Retrieved Course Data):
{context}

User Question:
{question}

Instructions:
1. Read ALL context chunks carefully
2. If the answer exists in the context, provide it with course codes
3. If multiple courses match, list them all
4. If you need to count or aggregate, do so based on the context
5. If information is missing, clearly state it

Answer:
"""


# ==================================================
# QUERY UNDERSTANDING (NEW!)
# ==================================================
class QueryProcessor:
    """
    BEST PRACTICE: Parse queries to extract structured filters
    """

    @staticmethod
    def extract_filters(query: str) -> dict:
        """
        Extract year, semester, type, department from natural language query
        """
        filters = {
            "year": None,
            "semester": None,
            "type": None,
            "department": None
        }

        query_lower = query.lower()

        # Extract year (e.g., "second year", "year 2", "fifth year")
        year_patterns = [
            (r'\b(?:first|1st)\s+year\b', 1),
            (r'\b(?:second|2nd)\s+year\b', 2),
            (r'\b(?:third|3rd)\s+year\b', 3),
            (r'\b(?:fourth|4th)\s+year\b', 4),
            (r'\b(?:fifth|5th)\s+year\b', 5),
            (r'\byear\s+(\d)\b', None),  # "year 2" format
        ]

        for pattern, year_val in year_patterns:
            match = re.search(pattern, query_lower)
            if match:
                filters["year"] = year_val if year_val else int(match.group(1))
                break

        # Extract semester
        if re.search(r'\bfall\b', query_lower):
            filters["semester_name"] = "FALL"
        elif re.search(r'\bspring\b', query_lower):
            filters["semester_name"] = "SPRING"

        # Extract type (mandatory/elective)
        if re.search(r'\b(?:mandatory|compulsory|required|core)\b', query_lower):
            filters["type"] = "mandatory"
        elif re.search(r'\b(?:elective|optional)\b', query_lower):
            filters["type"] = "elective"

        # Extract department
        dept_keywords = {
            "software engineering": "Software Engineering",
            "computer engineering": "Computer Engineering",
            "industrial engineering": "Industrial Engineering",
            "electrical": "Electrical and Electronics Engineering",
            "electronics": "Electrical and Electronics Engineering",
        }

        for keyword, dept_name in dept_keywords.items():
            if keyword in query_lower:
                filters["department"] = dept_name
                break

        return filters

    @staticmethod
    def enrich_query(query: str, filters: dict) -> str:
        """
        BEST PRACTICE: Add metadata keywords to query for better retrieval
        """
        enriched_parts = [query]

        if filters.get("year"):
            enriched_parts.append(f"year {filters['year']}")
        if filters.get("semester_name"):
            enriched_parts.append(filters["semester_name"])
        if filters.get("type"):
            enriched_parts.append(filters["type"])
        if filters.get("department"):
            enriched_parts.append(filters["department"])

        return " ".join(enriched_parts)


# ==================================================
# HELPERS
# ==================================================
def format_context(docs: list[dict], max_chunks: int = 20) -> str:
    """
    BEST PRACTICE: Format with priorities
    - Prioritize curriculum_summary and metadata_summary chunks
    - Limit total chunks to avoid context overflow
    """
    # Separate by chunk type
    summaries = [d for d in docs if d["metadata"].get("chunk_type") in ["curriculum_summary", "metadata_summary"]]
    others = [d for d in docs if d["metadata"].get("chunk_type") not in ["curriculum_summary", "metadata_summary"]]

    # Prioritize summaries (they have aggregated info)
    selected = summaries[:10] + others[:max_chunks - 10]

    blocks = []
    for d in selected:
        meta = d["metadata"]
        header = (
            f"[Dept: {meta.get('department', 'N/A')} | "
            f"Code: {meta.get('course_code', 'N/A')} | "
            f"Year: {meta.get('year', 'N/A')} | "
            f"Type: {meta.get('type', 'N/A')} | "
            f"ChunkType: {meta.get('chunk_type', 'N/A')}]"
        )
        blocks.append(f"{header}\n{d['text']}")

    return "\n\n".join(blocks)


# ==================================================
# HYBRID RAG ENGINE (NEW!)
# ==================================================
class FAISSRAG:

    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.query_processor = QueryProcessor()

        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise RuntimeError(
                "FAISS data not found. Run build_faiss.py first."
            )

        self.index = faiss.read_index(str(INDEX_PATH))

        with open(META_PATH, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.model = "llama3.1:8b"

    # --------------------
    # HYBRID RETRIEVAL (SEMANTIC + METADATA FILTERING)
    # --------------------
    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        BEST PRACTICE: Hybrid search
        1. Extract metadata filters from query
        2. Enrich query with metadata keywords
        3. Semantic search with FAISS
        4. Post-filter by metadata
        5. Re-rank by relevance
        """
        # Step 1: Parse query for filters
        filters = self.query_processor.extract_filters(query)

        # Step 2: Enrich query
        enriched_query = self.query_processor.enrich_query(query, filters)

        # Step 3: Semantic search
        q_emb = self.embedder.encode(
            enriched_query,
            normalize_embeddings=True
        ).astype("float32")

        # Get more candidates for filtering
        scores, indices = self.index.search(
            np.array([q_emb]), top_k * 2
        )

        # Step 4: Post-filter by metadata
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.data):
                continue

            doc = self.data[idx]
            doc["score"] = float(score)

            # Apply metadata filters
            meta = doc.get("metadata", {})

            # Year filter
            if filters.get("year") and meta.get("year") != filters["year"]:
                continue

            # Semester filter
            if filters.get("semester_name") and meta.get("semester_name") != filters["semester_name"]:
                continue

            # Type filter (flexible matching)
            if filters.get("type"):
                doc_type = meta.get("type", "").lower()
                if filters["type"] not in doc_type:
                    continue

            # Department filter
            if filters.get("department"):
                doc_dept = meta.get("department", "")
                if filters["department"].lower() not in doc_dept.lower():
                    continue

            candidates.append(doc)

        # Step 5: Return top-K after filtering
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    # --------------------
    # GENERATION WITH VALIDATION
    # --------------------
    def answer(self, question: str) -> str:
        # Greeting check
        if question.lower().strip() in ["hello", "hi", "merhaba", "hey"]:
            return "Hello! I am the IEU Engineering Curriculum Assistant. How can I help you today?"

        # Retrieve with hybrid search
        retrieved = self.retrieve(question, top_k=25)

        if not retrieved:
            return "No relevant curriculum information was found."

        # Format context with priorities
        context = format_context(retrieved, max_chunks=20)

        # Generate answer
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
                    'temperature': 0.0,
                    'num_predict': 500  # Longer responses for listing courses
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}. Please ensure Ollama is running."

    # --------------------
    # DEBUG: Show retrieved chunks
    # --------------------
    def debug_retrieve(self, question: str) -> None:
        """Helper to debug what's being retrieved"""
        print(f"\n🔍 DEBUG MODE: '{question}'")

        filters = self.query_processor.extract_filters(question)
        print(f"📊 Extracted filters: {filters}")

        retrieved = self.retrieve(question, top_k=10)
        print(f"📚 Retrieved {len(retrieved)} chunks:")

        for i, doc in enumerate(retrieved, 1):
            meta = doc["metadata"]
            print(f"\n{i}. Score: {doc.get('score', 0):.4f}")
            print(f"   Type: {meta.get('chunk_type')}")
            print(f"   Dept: {meta.get('department')}")
            print(f"   Year: {meta.get('year')}, Semester: {meta.get('semester')}")
            print(f"   Text: {doc['text'][:150]}...")


# ==================================================
# CLI APP
# ==================================================
def main():
    print("=" * 60)
    print("IEU Engineering Curriculum Chatbot (Enhanced RAG)")
    print("=" * 60)
    print("Commands:")
    print("  - Type your question normally")
    print("  - Type 'debug: <question>' to see what's retrieved")
    print("  - Type 'exit' to quit\n")

    try:
        rag = FAISSRAG()
        print(f"✅ System ready! Loaded {len(rag.data)} chunks.\n")
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # Debug mode
        if q.lower().startswith("debug:"):
            debug_query = q[6:].strip()
            rag.debug_retrieve(debug_query)
            continue

        ans = rag.answer(q)
        print("\nBot:", ans, "\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
"""
Enhanced RAG System for IEU Engineering Curriculum
Production-ready implementation with advanced query processing
"""

import os
import json
import faiss
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
TOP_K = 25

# ==================================================
# ENHANCED SYSTEM PROMPT
# ==================================================
SYSTEM_PROMPT = """
You are an academic assistant for Izmir University of Economics Engineering Faculty.

STRICT OUTPUT RULES:
0. ONLY provide the final answer. 
1. DO NOT write your internal thinking process (e.g., "I will review...", "Upon reviewing...").
2. DO NOT write your CHUNKING process (e.g., "I will extract courses from both Fall and Spring semester curriculum_summary chunks.").
3. DO NOT explain how you found the answer unless specifically asked for the calculation steps.
4. If the answer is not in the context, say ONLY: "The curriculum data does not support this information."
5. If it is a greeting, say "Hello! ... ".

Key Facts:
- Engineering programs: Software Engineering, Computer Engineering, Industrial Engineering, Electrical and Electronics Engineering
- All programs are 4 years (8 semesters)
- Fall semesters: 1, 3, 5, 7 | Spring semesters: 2, 4, 6, 8
- Each year requires 60 ECTS (30 per semester)

ELECTIVE SYSTEM:
▸ ELEC 001, ELEC 002, ELEC 003, etc. (Course codes starting with "ELEC")
   - These are ELECTIVE COURSE SLOTS in the curriculum
   - Type: mandatory (students MUST fill these slots)
   - When asked "How many elective courses?", COUNT THESE SLOTS
   - Example: If curriculum has ELEC 001, ELEC 002, ELEC 003 → Answer: "3 elective courses"

▸ ELECTIVE OPTIONS (Type=elective in metadata)
   - Actual courses students can choose from
   - Examples: CE 304, CE 315, SE 360, etc.
   - Students select from these to fill their ELEC slots

QUANTITATIVE QUERIES - CRITICAL INSTRUCTIONS:
When asked "How many" or "Total ECTS" or "Count":
1. Look for curriculum_summary and statistics_summary chunks FIRST
2. Extract numbers from course lists in the context
3. ALWAYS show your calculation step-by-step
4. Format: "Fall: X courses + Spring: Y courses = Total: Z courses"
5. For ECTS: Sum all ECTS values and show breakdown

CRITICAL RULES:
- Answer ONLY based on provided context chunks
- NEVER speculate or make up information
- For greetings, respond politely
- When listing courses, include course codes AND names
- For counting questions, provide explicit numeric answers
- If data is missing, say: "The curriculum data does not support this information."
"""

USER_PROMPT_TEMPLATE = """
Context (Retrieved Course Data):
{context}

User Question:
{question}

Instructions:
1. Read ALL context chunks carefully, especially curriculum_summary chunks
2. If this is a counting question, extract ALL relevant courses and count them
3. Show your calculation step-by-step for transparency
4. Include course codes AND names when listing
5. If comparing departments, provide side-by-side data
6. If information is incomplete, clearly state what's missing
7. If you have to list courses use bulletlist format.

Answer:
"""


# ==================================================
# ENHANCED QUERY PROCESSOR
# ==================================================
class QueryProcessor:
    """
    Advanced query understanding and classification
    Now supports multi-department extraction.
    """

    @staticmethod
    def extract_filters(query: str) -> Dict[str, any]:
        """Extract metadata filters from natural language query, supporting multi-department"""
        filters = {
            "year": None,
            "semester_name": None,
            "type": None,
            "departments": [],  # List of detected departments
            "query_type": "general"
        }

        query_lower = query.lower()

        # Classify query type
        filters["query_type"] = QueryProcessor._classify_query(query_lower)

        # Extract year
        year_patterns = [
            (r'\b(?:first|1st)\s+year\b', 1),
            (r'\b(?:second|2nd)\s+year\b', 2),
            (r'\b(?:third|3rd)\s+year\b', 3),
            (r'\b(?:fourth|4th|final)\s+year\b', 4),
            (r'\byear\s+(\d)\b', None),
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

        # Extract type
        if re.search(r'\b(?:mandatory|compulsory|required|core)\b', query_lower):
            filters["type"] = "mandatory"
        elif re.search(r'\b(?:elective|optional)\b', query_lower):
            filters["type"] = "elective"

        # Extract departments (multi-department aware)
        dept_keywords = {
            "software engineering": "Software Engineering",
            "computer engineering": "Computer Engineering",
            "industrial engineering": "Industrial Engineering",
            "electrical and electronics engineering": "Electrical and Electronics Engineering",
            "electrical engineering": "Electrical and Electronics Engineering",
            "electronics engineering": "Electrical and Electronics Engineering",
            "electrical": "Electrical and Electronics Engineering",
            "electronics": "Electrical and Electronics Engineering",
        }
        found_departments = set()
        for keyword, dept_name in dept_keywords.items():
            if keyword in query_lower:
                found_departments.add(dept_name)
        # Special: if the query says "all departments" or "each department"
        if re.search(r'\ball departments\b|\beach department\b|\bcompare\b', query_lower):
            found_departments = set([
                "Software Engineering",
                "Computer Engineering",
                "Industrial Engineering",
                "Electrical and Electronics Engineering"
            ])
        filters["departments"] = list(sorted(found_departments))

        return filters

    @staticmethod
    def _classify_query(query: str) -> str:
        """Classify query type for specialized handling"""
        if re.search(r'\b(?:how many|count|number of)\b', query):
            return "counting"
        elif re.search(r'\b(?:total|sum).*ects\b', query):
            return "ects_sum"
        elif re.search(r'\b(?:highest|most|maximum|compare)\b', query):
            return "comparison"
        elif re.search(r'\b(?:list|which courses|what courses)\b', query):
            return "listing"
        else:
            return "general"

    @staticmethod
    def is_counting_query(query: str) -> bool:
        """Detect if query requires counting/aggregation"""
        counting_keywords = [
            r'\bhow many\b',
            r'\btotal\b.*\bects\b',
            r'\bcount\b',
            r'\bnumber of\b',
            r'\bhighest\b.*\b(?:load|ects)\b'
        ]
        return any(re.search(kw, query.lower()) for kw in counting_keywords)

    @staticmethod
    def enrich_query(query: str, filters: Dict) -> str:
        """Add metadata keywords to improve retrieval"""
        enriched_parts = [query]

        if filters.get("year"):
            enriched_parts.append(f"year {filters['year']}")
        if filters.get("semester_name"):
            enriched_parts.append(filters["semester_name"])
        if filters.get("type"):
            enriched_parts.append(filters["type"])
        if filters.get("departments"):
            enriched_parts.extend(filters["departments"])

        # Add query-type-specific keywords
        if filters.get("query_type") == "counting":
            enriched_parts.append("curriculum summary statistics")
        elif filters.get("query_type") == "ects_sum":
            enriched_parts.append("ECTS credits total")

        return " ".join(enriched_parts)


# ==================================================
# CONTEXT FORMATTING
# ==================================================
def format_context(docs: List[Dict], max_chunks: int = 20, query_type: str = "general") -> str:
    """
    Format retrieved chunks grouped by department for multi-department queries.
    """
    # Group by department
    dept_blocks = {}
    for d in docs:
        meta = d["metadata"]
        dept = meta.get("department", "N/A")
        if dept not in dept_blocks:
            dept_blocks[dept] = []
        dept_blocks[dept].append(d)

    # For each department, apply prioritization
    def prioritize(doc_list):
        summaries = []
        metadata = []
        others = []
        for d in doc_list:
            chunk_type = d["metadata"].get("chunk_type", "")
            if chunk_type == "curriculum_summary":
                summaries.append(d)
            elif chunk_type == "metadata_summary":
                metadata.append(d)
            elif chunk_type == "statistics_summary":
                summaries.insert(0, d)  # Highest priority for stats
            else:
                others.append(d)
        # Prioritization based on query type
        if query_type in ["counting", "ects_sum", "comparison"]:
            selected = summaries[:8] + metadata[:8] + others[:max_chunks - 16]
        else:
            selected = summaries[:5] + metadata[:10] + others[:max_chunks - 15]
        return selected[:max_chunks]

    # Compose grouped context
    context_blocks = []
    for dept, doc_list in dept_blocks.items():
        selected = prioritize(doc_list)
        if not selected:
            continue
        context_blocks.append(f"=== {dept} ===")
        for d in selected:
            meta = d["metadata"]
            header = (
                f"[Dept: {meta.get('department', 'N/A')} | "
                f"Code: {meta.get('course_code', 'N/A')} | "
                f"Year: {meta.get('year', 'N/A')} | "
                f"Semester: {meta.get('semester_name', 'N/A')} | "
                f"Type: {meta.get('type', 'N/A')} | "
                f"ECTS: {meta.get('ects', 'N/A')} | "
                f"ChunkType: {meta.get('chunk_type', 'N/A')}]"
            )
            context_blocks.append(f"{header}\n{d['text']}\n")
    return "\n".join(context_blocks)


# ==================================================
# MAIN RAG ENGINE
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

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Balanced Multi-Department Retrieval with dynamic TOP_K.
        If multiple departments are detected, search for each and merge results.
        """
        filters = self.query_processor.extract_filters(query)
        query_type = filters.get("query_type", "general")
        departments = filters.get("departments", [])

        # Determine dynamic TOP_K based on query type and department count
        # Single department or general: 30; cross-department or comparison/counting/ects_sum: 50
        if (departments and len(departments) > 1) or query_type in ["comparison", "counting", "ects_sum"]:
            TOP_K_DYNAMIC = 50
        else:
            TOP_K_DYNAMIC = 30

        num_departments = len(departments) if departments else 1
        per_dept_k = max(1, TOP_K_DYNAMIC // num_departments)

        if departments and len(departments) > 1:
            # Multi-department: search for each separately and merge
            all_results = []
            for dept in departments:
                filters_dept = dict(filters)
                filters_dept["departments"] = [dept]
                # For enrich_query, only use this department
                enriched_query = self.query_processor.enrich_query(query, {**filters, "departments": [dept]})
                docs = self._perform_search(enriched_query, filters, dept, per_dept_k)
                all_results.extend(docs)
            # Optionally, sort by department for context formatting
            return all_results
        else:
            # Single department or no explicit department: normal search (default: all)
            enriched_query = self.query_processor.enrich_query(query, filters)
            dept = departments[0] if departments else None
            docs = self._perform_search(enriched_query, filters, dept, per_dept_k)
            return docs

    def _perform_search(self, enriched_query, filters, dept, top_k):
        """Helper: Perform semantic search and metadata filtering for a single department."""
        q_emb = self.embedder.encode(
            enriched_query,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            np.array([q_emb]), top_k * 2
        )

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.data):
                continue
            doc = self.data[idx].copy()
            doc["score"] = float(score)
            meta = doc.get("metadata", {})
            # Apply filters
            if filters.get("year") and meta.get("year") != filters["year"]:
                continue
            if filters.get("semester_name") and meta.get("semester_name") != filters["semester_name"]:
                continue
            if filters.get("type"):
                doc_type = meta.get("type", "").lower()
                if filters["type"] not in doc_type:
                    continue
            # Department filtering (if dept specified)
            if dept:
                doc_dept = meta.get("department", "")
                if dept.lower() not in doc_dept.lower():
                    continue
            candidates.append(doc)
        # Re-rank with boost for summary chunks
        for doc in candidates:
            chunk_type = doc["metadata"].get("chunk_type", "")
            if chunk_type in ["curriculum_summary", "statistics_summary"]:
                doc["score"] *= 1.3
            elif chunk_type == "metadata_summary":
                doc["score"] *= 1.1
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def answer(self, question: str) -> str:
        """
        Generate answer with validation
        """
        # Greeting check
        if question.lower().strip() in ["hello", "hi", "merhaba", "hey"]:
            return "Hello! I am the IEU Engineering Curriculum Assistant. How can I help you today?"

        # Retrieve
        retrieved = self.retrieve(question, top_k=25)

        if not retrieved:
            return "No relevant curriculum information was found."

        # Detect query type for formatting
        filters = self.query_processor.extract_filters(question)
        query_type = filters.get("query_type", "general")

        # Format context
        context = format_context(retrieved, max_chunks=20, query_type=query_type)

        # Adjust generation parameters
        max_tokens = 500
        if query_type in ["counting", "listing"]:
            max_tokens = 800  # Allow longer responses for lists
        elif query_type == "comparison":
            max_tokens = 700

        # Generate
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
                    'num_predict': max_tokens
                }
            )

            answer = response['message']['content'].strip()

            # Validate quantitative answers
            if query_type in ["counting", "ects_sum"]:
                answer = self._validate_quantitative_answer(question, answer, context)

            return answer

        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}. Please ensure Ollama is running."

    def _validate_quantitative_answer(self, question: str, answer: str, context: str) -> str:
        """
        Ensure quantitative queries have numeric answers
        """
        # Check if answer has numbers
        numbers = re.findall(r'(?<![A-Z])\b(\d+)\b(?!\s*-)', answer)

        if not numbers:
            # Try to extract from context and provide basic count
            course_codes = re.findall(r'\b[A-Z]{2,4}\s+\d{3}\b', context)
            if course_codes:
                return f"Based on the curriculum data, I found {len(set(course_codes))} relevant courses. However, for more detailed information, please refer to the specific curriculum documents."
            else:
                return "I couldn't extract specific numeric data from the available curriculum information."

        return answer

    def debug_retrieve(self, question: str) -> None:
        """Debug helper to inspect retrieval"""
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
            print(f"   Year: {meta.get('year')}, Semester: {meta.get('semester_name')}")
            print(f"   Text: {doc['text'][:150]}...")


# ==================================================
# CLI APP
# ==================================================
def main():
    print("=" * 60)
    print("IEU Engineering Curriculum Chatbot (Production Ready)")
    print("=" * 60)
    print("Commands:")
    print("  - Type your question normally")
    print("  - Type 'debug: <question>' to see retrieval details")
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

        if q.lower().startswith("debug:"):
            debug_query = q[6:].strip()
            rag.debug_retrieve(debug_query)
            continue

        ans = rag.answer(q)
        print("\nBot:", ans, "\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
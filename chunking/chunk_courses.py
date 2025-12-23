import json
import re
from tqdm import tqdm
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "iue_curriculum.json"
OUT_PATH = BASE_DIR / "data" / "processed" / "course_chunks.json"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_str(text: str) -> str:
    """Make strings filesystem / id safe"""
    return (
        text.lower()
        .replace("&", "and")
        .replace(" ", "_")
        .replace("-", "_")
    )


def build_base_metadata(course: dict) -> dict:
    return {
        "department": course.get("department"),
        "course_code": course.get("course_code"),
        "year": course.get("year"),
        "semester": course.get("semester"),
        "type": course.get("type"),
        "ects": course.get("ects"),
        "local_credit": course.get("local_credit"),
    }


# --------------------------------------------------
# CHUNKING LOGIC
# --------------------------------------------------
def chunk_course(course: dict) -> list[dict]:
    chunks = []

    base_meta = build_base_metadata(course)

    department = safe_str(course.get("department", "unknown_department"))
    course_code_raw = course.get("course_code", "")
    course_code = course_code_raw.replace(" ", "").lower()

    # --------------------------------------------------
    # 1. COURSE OVERVIEW
    # --------------------------------------------------
    overview_text = f"{course.get('course_name', '')}. {course.get('description', '')}"
    overview_text = normalize_text(overview_text)

    if overview_text:
        chunk_id = f"{department}_{course_code}_overview_0"
        chunks.append({
            "id": chunk_id,
            "text": overview_text,
            "metadata": {
                **base_meta,
                "chunk_type": "course_overview"
            }
        })

    # --------------------------------------------------
    # 2. OBJECTIVES
    # --------------------------------------------------
    objectives_text = normalize_text(course.get("objectives", ""))
    if objectives_text:
        chunk_id = f"{department}_{course_code}_objectives_0"
        chunks.append({
            "id": chunk_id,
            "text": objectives_text,
            "metadata": {
                **base_meta,
                "chunk_type": "objectives"
            }
        })

    # --------------------------------------------------
    # 3. PREREQUISITES
    # --------------------------------------------------
    prereq_text = normalize_text(course.get("prerequisites", ""))
    if prereq_text and prereq_text.lower() not in ["none", "n/a", "-"]:
        chunk_id = f"{department}_{course_code}_prerequisites_0"
        chunks.append({
            "id": chunk_id,
            "text": prereq_text,
            "metadata": {
                **base_meta,
                "chunk_type": "prerequisites"
            }
        })

    # --------------------------------------------------
    # 4. WEEKLY TOPICS (INDEXED!)
    # --------------------------------------------------
    weekly_topics = course.get("weekly_topics", [])
    for idx, wt in enumerate(weekly_topics):
        week = wt.get("week")
        topic = normalize_text(wt.get("topic"))

        if not topic:
            continue

        chunk_id = f"{department}_{course_code}_weekly_{idx}"

        chunks.append({
            "id": chunk_id,
            "text": f"Week {week}: {topic}",
            "metadata": {
                **base_meta,
                "chunk_type": "weekly_topic",
                "week": int(week) if str(week).isdigit() else week
            }
        })

    return chunks


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        courses = json.load(f)

    all_chunks = []

    for course in tqdm(courses, desc="Chunking courses"):
        if not course.get("course_code"):
            continue
        course_chunks = chunk_course(course)
        all_chunks.extend(course_chunks)

    # --------- FINAL SAFETY CHECK ---------
    ids = [c["id"] for c in all_chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("CRITICAL: Duplicate chunk IDs still exist!")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)

    print(f"\nDONE. Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
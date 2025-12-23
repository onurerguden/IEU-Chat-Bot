import json
import os
import re
from tqdm import tqdm
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "iue_curriculum.json"
OUT_PATH = BASE_DIR / "data" / "processed" / "course_chunks.json"


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def chunk_course(course: dict) -> list[dict]:
    chunks = []
    base_meta = build_base_metadata(course)
    code = course.get("course_code", "").replace(" ", "")

    # 1. COURSE OVERVIEW (Name + Description)
    overview_text = f"{course.get('course_name', '')}. {course.get('description', '')}"
    overview_text = normalize_text(overview_text)

    if overview_text:
        chunks.append({
            "id": f"{code}_overview",
            "text": overview_text,
            "metadata": {
                **base_meta,
                "chunk_type": "course_overview"
            }
        })

    # 2. OBJECTIVES
    objectives_text = normalize_text(course.get("objectives", ""))
    if objectives_text:
        chunks.append({
            "id": f"{code}_objectives",
            "text": objectives_text,
            "metadata": {
                **base_meta,
                "chunk_type": "objectives"
            }
        })

    # 3. PREREQUISITES
    prereq_text = normalize_text(course.get("prerequisites", ""))
    if prereq_text and prereq_text.lower() not in ["none", "n/a", "-"]:
        chunks.append({
            "id": f"{code}_prerequisites",
            "text": prereq_text,
            "metadata": {
                **base_meta,
                "chunk_type": "prerequisites"
            }
        })

    # 4. WEEKLY TOPICS (ONE CHUNK PER WEEK)
    weekly_topics = course.get("weekly_topics", [])
    for wt in weekly_topics:
        week = wt.get("week")
        topic = normalize_text(wt.get("topic"))

        if not topic:
            continue

        chunks.append({
            "id": f"{code}_week_{week}",
            "text": f"Week {week}: {topic}",
            "metadata": {
                **base_meta,
                "chunk_type": "weekly_topic",
                "week": int(week) if str(week).isdigit() else week
            }
        })

    return chunks


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

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)

    print(f"\nDONE. Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
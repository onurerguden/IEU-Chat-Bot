import json
import re
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

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
        "semester_name": course.get("semester_name"),
        "type": course.get("type"),
        "ects": course.get("ects"),
        "local_credit": course.get("local_credit"),
    }


# --------------------------------------------------
# ENHANCED CHUNKING LOGIC
# --------------------------------------------------
def chunk_course(course: dict) -> list[dict]:
    """
    BEST PRACTICE: Multi-level chunking
    - Course overview (searchable by description)
    - Metadata-rich summaries (searchable by year/semester/type)
    - Objectives, Prerequisites, Weekly topics
    """
    chunks = []
    base_meta = build_base_metadata(course)

    department = safe_str(course.get("department", "unknown_department"))
    course_code_raw = course.get("course_code", "")
    course_code = course_code_raw.replace(" ", "").lower()

    # --------------------------------------------------
    # 1. METADATA-RICH COURSE SUMMARY (CRITICAL!)
    # --------------------------------------------------
    # Bu chunk, "second year mandatory courses" gibi sorulara cevap verebilmek için
    summary_parts = [
        f"Course: {course.get('course_name', 'N/A')}",
        f"Code: {course_code_raw}",
        f"Department: {course.get('department', 'N/A')}",
    ]

    if course.get('year'):
        summary_parts.append(f"Year: {course.get('year')} (Year {course.get('year')})")
    if course.get('semester_name'):
        summary_parts.append(f"Semester: {course.get('semester_name')}")
    if course.get('type'):
        summary_parts.append(f"Type: {course.get('type').capitalize()}")
    if course.get('ects'):
        summary_parts.append(f"ECTS: {course.get('ects')}")

    # Description ekliyoruz ki semantic search de çalışsın
    if course.get('description'):
        summary_parts.append(f"Description: {normalize_text(course.get('description', ''))[:200]}")

    summary_text = ". ".join(summary_parts) + "."

    chunks.append({
        "id": f"{department}_{course_code}_metadata_summary",
        "text": summary_text,
        "metadata": {
            **base_meta,
            "chunk_type": "metadata_summary"
        }
    })

    # --------------------------------------------------
    # 2. COURSE OVERVIEW (Full description for content search)
    # --------------------------------------------------
    overview_text = f"{course.get('course_name', '')}. {course.get('description', '')}"
    overview_text = normalize_text(overview_text)

    if overview_text:
        chunks.append({
            "id": f"{department}_{course_code}_overview",
            "text": overview_text,
            "metadata": {
                **base_meta,
                "chunk_type": "course_overview"
            }
        })

    # --------------------------------------------------
    # 3. OBJECTIVES (with metadata context)
    # --------------------------------------------------
    objectives_text = normalize_text(course.get("objectives", ""))
    if objectives_text:
        # Metadata ekliyoruz ki "optimization course objectives" bulabilsin
        enriched_obj = f"Course {course_code_raw} objectives: {objectives_text}"
        chunks.append({
            "id": f"{department}_{course_code}_objectives",
            "text": enriched_obj,
            "metadata": {
                **base_meta,
                "chunk_type": "objectives"
            }
        })

    # --------------------------------------------------
    # 4. PREREQUISITES
    # --------------------------------------------------
    prereq_text = normalize_text(course.get("prerequisites", ""))
    if prereq_text and prereq_text.lower() not in ["none", "n/a", "-"]:
        chunks.append({
            "id": f"{department}_{course_code}_prerequisites",
            "text": f"Prerequisites for {course_code_raw}: {prereq_text}",
            "metadata": {
                **base_meta,
                "chunk_type": "prerequisites"
            }
        })

    # --------------------------------------------------
    # 5. WEEKLY TOPICS (with context)
    # --------------------------------------------------
    weekly_topics = course.get("weekly_topics", [])
    for idx, wt in enumerate(weekly_topics):
        week = wt.get("week")
        topic = normalize_text(wt.get("topic"))

        if not topic:
            continue

        # Topic'lere course context ekliyoruz
        enriched_topic = f"{course_code_raw} Week {week}: {topic}"

        chunks.append({
            "id": f"{department}_{course_code}_weekly_{idx}",
            "text": enriched_topic,
            "metadata": {
                **base_meta,
                "chunk_type": "weekly_topic",
                "week": int(week) if str(week).isdigit() else week
            }
        })

    return chunks


# --------------------------------------------------
# CURRICULUM-LEVEL SUMMARIES (NEW!)
# --------------------------------------------------
def create_curriculum_summaries(courses: list[dict]) -> list[dict]:
    """
    BEST PRACTICE: Create aggregated chunks for year/semester/type queries
    Bu, "second year courses" gibi sorulara doğrudan cevap verebilmemizi sağlar
    """
    summaries = []

    # Department → Year → Semester → Type → Courses mapping
    curriculum_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for course in courses:
        dept = course.get("department")
        year = course.get("year")
        semester = course.get("semester")
        ctype = course.get("type", "unknown")

        if not all([dept, year, semester]):
            continue

        curriculum_map[dept][year][semester][ctype].append(course)

    # Generate summary chunks
    for dept, years in curriculum_map.items():
        dept_safe = safe_str(dept)

        for year, semesters in years.items():
            for semester, types in semesters.items():
                for ctype, course_list in types.items():
                    if not course_list:
                        continue

                    # Create a summary text
                    course_names = [f"{c.get('course_code')} - {c.get('course_name')}"
                                    for c in course_list if c.get('course_code')]

                    semester_name = course_list[0].get('semester_name', f'Semester {semester}')

                    summary_text = (
                        f"{dept} Year {year} {semester_name} {ctype.capitalize()} Courses: "
                        f"{', '.join(course_names[:10])}"  # İlk 10 kursu listele
                    )

                    if len(course_names) > 10:
                        summary_text += f" and {len(course_names) - 10} more courses"

                    total_ects = sum(c.get('ects', 0) for c in course_list if c.get('ects'))
                    summary_text += f". Total ECTS: {total_ects}."

                    summaries.append({
                        "id": f"{dept_safe}_y{year}_s{semester}_{ctype}_summary",
                        "text": summary_text,
                        "metadata": {
                            "department": dept,
                            "year": year,
                            "semester": semester,
                            "semester_name": semester_name,
                            "type": ctype,
                            "course_count": len(course_list),
                            "total_ects": total_ects,
                            "chunk_type": "curriculum_summary"
                        }
                    })

    return summaries


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

    # 1. Course-level chunks
    print("Creating course-level chunks...")
    for course in tqdm(courses, desc="Chunking courses"):
        if not course.get("course_code"):
            continue
        course_chunks = chunk_course(course)
        all_chunks.extend(course_chunks)

    # 2. Curriculum-level summaries (NEW!)
    print("Creating curriculum-level summaries...")
    curriculum_chunks = create_curriculum_summaries(courses)
    all_chunks.extend(curriculum_chunks)

    # Final safety check
    ids = [c["id"] for c in all_chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("CRITICAL: Duplicate chunk IDs exist!")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)

    print(f"\n✅ DONE. Total chunks created: {len(all_chunks)}")
    print(f"   - Course chunks: {len(all_chunks) - len(curriculum_chunks)}")
    print(f"   - Curriculum summaries: {len(curriculum_chunks)}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
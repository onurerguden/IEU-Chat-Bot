"""
Enhanced Course Chunking with Statistical Summaries
Optimized for quantitative queries
"""

import json
import re
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

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
    """Make strings filesystem/id safe"""
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
# COURSE-LEVEL CHUNKING
# --------------------------------------------------
def chunk_course(course: dict) -> List[dict]:
    """
    Multi-level chunking for each course
    """
    chunks = []
    base_meta = build_base_metadata(course)

    department = safe_str(course.get("department", "unknown_department"))
    course_code_raw = course.get("course_code", "")
    course_code = course_code_raw.replace(" ", "").lower()

    # 1. METADATA-RICH SUMMARY
    summary_parts = [
        f"Course: {course.get('course_name', 'N/A')}",
        f"Code: {course_code_raw}",
        f"Department: {course.get('department', 'N/A')}",
    ]

    if course.get('year'):
        summary_parts.append(f"Year: {course.get('year')}")
    if course.get('semester_name'):
        summary_parts.append(f"Semester: {course.get('semester_name')}")
    if course.get('type'):
        summary_parts.append(f"Type: {course.get('type').capitalize()}")
    if course.get('ects'):
        summary_parts.append(f"ECTS: {course.get('ects')}")

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

    # 2. COURSE OVERVIEW
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

    # 3. OBJECTIVES
    objectives_text = normalize_text(course.get("objectives", ""))
    if objectives_text:
        enriched_obj = f"Course {course_code_raw} objectives: {objectives_text}"
        chunks.append({
            "id": f"{department}_{course_code}_objectives",
            "text": enriched_obj,
            "metadata": {
                **base_meta,
                "chunk_type": "objectives"
            }
        })

    # 4. PREREQUISITES
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

    # 5. WEEKLY TOPICS
    weekly_topics = course.get("weekly_topics", [])
    for idx, wt in enumerate(weekly_topics):
        week = wt.get("week")
        topic = normalize_text(wt.get("topic"))

        if not topic:
            continue

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
# CURRICULUM SUMMARIES
# --------------------------------------------------
def create_curriculum_summaries(courses: List[dict]) -> List[dict]:
    """
    Create aggregated curriculum summaries
    Optimized for "list courses in year X" queries
    """
    summaries = []

    curriculum_map = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
    )

    for course in courses:
        dept = course.get("department")
        year = course.get("year")
        semester = course.get("semester")
        ctype = course.get("type", "unknown")

        if not all([dept, year, semester]):
            continue

        curriculum_map[dept][year][semester][ctype].append(course)

    for dept, years in curriculum_map.items():
        dept_safe = safe_str(dept)

        for year, semesters in years.items():
            for semester, types in semesters.items():
                for ctype, course_list in types.items():
                    if not course_list:
                        continue

                    course_names = [
                        f"{c.get('course_code')} - {c.get('course_name')}"
                        for c in course_list if c.get('course_code')
                    ]

                    semester_name = course_list[0].get('semester_name', f'Semester {semester}')

                    summary_text = (
                        f"{dept} Year {year} {semester_name} {ctype.capitalize()} Courses: "
                        f"{', '.join(course_names[:15])}"
                    )

                    if len(course_names) > 15:
                        summary_text += f" and {len(course_names) - 15} more courses"

                    total_ects = sum(c.get('ects', 0) for c in course_list if c.get('ects'))
                    summary_text += f". Total ECTS: {total_ects}. Course count: {len(course_list)}."

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
# STATISTICAL SUMMARIES (NEW!)
# --------------------------------------------------
def create_detailed_statistics(courses: List[dict]) -> List[dict]:
    """
    Create comprehensive statistical summaries
    Optimized for quantitative queries
    """
    stats_chunks = []

    # Department-level statistics
    dept_stats = defaultdict(lambda: {
        'total_courses': 0,
        'total_ects': 0,
        'mandatory_count': 0,
        'elective_count': 0,
        'by_year': defaultdict(lambda: {
            'courses': 0,
            'ects': 0,
            'mandatory': 0,
            'elective': 0,
            'by_semester': defaultdict(lambda: {
                'courses': [],
                'ects': 0
            })
        })
    })

    for course in courses:
        dept = course.get('department')
        year = course.get('year')
        semester_name = course.get('semester_name')
        ctype = course.get('type', 'unknown')
        ects = course.get('ects', 0)
        code = course.get('course_code')

        if not all([dept, year]):
            continue

        stats = dept_stats[dept]
        stats['total_courses'] += 1
        stats['total_ects'] += ects if ects else 0

        if ctype == 'mandatory':
            stats['mandatory_count'] += 1
        elif ctype == 'elective':
            stats['elective_count'] += 1

        year_stats = stats['by_year'][year]
        year_stats['courses'] += 1
        year_stats['ects'] += ects if ects else 0

        if ctype == 'mandatory':
            year_stats['mandatory'] += 1
        elif ctype == 'elective':
            year_stats['elective'] += 1

        if semester_name:
            sem_stats = year_stats['by_semester'][semester_name]
            sem_stats['courses'].append(code)
            sem_stats['ects'] += ects if ects else 0

    # Generate comprehensive stat chunks
    for dept, stats in dept_stats.items():
        dept_safe = safe_str(dept)

        # Overall department statistics
        summary_text = (
            f"{dept} Complete Statistics: "
            f"Total courses: {stats['total_courses']} "
            f"(Mandatory: {stats['mandatory_count']}, Elective: {stats['elective_count']}). "
            f"Total ECTS: {stats['total_ects']}. "
        )

        # Year-by-year breakdown
        year_details = []
        for year in sorted(stats['by_year'].keys()):
            y_stats = stats['by_year'][year]
            year_text = (
                f"Year {year}: {y_stats['courses']} courses, "
                f"{y_stats['ects']} ECTS "
                f"({y_stats['mandatory']} mandatory, {y_stats['elective']} elective)"
            )

            # Semester breakdown
            for sem_name in ['FALL', 'SPRING']:
                if sem_name in y_stats['by_semester']:
                    sem_data = y_stats['by_semester'][sem_name]
                    year_text += f"; {sem_name}: {len(sem_data['courses'])} courses"

            year_details.append(year_text)

        summary_text += ". ".join(year_details) + "."

        stats_chunks.append({
            "id": f"{dept_safe}_statistics_summary",
            "text": summary_text,
            "metadata": {
                "department": dept,
                "chunk_type": "statistics_summary",
                "total_courses": stats['total_courses'],
                "total_ects": stats['total_ects'],
                "mandatory_count": stats['mandatory_count'],
                "elective_count": stats['elective_count']
            }
        })

        # Year-specific stat chunks
        for year in sorted(stats['by_year'].keys()):
            y_stats = stats['by_year'][year]

            year_text = (
                f"{dept} Year {year} Statistics: "
                f"{y_stats['courses']} total courses, "
                f"{y_stats['ects']} ECTS. "
                f"Mandatory: {y_stats['mandatory']}, Elective: {y_stats['elective']}. "
            )

            # Semester details
            for sem_name in ['FALL', 'SPRING']:
                if sem_name in y_stats['by_semester']:
                    sem_data = y_stats['by_semester'][sem_name]
                    year_text += (
                        f"{sem_name} semester: {len(sem_data['courses'])} courses "
                        f"({sem_data['ects']} ECTS). "
                    )

            stats_chunks.append({
                "id": f"{dept_safe}_y{year}_statistics",
                "text": year_text,
                "metadata": {
                    "department": dept,
                    "year": year,
                    "chunk_type": "statistics_summary",
                    "course_count": y_stats['courses'],
                    "total_ects": y_stats['ects']
                }
            })

    return stats_chunks


# --------------------------------------------------
# MAIN PROCESSING
# --------------------------------------------------
def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        courses = json.load(f)

    print(f" Loaded {len(courses)} courses")

    all_chunks = []

    # 1. Course-level chunks
    print("\n🔨 Creating course-level chunks...")
    for course in tqdm(courses, desc="Chunking courses"):
        if not course.get("course_code"):
            continue
        course_chunks = chunk_course(course)
        all_chunks.extend(course_chunks)

    # 2. Curriculum summaries
    print("\n Creating curriculum summaries...")
    curriculum_chunks = create_curriculum_summaries(courses)
    all_chunks.extend(curriculum_chunks)

    # 3. Statistical summaries (NEW!)
    print("\n Creating statistical summaries...")
    stats_chunks = create_detailed_statistics(courses)
    all_chunks.extend(stats_chunks)

    # Validation
    ids = [c["id"] for c in all_chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("CRITICAL: Duplicate chunk IDs exist!")

    # Save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n DONE. Total chunks created: {len(all_chunks)}")
    print(f"    Course chunks: {len(all_chunks) - len(curriculum_chunks) - len(stats_chunks)}")
    print(f"    Curriculum summaries: {len(curriculum_chunks)}")
    print(f"    Statistical summaries: {len(stats_chunks)}")
    print(f"\n Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
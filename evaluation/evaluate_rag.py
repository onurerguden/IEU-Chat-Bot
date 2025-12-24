"""
RAG System Evaluation Script
Tests the system against project requirements
"""

import json
import sys
from pathlib import Path

from rag.rag_app import FAISSRAG

# Add parent directory to path to import from rag folder
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "rag"))

# Now import from rag folder
from rag import rag_app

# Test questions from project requirements
TEST_QUESTIONS = {
    "single_department": [
        "What are the core courses in the second year of Software Engineering?",
        "Which mandatory courses are offered in the third year of Computer Engineering?",
        "Which courses focus on optimization techniques in Industrial Engineering?",
        "List all database-related courses in Computer Engineering.",
        "What programming courses are taught in year 1 of Software Engineering?",
        "Which courses in Electrical Engineering cover power systems?",
        "What are the elective options for Industrial Engineering students?",
        "Which Software Engineering courses have 6 ECTS credit?",
        "List the mandatory courses in the final year of Computer Engineering.",
        "What courses in Industrial Engineering focus on quality control?"

    ],
    "topic_based": [
        "Which engineering departments offer machine-learning-related courses?",
        "Which courses cover signal processing topics?",
        "List all courses related to database management across all departments.",
        "Which courses teach web development?",
        "What courses focus on artificial intelligence?",
        "Which departments offer courses on optimization?",
        "List all courses related to computer networks.",
        "Which courses cover mobile application development?",
        "What courses teach data analysis techniques?",
        "Which courses focus on software architecture?"

    ],
    "cross_department": [
        "List programming-focused courses in Software Engineering and Computer Engineering.",
        "Which department emphasizes data analysis more strongly?",
        "Compare machine learning courses in Computer and Industrial Engineering.",
        "Which department has more elective courses: Software or Computer Engineering?",
        "What are the differences in signal processing education between Electrical and Computer Engineering?",
        "Compare the database courses offered in Software and Computer Engineering.",
        "Which department requires more mathematics courses in year 1?",
        "List optimization courses in Industrial versus Computer Engineering.",
        "Which department offers more embedded systems courses?",
        "Compare the total ECTS load in year 3 across all engineering departments."

    ],
    "quantitative": [
        "How many elective courses are offered in the final year of Software Engineering?",
        "Which academic year has the highest total ECTS load?",
        "How many mandatory courses are in year 2 of Computer Engineering?",
        "What is the total ECTS of elective courses in Industrial Engineering?",
        "How many courses focus on programming in Software Engineering?",
        "Which year has the most courses in Electrical Engineering?",
        "How many 6-ECTS courses are there in Computer Engineering?",
        "What is the total number of elective slots in Software Engineering curriculum?",
        "How many courses are offered in Fall semester of year 3 for Industrial Engineering?",
        "Which department has the highest number of technical elective options?"

    ],
    "hallucination_trap": [
        "Does the Software Engineering department offer a course on Quantum Thermodynamics?",
        "Is there a compulsory course on Astrophysics in Computer Engineering?",
        "Does Industrial Engineering teach a course on Ancient Greek Philosophy?",
        "Is there a course on Marine Biology in Electrical Engineering?",
        "Does Computer Engineering offer a course on Medieval History?",
        "Is Organic Chemistry a mandatory course in Software Engineering?",
        "Does the Industrial Engineering curriculum include a course on Archaeology?",
        "Is there a course on Constitutional Law in Computer Engineering?",
        "Does Electrical Engineering offer a course on Paleontology?",
        "Is there a mandatory course on Zoology in Software Engineering?",
        "Does Computer Engineering teach a course called 'Quantum Entanglement Theory'?",
        "Is 'Advanced Alchemy' an elective in Industrial Engineering?",
        "Does Software Engineering offer 'Introduction to Telepathy'?",
        "Is there a course on 'Interstellar Navigation' in Electrical Engineering?",
        "Does Industrial Engineering have a course called 'Time Travel Engineering'?",
        "Is 'Mythological Creatures Studies' offered in Computer Engineering?",
        "Does Software Engineering teach 'Advanced Potion Making'?",
        "Is there a course on 'Underwater Basket Weaving' in any department?",
        "Does Electrical Engineering offer 'Introduction to Dragon Taming'?",
        "Is 'Medieval Sword Fighting' a technical elective in Software Engineering?"

    ]
}


def evaluate_system():
    """
    Run evaluation and save results
    """
    print("=" * 70)
    print("RAG SYSTEM EVALUATION")
    print("=" * 70)

    try:
        rag = FAISSRAG()
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        return

    results = []

    for category, questions in TEST_QUESTIONS.items():
        print(f"\n{'=' * 70}")
        print(f"CATEGORY: {category.upper().replace('_', ' ')}")
        print(f"{'=' * 70}")

        for i, question in enumerate(questions, 1):
            print(f"\n[Q{i}] {question}")
            print("-" * 70)

            try:
                answer = rag.answer(question)
                print(f"[A] {answer}")

                # Evaluate answer quality
                quality = evaluate_answer(question, answer, category)

                results.append({
                    "category": category,
                    "question": question,
                    "answer": answer,
                    "quality": quality
                })

                print(f"\n✓ Quality: {quality}")

            except Exception as e:
                print(f"❌ ERROR: {e}")
                results.append({
                    "category": category,
                    "question": question,
                    "answer": f"ERROR: {str(e)}",
                    "quality": "error"
                })

    # Save results
    output_path = Path("evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"✅ Evaluation complete! Results saved to: {output_path}")
    print(f"{'=' * 70}")

    # Summary
    print_summary(results)


def evaluate_answer(question: str, answer: str, category: str, re=None) -> str:
    """
    Simple heuristic evaluation
    """
    answer_lower = answer.lower()

    # Check for "no data" responses
    if "does not support" in answer_lower or "no relevant" in answer_lower:
        if category == "hallucination_trap":
            return "CORRECT (Properly rejected)"
        else:
            return "POOR (Missing data)"

    # Check for hallucination traps
    if category == "hallucination_trap":
        # Should say "no" or "does not support"
        if any(word in answer_lower for word in ["yes", "offers", "available", "there is"]):
            return "HALLUCINATION (False positive)"
        else:
            return "CORRECT (Properly rejected)"

    # Check for course listings
    if "list" in question.lower() or "which courses" in question.lower():
        # Count if answer has course codes (e.g., "SE 226", "CE 316")
        import re
        course_codes = re.findall(r'\b[A-Z]{2,4}\s+\d{3}\b', answer)
        if len(course_codes) >= 2:
            return "GOOD (Listed courses)"
        elif len(course_codes) == 1:
            return "PARTIAL (Few courses listed)"
        else:
            return "POOR (No course codes)"

    # Check for counting questions
    if any(word in question.lower() for word in ["how many", "count", "total"]):
        if re.search(r'\b\d+\b', answer):
            return "GOOD (Provided count)"
        else:
            return "POOR (No numeric answer)"

    # Default: if answer is substantial
    if len(answer) > 100 and answer != "The curriculum data does not support this information.":
        return "GOOD (Detailed answer)"
    else:
        return "PARTIAL (Brief answer)"


def print_summary(results: list):
    """
    Print evaluation summary
    """
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    from collections import Counter

    quality_counts = Counter([r["quality"] for r in results])
    category_counts = Counter([r["category"] for r in results])

    print(f"\nTotal questions: {len(results)}")
    print(f"\nBy category:")
    for cat, count in category_counts.items():
        print(f"  - {cat.replace('_', ' ').title()}: {count}")

    print(f"\nBy quality:")
    for quality, count in sorted(quality_counts.items()):
        print(f"  - {quality}: {count}")

    # Calculate success rate
    success_count = sum(1 for r in results if "GOOD" in r["quality"] or "CORRECT" in r["quality"])
    success_rate = (success_count / len(results)) * 100

    print(f"\n{'=' * 70}")
    print(f"SUCCESS RATE: {success_rate:.1f}%")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    evaluate_system()
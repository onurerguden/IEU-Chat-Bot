"""
RAG System Evaluation Script - Enhanced Version
Tests the system against project requirements with improved metrics
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "rag"))

from rag.rag_app import FAISSRAG

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


class EvaluationMetrics:
    """Enhanced evaluation metrics with detailed analysis"""

    @staticmethod
    def has_numeric_answer(answer: str) -> bool:
        """Check if answer contains numeric information"""
        # Look for standalone numbers (not part of course codes)
        numbers = re.findall(r'(?<![A-Z])\b(\d+)\b(?!\s*-)', answer)
        return len(numbers) > 0

    @staticmethod
    def extract_course_codes(answer: str) -> List[str]:
        """Extract course codes from answer"""
        return re.findall(r'\b[A-Z]{2,4}\s+\d{3}\b', answer)

    @staticmethod
    def is_negative_response(answer: str) -> bool:
        """Check if answer is a negative/rejection response"""
        negative_phrases = [
            "does not support",
            "no relevant",
            "not found",
            "cannot find",
            "no information",
            "data does not contain"
        ]
        return any(phrase in answer.lower() for phrase in negative_phrases)

    @staticmethod
    def is_positive_hallucination(answer: str) -> bool:
        """Check if answer falsely confirms non-existent information"""
        positive_phrases = [
            "yes",
            "offers",
            "available",
            "there is",
            "includes",
            "provides",
            "teaches"
        ]
        return any(phrase in answer.lower() for phrase in positive_phrases)


def evaluate_answer(question: str, answer: str, category: str) -> Tuple[str, Dict]:
    """
    Enhanced evaluation with detailed metrics

    Returns:
        Tuple of (quality_label, detailed_metrics)
    """
    metrics = EvaluationMetrics()
    answer_lower = answer.lower()

    detailed = {
        "has_numbers": metrics.has_numeric_answer(answer),
        "course_count": len(metrics.extract_course_codes(answer)),
        "answer_length": len(answer),
        "is_negative": metrics.is_negative_response(answer)
    }

    # === HALLUCINATION TRAP EVALUATION ===
    if category == "hallucination_trap":
        if metrics.is_negative_response(answer):
            return "CORRECT (Properly rejected)", detailed
        elif metrics.is_positive_hallucination(answer):
            return "HALLUCINATION (False positive)", detailed
        else:
            return "CORRECT (Properly rejected)", detailed

    # === MISSING DATA CHECK ===
    if metrics.is_negative_response(answer):
        return "POOR (Missing data)", detailed

    # === QUANTITATIVE EVALUATION ===
    if category == "quantitative":
        counting_keywords = ["how many", "count", "total", "number of"]
        is_counting = any(kw in question.lower() for kw in counting_keywords)

        if is_counting:
            if metrics.has_numeric_answer(answer):
                # Check if calculation is shown
                if "+" in answer or "=" in answer or "total:" in answer_lower:
                    return "EXCELLENT (Detailed calculation)", detailed
                return "GOOD (Provided count)", detailed
            else:
                return "POOR (No numeric answer)", detailed

        # Comparison questions (highest, most, etc.)
        comparison_keywords = ["highest", "most", "compare", "versus"]
        is_comparison = any(kw in question.lower() for kw in comparison_keywords)

        if is_comparison and metrics.has_numeric_answer(answer):
            return "GOOD (Comparative analysis)", detailed

    # === COURSE LISTING EVALUATION ===
    listing_keywords = ["list", "which courses", "what courses"]
    is_listing = any(kw in question.lower() for kw in listing_keywords)

    if is_listing:
        course_codes = metrics.extract_course_codes(answer)
        if len(course_codes) >= 3:
            return "GOOD (Listed courses)", detailed
        elif len(course_codes) >= 1:
            return "PARTIAL (Few courses listed)", detailed
        else:
            # Check if it's a descriptive answer without codes
            if len(answer) > 150:
                return "GOOD (Detailed description)", detailed
            return "POOR (No course codes)", detailed

    # === DEFAULT EVALUATION ===
    if len(answer) > 150:
        return "GOOD (Detailed answer)", detailed
    elif len(answer) > 50:
        return "PARTIAL (Brief answer)", detailed
    else:
        return "POOR (Too brief)", detailed


def evaluate_system():
    """Run comprehensive evaluation"""
    print("=" * 70)
    print("RAG SYSTEM EVALUATION - ENHANCED VERSION")
    print("=" * 70)

    try:
        rag = FAISSRAG()
        print(f" RAG initialized with {len(rag.data)} chunks\n")
    except Exception as e:
        print(f" Failed to initialize RAG: {e}")
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
                print(f"[A] {answer[:300]}{'...' if len(answer) > 300 else ''}")

                quality, details = evaluate_answer(question, answer, category)

                result = {
                    "category": category,
                    "question": question,
                    "answer": answer,
                    "quality": quality,
                    "metrics": details
                }
                results.append(result)

                print(f"\n✓ Quality: {quality}")
                if details.get("course_count", 0) > 0:
                    print(f"   Courses found: {details['course_count']}")
                if details.get("has_numbers"):
                    print(f"  Contains numeric data: Yes")

            except Exception as e:
                print(f" ERROR: {e}")
                import traceback
                traceback.print_exc()

                results.append({
                    "category": category,
                    "question": question,
                    "answer": f"ERROR: {str(e)}",
                    "quality": "ERROR",
                    "metrics": {"error": str(e)}
                })

    # Save results
    output_path = Path("evaluation_results_enhanced.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f" Evaluation complete! Results saved to: {output_path}")
    print(f"{'=' * 70}")

    print_summary(results)


def print_summary(results: List[Dict]):
    """Print detailed evaluation summary"""
    print("\n" + "=" * 70)
    print("DETAILED SUMMARY")
    print("=" * 70)

    quality_counts = Counter([r["quality"] for r in results])
    category_counts = Counter([r["category"] for r in results])

    print(f"\n Total questions: {len(results)}")

    # Category breakdown
    print(f"\n By category:")
    for cat, count in sorted(category_counts.items()):
        cat_results = [r for r in results if r["category"] == cat]
        success = sum(1 for r in cat_results if
                      "GOOD" in r["quality"] or "CORRECT" in r["quality"] or "EXCELLENT" in r["quality"])
        rate = (success / count * 100) if count > 0 else 0
        print(f"  • {cat.replace('_', ' ').title()}: {success}/{count} ({rate:.1f}%)")

    # Quality breakdown
    print(f"\n By quality:")
    for quality, count in sorted(quality_counts.items(), key=lambda x: -x[1]):
        print(f"  • {quality}: {count}")

    # Success metrics
    success_count = sum(1 for r in results if any(word in r["quality"] for word in ["GOOD", "CORRECT", "EXCELLENT"]))
    total = len(results)
    success_rate = (success_count / total * 100) if total > 0 else 0

    # Detailed category analysis
    print(f"\n Detailed Analysis:")
    for category in TEST_QUESTIONS.keys():
        cat_results = [r for r in results if r["category"] == category]
        if not cat_results:
            continue

        success = sum(1 for r in cat_results if any(word in r["quality"] for word in ["GOOD", "CORRECT", "EXCELLENT"]))
        total_cat = len(cat_results)

        # Find weakest question
        weakest = min(cat_results, key=lambda x: 1 if "GOOD" in x["quality"] or "CORRECT" in x["quality"] else 0)

        print(f"\n  {category.replace('_', ' ').title()}:")
        print(f"    Success: {success}/{total_cat} ({success / total_cat * 100:.1f}%)")
        if weakest["quality"] not in ["GOOD (Listed courses)", "GOOD (Detailed answer)", "CORRECT (Properly rejected)"]:
            print(f"      Weakest: {weakest['question'][:60]}...")
            print(f"       Quality: {weakest['quality']}")

    print(f"\n{'=' * 70}")
    print(f"OVERALL SUCCESS RATE: {success_rate:.1f}%")
    print(f"{'=' * 70}\n")

    # Recommendations
    print(" Recommendations:")
    if quality_counts.get("POOR (No numeric answer)", 0) > 0:
        print("  • Improve quantitative query handling")
    if quality_counts.get("POOR (Missing data)", 0) > 0:
        print("  • Add more curriculum summary chunks")
    if quality_counts.get("ERROR", 0) > 0:
        print("  • Fix error handling in retrieval pipeline")
    print()


if __name__ == "__main__":
    evaluate_system()
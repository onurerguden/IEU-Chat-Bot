"""Shared CLI and PyQt application service."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .corpus import CODE, Corpus, normalized
from .search import MODES, Search, evidence

DEFAULT_MODE = "rerank"
SEMESTER = re.compile(r"(?:semester\s*(\d+)|(\d+)\.?\s*(?:dönem|donem|semester))")
AGGREGATE = re.compile(r"\b(ects|akts|how many|kaç|kac|count|total|toplam)\b")
OUTSIDE = re.compile(r"\b(tuition|enrollment deadline|visa|scholarship|vize|burs|harç)\b")


@dataclass(frozen=True)
class Answer:
    text: str
    records: tuple = ()
    values: dict | None = None


class CourseIntelligence:
    def __init__(self, mode=None):
        self.mode = mode or os.getenv("COURSE_RETRIEVAL", DEFAULT_MODE)
        if self.mode not in MODES:
            raise ValueError(f"Unknown retrieval mode: {self.mode}")
        self.corpus = Corpus()
        self.index = None

    def answer(self, question):
        result = self.evaluate(question)
        sources = [
            f"[{row.code} · {row.department} · {row.id}]" for row in result.records
        ]
        if not sources:
            return result.text
        return "\n".join([result.text, "", *sources, result.records[0].source])

    def evaluate(self, question):
        query = normalized(question)
        departments = [name for name in self.corpus.departments if normalized(name) in query]
        aliases = {
            "yazilim muhendisligi": "Software Engineering",
            "bilgisayar muhendisligi": "Computer Engineering",
            "endustri muhendisligi": "Industrial Engineering",
            "elektrik elektronik muhendisligi": "Electrical and Electronics Engineering",
        }
        translated = query.translate(str.maketrans("çğöşü", "cgosu"))
        for alias, name in aliases.items():
            if alias in translated and name in self.corpus.departments and name not in departments:
                departments.append(name)
        match = SEMESTER.search(query)
        semester = int(match.group(1) or match.group(2)) if match else None
        kind = None
        if re.search(r"mandatory|required|zorunlu", query):
            kind = "mandatory"
        elif re.search(r"elective|seçmeli", query):
            kind = "elective"
        allowed = self.corpus.select(departments, semester, kind)
        codes = {f"{m[1].upper()} {m[2].upper()}" for m in CODE.finditer(question)}
        if codes:
            allowed = [row for row in allowed if row.code in codes]
            if not allowed:
                return Answer("No matching course in this curriculum snapshot.")
        if OUTSIDE.search(query):
            return Answer("The curriculum snapshot does not contain that information.")
        if AGGREGATE.search(query) and (departments or codes):
            if not allowed:
                return Answer("No matching records for those filters.")
            values = {}
            for row in allowed:
                group = values.setdefault(row.department, {"count": 0, "ects": 0, "unknown_ects": 0})
                group["count"] += 1
                if row.ects is None:
                    group["unknown_ects"] += 1
                else:
                    group["ects"] += row.ects
            lines = [
                f"{name}: {value['count']} records, {value['ects']} known ECTS "
                f"({value['unknown_ects']} records have unknown ECTS)."
                for name, value in sorted(values.items())
            ]
            return Answer("\n".join(lines), tuple(allowed), values)
        if codes:
            return Answer("\n".join(
                f"{row.code} — {row.name} ({row.department})"
                + (f", semester {row.semester}" if row.semester is not None else "")
                for row in allowed
            ), tuple(allowed))
        if self.index is None:
            self.index = Search(self.corpus, cache=Path(".cache"), dense=self.mode != "bm25")
        excerpts = evidence(question, self.index.search(question, self.mode, allowed))
        if not excerpts:
            return Answer("I could not verify an answer in the curriculum snapshot.")
        return Answer("\n".join(f"{row.code} — {quote}" for row, quote in excerpts),
                      tuple(row for row, _ in excerpts))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Search the IEU curriculum snapshot")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES)
    args = parser.parse_args()
    print(CourseIntelligence(args.mode).answer(args.question))


if __name__ == "__main__":
    main()

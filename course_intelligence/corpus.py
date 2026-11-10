"""Read the committed curriculum snapshot and retain stable record provenance."""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/raw/iue_curriculum.json"
SNAPSHOT_URL = (
    "https://github.com/onurerguden/IEU-Chat-Bot/blob/"
    "7a6f90e40a2c560d4dd0197c46a12e6a17cf91fd/data/raw/iue_curriculum.json"
)
CODE = re.compile(r"\b([A-ZÇĞİÖŞÜ]{2,5})[ -]?(\d{3,4}[A-Z]?)\b", re.IGNORECASE)


def normalized(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return value.replace("i\u0307", "i").replace("ı", "i")


def tokens(value):
    return re.findall(r"[^\W_]+", normalized(value))


@dataclass(frozen=True)
class Course:
    id: str
    department: str
    code: str
    name: str
    semester: int | None
    kind: str
    ects: int | float | None
    text: str

    @property
    def source(self):
        return SNAPSHOT_URL


class Corpus:
    def __init__(self, path=SNAPSHOT):
        raw = path.read_bytes()
        self.sha = hashlib.sha256(raw).hexdigest()
        self.records = {}
        for row in json.loads(raw):
            department = str(row.get("department") or "").strip()
            code = " ".join(str(row.get("course_code") or "").upper().split())
            if not department or not code:
                continue
            semester = row.get("semester")
            semester = int(semester) if str(semester).isdigit() else None
            key = (department, semester, code)
            id_ = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
            kind = str(row.get("type") or "")
            ects = row.get("ects")
            ects = ects if type(ects) in (int, float) else None
            name = str(row.get("course_name") or code)
            text = "\n".join(str(item) for item in (
                f"{code} {name} | {department} | {kind}", row.get("objectives"),
                row.get("description"), row.get("prerequisites"), row.get("weekly_topics")
            ) if item)
            course = Course(id_, department, code, name, semester, kind, ects, text)
            if id_ in self.records and self.records[id_] != course:
                raise ValueError(f"Conflicting snapshot entries for {key}")
            self.records[id_] = course
        self.departments = sorted({row.department for row in self.records.values()})

    def select(self, departments=(), semester=None, kind=None):
        return [row for row in self.records.values()
                if (not departments or row.department in departments)
                and (semester is None or row.semester == semester)
                and (kind is None or normalized(kind) in normalized(row.kind))]


class BM25:
    """Sparse BM25 with explicit corpus statistics shared by every query."""

    def __init__(self, texts):
        from math import log
        self.documents = [Counter(tokens(text)) for text in texts]
        self.average_length = sum(map(lambda doc: sum(doc.values()), self.documents)) / len(texts)
        frequency = Counter(word for doc in self.documents for word in doc)
        self.vocabulary = {word: i for i, word in enumerate(sorted(frequency))}
        size = len(self.documents)
        self.idf = {word: log(1 + (size - n + 0.5) / (n + 0.5)) for word, n in frequency.items()}

    def document(self, index):
        doc = self.documents[index]
        norm = 1.2 * (0.25 + 0.75 * sum(doc.values()) / self.average_length)
        return {self.vocabulary[word]: (2.2 * count) / (count + norm)
                for word, count in doc.items()}

    def query(self, text):
        return {self.vocabulary[word]: self.idf[word]
                for word in set(tokens(text)) if word in self.vocabulary}

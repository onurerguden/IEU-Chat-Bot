import json
from collections import Counter

from course_intelligence.app import CourseIntelligence
from course_intelligence.corpus import ROOT
from course_intelligence.search import Search


def test_frozen_fixture_and_structured_answers():
    app = CourseIntelligence()
    fixture = json.loads((ROOT / "evaluation/curriculum_questions.json").read_text())
    assert fixture["snapshot_sha256"] == app.corpus.sha
    cases = fixture["questions"]
    assert Counter(case["split"] for case in cases) == {"dev": 24, "test": 56}
    for case in cases:
        assert set(case["expected"]) <= app.corpus.records.keys()
        if case["kind"] == "search":
            continue
        answer = app.evaluate(case["question"])
        assert {row.id for row in answer.records} == set(case["expected"]), case["id"]
        assert answer.values == case.get("values"), case["id"]


def test_bm25_respects_program_and_code():
    app = CourseIntelligence()
    allowed = app.corpus.select(["Software Engineering"])
    index = Search(app.corpus, dense=False)
    matches = index.search("SE 226 Advanced Programming", "bm25", allowed)
    assert matches[0].code == "SE 226"
    assert all(row.department == "Software Engineering" for row in matches)


def test_model_can_only_select_supplied_excerpt_ids():
    from types import SimpleNamespace

    from course_intelligence.search import evidence

    row = next(row for row in CourseIntelligence().corpus.records.values()
               if row.code == "SE 226" and row.department == "Software Engineering")

    def reply(**request):
        offered = request["format"]["properties"]["excerpt_ids"]["items"]["enum"]
        assert 1 in offered
        return SimpleNamespace(message=SimpleNamespace(content='{"excerpt_ids":[999,"1",1]}'))

    assert evidence("advanced programming", [row], SimpleNamespace(chat=reply)) == [
        (row, row.text.splitlines()[0][:240])
    ]

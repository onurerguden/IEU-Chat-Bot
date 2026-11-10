"""Evaluate fixed record judgments and preserve per-query evidence."""

import argparse
import json
import math
import platform
import random
import resource
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from .app import CourseIntelligence
from .corpus import ROOT, Corpus
from .search import DEVICE, EMBEDDING, MODEL, MODES, REVISIONS, Search, evidence, snapshot


def summary(rows):
    return {
        "n": len(rows),
        "recall_at_10": statistics.mean(row["recall"] for row in rows),
        "ndcg_at_10": statistics.mean(row["ndcg"] for row in rows),
        "p50_ms": statistics.median(row["ms"] for row in rows),
        "p95_ms": sorted(row["ms"] for row in rows)[math.ceil(len(rows) * .95) - 1],
    }


def rank_metrics(found, expected):
    relevant = set(expected)
    dcg = sum(1 / math.log2(i + 2) for i, id_ in enumerate(found[:10]) if id_ in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return len(set(found[:10]) & relevant) / len(relevant), dcg / ideal


def paired_interval(baseline, candidate):
    group_scores = defaultdict(list)
    for left, right in zip(baseline, candidate, strict=True):
        assert left["id"] == right["id"]
        group_scores[left["group"]].append(right["ndcg"] - left["ndcg"])
    pairs = [statistics.mean(items) for items in group_scores.values()]
    rng = random.Random(2027)
    simulations = sorted(statistics.mean(rng.choices(pairs, k=len(pairs))) for _ in range(2000))
    return {"groups": len(pairs), "mean": statistics.mean(pairs),
            "percentile_95": [simulations[50], simulations[1949]]}


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1048576 if sys.platform == "darwin" else 1024)


def score_evidence(corpus, case, hits):
    start = time.perf_counter()
    quotes = evidence(case["question"], [corpus.records[id_] for id_ in hits])
    ids = {row.id for row, _ in quotes}
    return {"id": case["id"], "kind": "evidence",
            "expected_record_selected": bool(ids & set(case["expected"])),
            "accepted_excerpts": [{"id": row.id, "quote": quote} for row, quote in quotes],
            "ms": (time.perf_counter() - start) * 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--answers", action="store_true", help="Use local Ollama to score evidence selection")
    parser.add_argument("--evidence-replay", type=Path, help="Re-evaluate generation on frozen retrieval hits")
    parser.add_argument("--legacy", action="store_true", help="Run the pre-existing FAISS path as a separate reference")
    parser.add_argument("--profile-batch", type=int, choices=[1, 8], help="Measure one embedding batch size in a fresh process")
    args = parser.parse_args()
    corpus = Corpus()
    fixtures = json.loads((ROOT / "evaluation/curriculum_questions.json").read_text())
    assert fixtures["snapshot_sha256"] == corpus.sha
    cases = fixtures["questions"]
    if args.evidence_replay:
        frozen = json.loads(args.evidence_replay.read_text())
        if frozen["snapshot_sha256"] != corpus.sha:
            parser.error("The frozen results use another snapshot.")
        mode = frozen["selected"]
        hits = {row["id"]: row["hits"] for row in frozen["modes"][mode]["queries"]}
        rows = []
        for case in cases:
            if case["kind"] != "search" or case["split"] != "test":
                continue
            rows.append(score_evidence(corpus, case, hits[case["id"]]))
        args.output.write_text(json.dumps({"snapshot_sha256": corpus.sha, "retrieval_mode": mode,
                                           "generator": MODEL, "answer_checks": rows},
                                          indent=2, ensure_ascii=False) + "\n")
        return

    modes = args.mode or MODES
    start = time.perf_counter()
    if args.profile_batch:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(snapshot(EMBEDDING), device=DEVICE, truncate_dim=512)
        model.max_seq_length = 768
        sample = [row for department in corpus.departments for row in corpus.select([department])[:16]]
        texts = [row.text for row in sample]
        model.encode(texts[:1], batch_size=1, normalize_embeddings=True)
        samples = []
        for offset in range(0, len(texts), args.profile_batch):
            start = time.perf_counter()
            batch = texts[offset:offset + args.profile_batch]
            model.encode(batch, batch_size=args.profile_batch, normalize_embeddings=True)
            samples.append((time.perf_counter() - start) * 1000)
        result = {"batch_size": args.profile_batch, "hardware": platform.platform(), "model": EMBEDDING,
                  "device": DEVICE, "revision": REVISIONS[EMBEDDING], "documents": len(texts), "throughput_per_second": len(texts) / (sum(samples) / 1000),
                  "p50_batch_ms": statistics.median(samples),
                  "p95_batch_ms": sorted(samples)[math.ceil(len(samples) * .95) - 1],
                  "peak_process_mib": peak_mb()}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return

    index = Search(corpus, cache=ROOT / ".cache", dense=any(mode != "bm25" for mode in modes))
    startup = time.perf_counter() - start
    app = CourseIntelligence()
    by_mode = {}
    runs = {}
    for mode in modes:
        rows = []
        for case in cases:
            if case["kind"] != "search":
                continue
            allowed = corpus.select(case["departments"])
            start = time.perf_counter()
            hits = index.search(case["question"], mode, allowed)
            elapsed = (time.perf_counter() - start) * 1000
            ids = [row.id for row in hits]
            recall, ndcg = rank_metrics(ids, case["expected"])
            rows.append({**case, "hits": ids, "recall": recall, "ndcg": ndcg, "ms": elapsed})
        print(f"{mode}: scored {len(rows)} retrieval queries", flush=True)
        runs[mode] = rows
        test = [row for row in rows if row["split"] == "test"]
        by_mode[mode] = {
            "dev": summary([row for row in rows if row["split"] == "dev"]),
            "test": summary(test),
            "test_slices": {key: summary([row for row in test if row[field] == key])
                            for field, keys in (("language", ("en", "tr")), ("slice", ("code", "semantic")))
                            for key in keys},
            "queries": rows,
        }
    choose = lambda modes_: max(modes_, key=lambda mode: (
        by_mode[mode]["dev"]["ndcg_at_10"], by_mode[mode]["dev"]["recall_at_10"]))
    basic = [mode for mode in ("bm25", "dense") if mode in runs]
    advanced = [mode for mode in ("hybrid", "rerank") if mode in runs]
    selected = choose(basic or list(runs))
    comparison = None
    if basic and advanced:
        candidate = choose(advanced)
        ref, trial = by_mode[selected]["test"], by_mode[candidate]["test"]
        comparison = {"baseline": selected, "candidate": candidate,
                      "paired_test_ndcg": paired_interval(
                          [row for row in runs[selected] if row["split"] == "test"],
                          [row for row in runs[candidate] if row["split"] == "test"])}
        if trial["recall_at_10"] >= ref["recall_at_10"] and trial["ndcg_at_10"] > ref["ndcg_at_10"]:
            selected = candidate

    outcomes = []
    for case in cases:
        if case["split"] != "test":
            continue
        if case["kind"] in ("exact", "negative"):
            answer = app.evaluate(case["question"])
            supported = set(row.id for row in answer.records) == set(case["expected"])
            outcomes.append({"id": case["id"], "kind": case["kind"], "correct": supported and
                             (answer.values == case["values"] if case["kind"] == "exact" else not answer.records)})
        elif args.answers:
            sample = next(row for row in runs[selected] if row["id"] == case["id"])
            outcomes.append(score_evidence(corpus, case, sample["hits"]))

    legacy = None
    if args.legacy:
        from rag import rag_app
        rag_app.EMBED_MODEL = snapshot(rag_app.EMBED_MODEL)
        legacy_model = rag_app.FAISSRAG()
        results = []
        identities = {(row.department, row.code): row.id for row in corpus.records.values()}
        for case in cases:
            if case["split"] != "test" or case["kind"] != "search":
                continue
            start = time.perf_counter()
            docs = legacy_model.retrieve(case["question"], top_k=10)
            elapsed = (time.perf_counter() - start) * 1000
            ids = list(dict.fromkeys(identities.get((doc.get("metadata", {}).get("department"),
                                                     doc.get("metadata", {}).get("course_code"))) for doc in docs))
            recall, ndcg = rank_metrics([id_ for id_ in ids if id_], case["expected"])
            results.append({"id": case["id"], "recall": recall, "ndcg": ndcg, "ms": elapsed})
        legacy = {"summary": summary(results), "queries": results}
    output = {"snapshot_sha256": corpus.sha, "hardware": platform.platform(),
              "python": platform.python_version(), "device": DEVICE, "model_revisions": REVISIONS,
              "generator": MODEL if args.answers else None, "startup_seconds": startup,
              "peak_process_mib": peak_mb(), "modes": by_mode, "selected": selected,
              "comparison": comparison, "answer_checks": outcomes, "legacy_reference": legacy}
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

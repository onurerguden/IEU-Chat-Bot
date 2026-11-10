# Course Intelligence RAG

Ask questions about an İzmir University of Economics engineering curriculum snapshot in a desktop app or terminal. My contribution covers the data pipeline, shared application service, controlled retrieval benchmark and evidence display.

[Desktop demo](docs/demo.mp4) · [Evaluation and failure analysis](evaluation/README.md) · [Frozen benchmark questions](evaluation/curriculum_questions.json)

```sh
python -m course_intelligence.app 'Find SE 226 in Software Engineering'
```

## Run locally

From the repository root, use Python 3.11 and install the requirements in a virtual environment. Exact code lookup and credit totals run from the snapshot. Open-ended questions use local Ollama. Model weights are downloaded separately.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3:4b-instruct-2507-q4_K_M
python gui_app.py
```

To try the same service from the terminal:

```sh
python -m course_intelligence.app 'Compare total mandatory ECTS in semester 4 in Computer Engineering and Software Engineering'
python -m course_intelligence.app 'What is the tuition for Software Engineering?'
python -m course_intelligence.app --mode rerank 'Which Software Engineering course covers architectural design?'
```

The first neural run downloads pinned Qwen embedding and reranking weights. Set `COURSE_RETRIEVAL` to `bm25`, `dense`, `hybrid` or `rerank` for the desktop app. The CPU is the default. Apple Silicon users can set `COURSE_DEVICE=mps`. Ollama listens on loopback by default. `OLLAMA_HOST` overrides its address.

## How the application answers

Exact course-code matches use stable record IDs. Counts and ECTS use all records matching the requested department, semester and type. Missing credits are reported separately. For open-ended questions, retrieval returns candidate records and the local model selects excerpts. The service displays an excerpt only if its record ID and literal text match the supplied evidence. The snapshot link and record IDs identify the evidence used. The snapshot has no academic-year field.

The Qdrant index is local and built from the committed snapshot. It uses the same records, filters and candidate limit for sparse BM25, dense cosine, RRF and the reranked variant. The original FAISS/SBERT implementation remains available as a separate reference.

## Reproduce the evaluation

```sh
python -m pip install pytest
python -m pytest -q
python -m course_intelligence.benchmark --legacy --answers --output /tmp/course-results.json
python -m course_intelligence.benchmark --profile-batch 1 --output /tmp/course-batch1.json
python -m course_intelligence.benchmark --profile-batch 8 --output /tmp/course-batch8.json
```

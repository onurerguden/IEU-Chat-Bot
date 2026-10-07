# Course Intelligence RAG

A local retrieval-augmented generation assistant for İzmir University of Economics engineering curricula. It retrieves course evidence across departments before asking Llama 3.1 to answer, with filters for department, year, semester and course type.

Built by [Onur Ergüden](https://onurerguden.dev). [Read the case study](https://onurerguden.dev/en/projects/course-intelligence).

## What it demonstrates

- Hierarchical course chunks, curriculum summaries and metadata-aware retrieval.
- Sentence-transformer embeddings and a FAISS inner-product index over normalized vectors.
- Query processing for comparisons, course counts and ECTS totals across departments.
- Prompt constraints intended to reject questions that the retrieved curriculum cannot support.
- A PyQt6 desktop interface and a command-line entry point.

## Architecture

```text
Curriculum JSON → hierarchical chunks → SBERT embeddings → FAISS index
Question → query filters → retrieved context → Llama 3.1 via Ollama → answer
```

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`; the generation model is `llama3.1:8b`.

## Run locally

Use a Python environment compatible with PyQt6, FAISS and sentence-transformers; Python 3.12 is a suitable starting point. Install [Ollama](https://ollama.com) and keep its local service running.

```sh
git clone https://github.com/onurerguden/IEU-Chat-Bot.git
cd IEU-Chat-Bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install faiss-cpu ollama PyQt6
ollama pull llama3.1:8b
python gui_app.py
```

On Windows, activate with `.venv\Scripts\activate`. The extra installation line supplies runtime imports missing from the current `requirements.txt`. The first run downloads the embedding model; local generation requires enough memory for the 8B model.

The repository includes a FAISS index and metadata. To rebuild them from the committed curriculum snapshot, run from the repository root:

```sh
python chunking/chunk_courses.py
python embedding/embed_chunks.py
python vector_db/build_faiss.py
```

For a terminal interface:

```sh
python rag/rag_app.py
```

## Evaluation

```sh
python evaluation/evaluate_rag.py
```

The evaluator covers qualitative, quantitative and trap questions and writes `evaluation_results_enhanced.json` in the working directory. [Committed evaluation results](evaluation/evaluation_results_enhanced.json) are available for inspection.

The evaluation uses project-specific answer heuristics. Its success labels describe this question set; they do not establish general accuracy or guarantee that unsupported answers never occur. Re-running with a different model or curriculum snapshot may change the results.

## Repository guide

| Path | Purpose |
| --- | --- |
| `data/raw/` | Curriculum snapshot |
| `chunking/` | Course and summary chunk construction |
| `embedding/` | Embedding generation |
| `vector_db/` | Index builder and stored FAISS artifacts |
| `rag/rag_app.py` | Query processing, retrieval and generation |
| `gui_app.py` | Desktop interface |
| `evaluation/` | Question set, evaluator and saved outputs |

This is an academic project using a curriculum snapshot. Check the university's current official course information before making academic decisions.

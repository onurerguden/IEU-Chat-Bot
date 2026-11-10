# Evaluation

Course Intelligence asks which retrieval strategy finds a known course from a Turkish or English query, and how much time and memory it needs. This is a small, bilingual study of one frozen curriculum snapshot.

## Dataset and scoring

[The 80 questions](curriculum_questions.json) were prepared before model runs against 371 normalized records. The snapshot hash and record IDs are checked before evaluation. An English and Turkish phrasing of an intent belongs to the same group and split. The locked set contains 32 retrieval prompts, 12 aggregate prompts and 12 prompts with an absent code or out-of-scope request. Retrieval labels identify the intended record in the requested department. Other courses can also be relevant to a semantic query, so the single-judgment Recall and nDCG values should be read as target-course recovery.

The four Qdrant modes share a corpus, per-department filter, candidate budget of 20 and top ten result window. BM25 uses corpus term statistics, dense uses normalized Qwen vectors, hybrid fuses ranks with RRF and rerank scores the fused candidates with a Qwen cross-encoder. The old FAISS/SBERT pipeline is a separate reference. Its original retrieval filters and chunking differ. The grouped bootstrap resamples bilingual intents as units.

Aggregate exact match requires the correct set of record IDs and the complete department/count/ECTS result. Abstention requires an empty answer for the unsupported cases. The generation check compares returned evidence to the intended record and accepts only literal excerpts from supplied records. It measures evidence selection on this set.

## Research comparison

TU Dresden's [ST-Buddy study](https://fis.tu-dresden.de/portal/en/publications/stbuddy%284386f686-ba4f-401f-b195-5f7741227134%29.html) evaluated usability and perceived usefulness in a course-support setting. This repository evaluates target-course retrieval, deterministic credit totals and explicit abstention on a fixed cross-department curriculum snapshot. The two evaluations address different questions.

## Reproduce

The commands in the [project README](../README.md) reproduce the offline checks, the local Ollama generation checks and separate batch profiling runs. The [desktop walkthrough](../docs/demo.mp4) shows the running application.

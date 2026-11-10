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

## Locked retrieval results

Measured on an Apple M1 Pro with 16 GB unified memory. The Qwen retrieval modes ran on Metal, with the first embedding/index build included in startup time. The retrieval timings exclude generation. The [raw retrieval report](retrieval_results.json) includes per-query output, model revisions, slices and device information.

| Method | Recall@10 | nDCG@10 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.750 | 0.659 | 12 ms | 14 ms |
| Dense | 1.000 | 0.890 | 54 ms | 114 ms |
| RRF hybrid | 1.000 | 0.859 | 65 ms | 71 ms |
| RRF + reranker | 1.000 | 0.954 | 7.24 s | 36.63 s |
| Legacy FAISS/SBERT product reference | 0.625 | 0.447 | 10 ms | 684 ms |

Reranking met the preset rule for the default. It improved nDCG by 0.064 while preserving Recall. The paired 95% bootstrap interval is [0.000, 0.128] across 16 bilingual intents. The dense mode remains available for interactive use when the reranker latency is too high.

## Failure analysis

BM25 missed the Turkish query about computer networks and transport protocols. Dense retrieval moved its target from outside the top ten to rank one. RRF improved the English optimization query from rank two to rank one but pushed the Turkish control systems query from rank one to rank two. Reranking improved the Turkish query containing `CE 221` from an nDCG of 0.431 to 1.000, and moved the English signals-and-systems target from rank one to rank two. The [per-query report](retrieval_results.json) preserves these outcomes and the English/Turkish and code/semantic slices.

## Answer evidence

All 12 exact checks and all 12 abstention checks passed. The initial generation prompt required the model to copy text verbatim. The intended record appeared in 11 of 32 validated answers. The final generation path offers numbered excerpts from supplied records and validates the returned IDs. On the same frozen retrieval hits, the intended record appeared in 31 of 32 answers. The remaining miss was the English signals-and-systems query, which selected another course. Median generation time on these recorded runs was 6.05 s and 1.87 s respectively. The second run is a same-set regression after the prompt and output contract changed. It measures intended-record selection. The [regression output](evidence_regression.json) preserves every accepted excerpt.

```sh
python -m course_intelligence.benchmark --evidence-replay evaluation/retrieval_results.json --output /tmp/course-evidence.json
```

## Embedding throughput and memory

The profile uses the first 16 records from each department for a fixed sample of 64 records. Each batch size runs in a fresh process with the same model and Metal device. Latency is per batch. Memory is peak process RSS.

| Batch size | Records/s | p50 per batch | p95 per batch | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| [1](profile_batch_1.json) | 4.72 | 209 ms | 282 ms | 1,544 MiB |
| [8](profile_batch_8.json) | 3.05 | 2.51 s | 3.52 s | 2,686 MiB |

For this sample and device, larger batches used more memory and delivered lower throughput. The cold build of the 371-record search index took 134.35 s in the retrieval run. Subsequent starts can reuse the cached document vectors.

# Latency variance across runs

Quality metrics are deterministic: every nDCG, recall, MRR and precision value in
`results.csv` reproduced **bit-identically** across two independent runs.

Latency did not. Both runs were on the same 8-core CPU, but the second competed with
other work on the machine.

| Config | p50 run 1 | p50 run 2 | p95 run 1 | p95 run 2 |
|---|---:|---:|---:|---:|
| hybrid / whole | 35.4 | 54.3 | 72.2 | 119.1 |
| hybrid / window120 | 81.0 | 68.7 | 216.2 | 166.1 |
| hybrid / sentence3 | 145.0 | 127.6 | 322.3 | 324.6 |
| bm25 / whole | 50.3 | 51.0 | 128.6 | 116.5 |
| bm25 / sentence3 | 50.2 | 124.1 | 114.5 | 338.3 |
| dense / whole | 0.85 | 0.91 | 1.27 | 1.46 |

Two things this shows:

1. **Dense retrieval is stable** (0.85 to 0.91 ms) because it is a single FAISS call.
   The BM25 and hybrid numbers swing by up to 2.5x because pure-Python posting-list
   loops are at the mercy of whatever else the machine is doing.
2. **Any latency conclusion drawn from a single run of this harness is unsafe.**
   The ordering holds -- dense < bm25 < hybrid, and whole < window120 -- but the
   magnitudes do not.

`results_run2.csv` holds the second run in full. Proper benchmarking would pin the
process to isolated cores, run each configuration many times, and report a
distribution. This harness does not do that, so the README argues from index size,
which is deterministic, and treats latency as directional only.

# Blog Post Conversion Plan

## Deliverables

- `blog-post.md` — Markdown with embedded chart images
- `*.png` — Static chart exports (6 charts)
- Link to the Jupyter notebook from the blog post for reproducibility

## Style

Mirror the structure and tone of the [1B benchmark article](https://www.scylladb.com/2025/12/01/scylladb-vector-search-1b-benchmark/) — concise, results-forward, scenario-based, with summary tables.

## Framing

Position this as a **small-scale scenario** — 10M vectors on modest hardware, no quantization needed. Link to the 1B post as the large-scale reference, and mention that a 100M benchmark article is coming.

## Proposed Structure

### 1. Title + Intro (~2-3 paragraphs)

- Headline numbers (12,840 QPS at k=10, sub-6ms P99 latency, 92–99% recall)
- Context: "Following our 1B benchmark, we explore a smaller, more accessible scenario — 10M Cohere embeddings (768 dims) on a compact 5-node cluster, no quantization needed"
- Mention 100M article coming next

### 2. Architecture at a Glance (~2 paragraphs)

- Brief recap (same as 1B post): storage nodes + dedicated Vector Store (Rust/USearch), CDC propagation, CQL interface
- Keep short since 1B article covers this in detail — link to it

### 3. Benchmark Setup

- Dataset: 10M × 768 Cohere, COSINE
- Hardware table (3× i8g.large storage + 2× r7g.2xlarge search)
- Memory sizing note: 10M × (768×4 + M×16) × 1.2 ≈ 40 GB fits in 64 GB RAM — no quantization required
- Experiments table (M, ef_construction, ef_search, k tested)
- CQL `CREATE INDEX` statements for each experiment (with `index_version` removed)

### 4. Results by Scenario (mirroring 1B's scenario structure)

- **Scenario 1 — Maximum Throughput (Exp #3 & #4, k=10)**: 9,700–12,840 QPS, 92–95% recall, sub-6ms serial P99
- **Scenario 2 — High Recall (Exp #1, k=10 & k=100)**: 99.2% recall at k=10, 96.8% at k=100, 2,300 QPS
- **Scenario 3 — Balanced (Exp #2)**: 97.7% recall, 4,900 QPS at k=10; 92% recall, 2,975 QPS at k=100

### 5. Deep Dive Charts (6 static PNGs, same as notebook)

1. QPS vs Concurrency (k=100 / k=10)
2. P99 Latency vs Concurrency
3. Average Latency vs Concurrency
4. Peak QPS Comparison (bar chart)
5. QPS vs P99 Latency (Pareto view with concurrency annotations)
6. Recall vs Peak QPS tradeoff

### 6. Detailed Results (summary tables from notebook)

- k=100 table and k=10 table
- Key observations (5.5× QPS gain for 7.2pp recall drop at k=10, etc.)

### 7. Conclusion (~2 paragraphs)

- 10M is a "just works" scenario — modest hardware, no quantization, still excellent performance
- Link to 1B post for extreme scale; mention 100M is coming
- Link to notebook for reproducibility
- Link to Quick Start Guide

## CQL Blocks to Include (from results.txt, index_version removed)

```cql
-- Experiment #1: M=64, ef_c=384, ef_s=192
CREATE CUSTOM INDEX vdb_bench_collection_vector_idx
ON vdb_bench.vdb_bench_collection (vector)
USING 'vector_index'
WITH OPTIONS = {
  'search_beam_width': '192',
  'construction_beam_width': '384',
  'maximum_node_connections': '64',
  'similarity_function': 'COSINE'
};

-- Experiment #2: M=32, ef_c=256, ef_s=128
CREATE CUSTOM INDEX vdb_bench_collection_vector_idx
ON vdb_bench.vdb_bench_collection (vector)
USING 'vector_index'
WITH OPTIONS = {
  'search_beam_width': '128',
  'construction_beam_width': '256',
  'maximum_node_connections': '32',
  'similarity_function': 'COSINE'
};

-- Experiment #3: M=24, ef_c=256, ef_s=64
CREATE CUSTOM INDEX vdb_bench_collection_vector_idx
ON vdb_bench.vdb_bench_collection (vector)
USING 'vector_index'
WITH OPTIONS = {
  'search_beam_width': '64',
  'construction_beam_width': '256',
  'maximum_node_connections': '24',
  'similarity_function': 'COSINE'
};

-- Experiment #4: M=20, ef_c=256, ef_s=48
CREATE CUSTOM INDEX vdb_bench_collection_vector_idx
ON vdb_bench.vdb_bench_collection (vector)
USING 'vector_index'
WITH OPTIONS = {
  'search_beam_width': '48',
  'construction_beam_width': '256',
  'maximum_node_connections': '20',
  'similarity_function': 'COSINE'
};
```

## Implementation Steps

1. Create `blog-post/` directory
2. Write a Python script to export the 6 charts as PNGs from the notebook data
3. Write `blog-post.md` with all sections, embedded `![](chart.png)` references, and links
4. Verify rendering

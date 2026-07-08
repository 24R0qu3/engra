# Retrieval evaluation harness

A small, dependency-free harness for measuring engra's retrieval quality over time.
Use it to compare a baseline retriever against a change (e.g. a future hybrid
dense + keyword retriever) on a fixed set of queries with known-good answers.

## Running

```bash
python tests/eval/run_eval.py
```

The script imports `engra.commands._data_search` and runs every query in
`golden.jsonl` against the **live index** on your machine. If the index is empty
it prints a message and exits `0` (a skip, not a failure), so it is safe to run in
environments without a corpus.

## Files

- `golden.jsonl` — one JSON object per line:
  - `query` — the search string.
  - `expect_filename` — basename of the document that should be retrieved.
  - `expect_page` — expected page (int), or `null` to accept any page of that file.
  - `note` — human hint describing what the query targets.
- `run_eval.py` — loads the golden set, runs retrieval, prints the metrics table.
  The metric math is in pure, importable functions (`recall_at_k`,
  `reciprocal_rank`, `aggregate`, ...) that need no index.

## Metrics

For each query the retriever returns a ranked list; each result is marked relevant
when its filename (and page, when `expect_page` is set) matches the golden record.

- **recall@k** — fraction of queries whose target appears in the top *k* results
  (reported for k = 1, 3, 5, 10). Higher is better.
- **MRR** (mean reciprocal rank) — average of `1 / rank` of the first relevant
  result per query (0 if the target is absent). Rewards ranking the right document
  near the top.

## Curating `golden.jsonl`

The expectations here are **domain-realistic placeholders**, not verified against a
specific index. Before trusting the numbers you **must** curate the golden set
against your live index: confirm each `expect_filename` matches an indexed
document's basename, and tighten `expect_page` where you know the exact page.

The query set deliberately mixes:

- **Concept queries** — natural language that dense retrieval handles well.
- **Exact-token queries** — specific symbols, PGN/DDI/SPN/error codes, part numbers,
  and section identifiers that pure dense retrieval often misses. These make the
  gain from a future hybrid (keyword-aware) retriever visible in the metrics.

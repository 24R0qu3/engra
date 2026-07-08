"""Retrieval-evaluation harness for engra.

Loads a curated golden set, runs each query through engra.commands._data_search,
and reports recall@k and MRR. The metric math lives in the pure functions below so
it can be unit-tested without an index (see tests/test_eval_metrics.py).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

GOLDEN_PATH = Path(__file__).with_name("golden.jsonl")
KS = (1, 3, 5, 10)


def load_golden(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def is_relevant(result: dict, record: dict) -> bool:
    got = os.path.basename(str(result.get("filename", ""))).lower()
    want = os.path.basename(str(record.get("expect_filename", ""))).lower()
    if got != want:
        return False
    expect_page = record.get("expect_page")
    if expect_page is None:
        return True
    return result.get("page") == expect_page


def first_relevant_rank(relevances: list[bool]) -> int | None:
    """1-based rank of the first relevant result, or None if none are relevant."""
    for idx, hit in enumerate(relevances, start=1):
        if hit:
            return idx
    return None


def recall_at_k(relevances: list[bool], k: int) -> float:
    """Fraction of this query's single target found in the top k (0.0 or 1.0)."""
    return 1.0 if any(relevances[:k]) else 0.0


def reciprocal_rank(relevances: list[bool]) -> float:
    rank = first_relevant_rank(relevances)
    return 0.0 if rank is None else 1.0 / rank


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(per_query_relevances: list[list[bool]], ks: tuple[int, ...]) -> dict:
    return {
        "recall": {k: mean([recall_at_k(r, k) for r in per_query_relevances]) for k in ks},
        "mrr": mean([reciprocal_rank(r) for r in per_query_relevances]),
        "n": len(per_query_relevances),
    }


def _print_table(metrics: dict, ks: tuple[int, ...]) -> None:
    print(f"\nEvaluated {metrics['n']} queries\n")
    header = "  ".join(f"R@{k:<5}" for k in ks) + "  MRR"
    print(header)
    row = "  ".join(f"{metrics['recall'][k]:<6.3f}" for k in ks) + f"  {metrics['mrr']:.3f}"
    print(row)


def main() -> int:
    try:
        from engra.commands import _data_search
    except Exception as exc:  # pragma: no cover - import environment guard
        print(f"Could not import engra.commands._data_search: {exc}", file=sys.stderr)
        return 0

    records = load_golden(GOLDEN_PATH)
    if not records:
        print("No golden records found; nothing to evaluate.")
        return 0

    max_k = max(KS)
    probe = _data_search("connectivity probe", top_k=1, projects=[])
    if not probe:
        print(
            "Index appears empty (a probe query returned no results). "
            "Index the corpus first, then curate golden.jsonl against it. Skipping.",
        )
        return 0

    per_query: list[list[bool]] = []
    for record in records:
        results = _data_search(record["query"], top_k=max_k, projects=[])
        per_query.append([is_relevant(r, record) for r in results])

    metrics = aggregate(per_query, KS)
    _print_table(metrics, KS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

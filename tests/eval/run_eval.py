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
    expect_pages = record.get("expect_pages")
    if expect_pages:
        return result.get("page") in expect_pages
    expect_page = record.get("expect_page")
    if expect_page is None:
        return True
    return result.get("page") == expect_page


def is_tight(record: dict) -> bool:
    """True when ground truth pins down specific page(s), making precision meaningful.

    Loose (whole-file-relevance) records inflate precision -- any page in the right
    file counts as a hit -- and would hide noise-padding effects the tight subset
    is designed to surface.
    """
    return bool(record.get("expect_pages")) or record.get("expect_page") is not None


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


def precision_at_k(relevances: list[bool], k: int) -> float:
    """Fraction of the first k returned results that are relevant.

    Denominator is the actual window size (<=k), so a query returning fewer than
    k hits isn't penalised for hits that don't exist. Mirrors recall_at_k's
    slicing style; returns 0.0 for an empty window (mean([]) == 0.0).
    """
    return mean(relevances[:k])


def aggregate(per_query_relevances: list[list[bool]], ks: tuple[int, ...]) -> dict:
    return {
        "recall": {k: mean([recall_at_k(r, k) for r in per_query_relevances]) for k in ks},
        "precision": {k: mean([precision_at_k(r, k) for r in per_query_relevances]) for k in ks},
        "mrr": mean([reciprocal_rank(r) for r in per_query_relevances]),
        "n": len(per_query_relevances),
    }


def _print_table(label: str, metrics: dict, ks: tuple[int, ...]) -> None:
    print(f"\n{label} -- {metrics['n']} queries\n")
    header = (
        "  ".join(f"R@{k:<5}" for k in ks) + "  " + "  ".join(f"P@{k:<5}" for k in ks) + "  MRR"
    )
    print(header)
    recall_cells = "  ".join(f"{metrics['recall'][k]:<6.3f}" for k in ks)
    precision_cells = "  ".join(f"{metrics['precision'][k]:<6.3f}" for k in ks)
    print(f"{recall_cells}  {precision_cells}  {metrics['mrr']:.3f}")


def _print_per_query(
    records: list[dict],
    before: list[list[bool]],
    after: list[list[bool]],
    ks: tuple[int, ...],
) -> None:
    print("\nPer-query precision, tight (page-verified) records only:\n")
    for record, before_rel, after_rel in zip(records, before, after):
        before_str = "  ".join(f"{precision_at_k(before_rel, k):.3f}" for k in ks)
        after_str = "  ".join(f"{precision_at_k(after_rel, k):.3f}" for k in ks)
        print(f"  {record['query']!r}")
        print(f"    before: {before_str}")
        print(f"    after:  {after_str}")


def main() -> int:
    try:
        from engra.commands import _data_search, _trim_by_confidence
    except Exception as exc:  # pragma: no cover - import environment guard
        print(f"Could not import engra.commands: {exc}", file=sys.stderr)
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

    # rerank=True matches the MCP engra_search default (the consumer this harness
    # is meant to represent) -- the raw _data_search default is False for
    # unrelated callers (CLI), so it must be passed explicitly here.
    per_query: list[list[bool]] = []
    for record in records:
        results = _data_search(record["query"], top_k=max_k, projects=[], rerank=True)
        per_query.append([is_relevant(r, record) for r in results])

    metrics = aggregate(per_query, KS)
    _print_table("All queries (recall/MRR unaffected by confidence trimming)", metrics, KS)

    tight_records = [r for r in records if is_tight(r)]
    if not tight_records:
        print(
            "\nNo page-verified (expect_page/expect_pages) records; precision above is "
            "loose (whole-file) and won't show confidence-trimming effects.",
        )
        return 0

    before: list[list[bool]] = []
    after: list[list[bool]] = []
    for record in tight_records:
        raw = _data_search(record["query"], top_k=max_k, projects=[], rerank=True)
        before.append([is_relevant(h, record) for h in raw])
        after.append([is_relevant(h, record) for h in _trim_by_confidence(raw)])

    _print_table(
        f"Page-verified queries, BEFORE confidence trim (n={len(tight_records)})",
        aggregate(before, KS),
        KS,
    )
    _print_table(
        f"Page-verified queries, AFTER confidence trim (n={len(tight_records)})",
        aggregate(after, KS),
        KS,
    )
    _print_per_query(tight_records, before, after, KS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

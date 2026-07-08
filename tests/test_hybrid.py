"""Tests for hybrid retrieval: RRF fusion, the FTS5 keyword index, and mode dispatch."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from engra import storage
from engra.commands import (
    _reciprocal_rank_fusion,
    _where_to_fts_sql,
)


# ── Reciprocal Rank Fusion (pure, no index) ─────────────────────────────────────


def test_rrf_empty():
    assert _reciprocal_rank_fusion([]) == []
    assert _reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list_preserves_order():
    fused = _reciprocal_rank_fusion([["a", "b", "c"]])
    assert [cid for cid, _ in fused] == ["a", "b", "c"]
    # scores strictly decreasing with rank
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_rewards_agreement_across_lists():
    # 'b' is high in both lists → should win over items appearing once.
    fused = _reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]])
    assert fused[0][0] == "b"
    order = [cid for cid, _ in fused]
    assert order.index("c") < order.index("d")  # c seen twice, d once


def test_rrf_k_matches_formula():
    fused = dict(_reciprocal_rank_fusion([["x"]], k=60))
    assert fused["x"] == pytest.approx(1.0 / 61)


def test_rrf_tiebreak_is_stable_by_id():
    # Both ids rank 1 in their own list → equal score → sorted by id ascending.
    fused = _reciprocal_rank_fusion([["zzz"], ["aaa"]])
    assert [cid for cid, _ in fused] == ["aaa", "zzz"]


# ── where → FTS5 SQL translation ────────────────────────────────────────────────


def test_where_translate_none():
    assert _where_to_fts_sql(None) is None
    assert _where_to_fts_sql({}) is None


def test_where_translate_single_project():
    assert _where_to_fts_sql({"project": "docs"}) == "project = 'docs'"


def test_where_translate_in():
    assert _where_to_fts_sql({"project": {"$in": ["a", "b"]}}) == "project IN ('a', 'b')"


def test_where_translate_and():
    sql = _where_to_fts_sql({"$and": [{"project": "p"}, {"filename": "f.pdf"}]})
    assert sql == "(project = 'p' AND filename = 'f.pdf')"


def test_where_translate_escapes_quotes():
    assert _where_to_fts_sql({"project": "O'Brien"}) == "project = 'O''Brien'"


# ── FTS5 storage CRUD + search ──────────────────────────────────────────────────


def _require_fts():
    if storage.get_fts_connection() is None:
        pytest.skip("SQLite FTS5 not available on this build")


def test_fts_add_and_search_exact_token():
    _require_fts()
    storage.fts_add(
        [
            ("c1", "hydraulic pump fault code P0420", "d1", "proj", "f.pdf"),
            ("c2", "brake caliper assembly", "d1", "proj", "f.pdf"),
        ]
    )
    hits = storage.fts_search("P0420")
    assert [h["chunk_id"] for h in hits] == ["c1"]


def test_fts_search_query_with_special_chars_does_not_crash():
    _require_fts()
    storage.fts_add([("c1", "diagnostic code DD-1234 and PGN 65262", "d1", "proj", "f.pdf")])
    # Punctuation in the query would break a raw MATCH; the quoting layer must survive it.
    assert storage.fts_search("DD-1234") != []  # tokenizer splits, phrase still matches
    assert storage.fts_search('unmatched "') == storage.fts_search("nomatchhere")


def test_fts_search_where_scoping():
    _require_fts()
    storage.fts_add(
        [
            ("c1", "shared widget term", "d1", "projA", "a.pdf"),
            ("c2", "shared widget term", "d2", "projB", "b.pdf"),
        ]
    )
    hits = storage.fts_search("widget", where="project = 'projA'")
    assert [h["chunk_id"] for h in hits] == ["c1"]


def test_fts_delete_by_ids():
    _require_fts()
    storage.fts_add([("c1", "alpha", "d1", "p", "f.pdf"), ("c2", "alpha", "d1", "p", "f.pdf")])
    storage.fts_delete_by_ids(["c1"])
    assert [h["chunk_id"] for h in storage.fts_search("alpha")] == ["c2"]


def test_fts_delete_by_doc_id():
    _require_fts()
    storage.fts_add([("c1", "beta", "d1", "p", "f.pdf"), ("c2", "beta", "d2", "p", "g.pdf")])
    storage.fts_delete_by_doc_id("d1")
    assert [h["chunk_id"] for h in storage.fts_search("beta")] == ["c2"]


def test_fts_delete_by_project():
    _require_fts()
    storage.fts_add([("c1", "gamma", "d1", "pA", "f.pdf"), ("c2", "gamma", "d2", "pB", "g.pdf")])
    storage.fts_delete_by_project("pA")
    assert [h["chunk_id"] for h in storage.fts_search("gamma")] == ["c2"]


def test_fts_update_project():
    _require_fts()
    storage.fts_add([("c1", "delta", "d1", "old", "f.pdf")])
    storage.fts_update_project(["c1"], "new")
    assert storage.fts_search("delta", where="project = 'new'") != []
    assert storage.fts_search("delta", where="project = 'old'") == []


# ── _data_search mode dispatch ──────────────────────────────────────────────────


def _model():
    m = MagicMock()
    m.query_embed.return_value = iter([np.zeros(128)])
    return m


META = {
    "source": "/tmp/doc.pdf",
    "filename": "doc.pdf",
    "page": 1,
    "page_label": "1",
    "total_pages": 3,
    "chunk": 0,
    "project": "proj",
}


def test_dense_mode_skips_keyword_index(monkeypatch):
    import engra.commands as cmd

    col = MagicMock()
    col.count.return_value = 1
    col.get.return_value = {"ids": ["id0"]}
    col.query.return_value = {
        "documents": [["dense hit"]],
        "metadatas": [[META]],
        "distances": [[0.1]],
        "ids": [["id0"]],
    }
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "read_session", lambda: [])
    # If dense mode wrongly consulted the keyword arm this would raise.
    monkeypatch.setattr(
        cmd, "fts_search", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called"))
    )

    results = cmd._data_search("q", mode="dense")
    assert len(results) == 1
    assert results[0]["retrieval"] == "dense"


def test_hybrid_surfaces_keyword_only_hit(monkeypatch):
    import engra.commands as cmd

    kw_meta = {**META, "page": 2, "filename": "manual.pdf"}
    col = MagicMock()
    col.count.return_value = 2
    col.query.return_value = {
        "documents": [["dense hit"]],
        "metadatas": [[META]],
        "distances": [[0.1]],
        "ids": [["id_dense"]],
    }

    def _get(ids=None, where=None, include=None, limit=None):
        if ids is not None:  # keyword-only metadata fetch
            return {
                "ids": ["id_kw"],
                "documents": ["exact token P0420 here"],
                "metadatas": [kw_meta],
                "embeddings": [np.zeros(128).tolist()],
            }
        return {"ids": ["id_dense", "id_kw"]}

    col.get.side_effect = _get
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "read_session", lambda: [])
    monkeypatch.setattr(
        cmd,
        "fts_search",
        lambda q, where=None, limit=15: [
            {
                "chunk_id": "id_kw",
                "text": "exact token P0420 here",
                "doc_id": "d",
                "project": "proj",
                "filename": "manual.pdf",
                "score": -6.0,
            }
        ],
    )

    results = cmd._data_search("P0420", mode="hybrid")
    texts = {r["text"] for r in results}
    assert "exact token P0420 here" in texts  # dense missed it, keyword surfaced it
    kw_hit = next(r for r in results if r["text"] == "exact token P0420 here")
    assert kw_hit["retrieval"] == "keyword"


def test_invalid_mode_raises(monkeypatch):
    import engra.commands as cmd

    col = MagicMock()
    col.count.return_value = 1
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "read_session", lambda: [])
    with pytest.raises(ValueError):
        cmd._data_search("q", mode="bogus")

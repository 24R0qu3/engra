"""Tests for _data_* retrieval functions and the MCP tool dispatch layer."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _col(metas, docs=None, ids=None):
    """Build a minimal chromadb Collection mock."""
    docs = docs or ["text"] * len(metas)
    ids = ids or [f"id{i}" for i in range(len(metas))]
    col = MagicMock()
    col.count.return_value = len(metas)
    col.get.return_value = {"metadatas": metas, "documents": docs, "ids": ids}
    col.query.return_value = {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [[0.1] * len(metas)],
        "ids": [ids],
    }
    return col


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
    "indexed_at": "2026-01-01T00:00:00+00:00",
    "source_mtime": 1234567890.0,
    "model": "intfloat/multilingual-e5-large",
    "chunk_size": 1500,
    "chunk_overlap": 200,
}


# ── _data_search ───────────────────────────────────────────────────────────────


def test_data_search_empty_db(monkeypatch):
    import engra.commands as cmd

    col = MagicMock()
    col.count.return_value = 0
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    from engra.commands import _data_search

    assert _data_search("anything") == []


def test_data_search_returns_correct_shape(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_search

    col = _col([META], ["some text"])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    results = _data_search("query", top_k=5)
    assert len(results) == 1
    r = results[0]
    for key in (
        "filename",
        "page",
        "page_label",
        "total_pages",
        "chunk",
        "score",
        "text",
        "project",
    ):
        assert key in r, f"missing key: {key}"


def test_data_search_min_score_filters(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_search

    metas = [META, {**META, "page": 2}]
    docs = ["high relevance", "low relevance"]
    # distances: 0.05 → score 0.95, 0.8 → score 0.2
    col = _col(metas, docs)
    col.query.return_value = {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [[0.05, 0.8]],
        "ids": [["id0", "id1"]],
    }
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    results = _data_search("q", min_score=0.5)
    assert len(results) == 1
    assert results[0]["text"] == "high relevance"
    assert results[0]["score"] >= 0.5


def test_data_search_uses_session_when_projects_none(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_search

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "read_session", lambda: ["proj_a"])

    _data_search("q", projects=None)

    call_kwargs = col.query.call_args[1]
    assert call_kwargs.get("where") == {"project": "proj_a"}


def test_data_search_empty_projects_searches_globally(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_search

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "read_session", lambda: ["proj_a"])

    _data_search("q", projects=[])

    call_kwargs = col.query.call_args[1]
    assert "where" not in call_kwargs  # no filter


# ── _data_get_chunks ───────────────────────────────────────────────────────────


def test_data_get_chunks_returns_sorted(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_get_chunks

    m0 = {**META, "chunk": 0, "page_label": "1"}
    m1 = {**META, "chunk": 1, "page_label": "1"}
    col = MagicMock()
    col.get.return_value = {"metadatas": [m1, m0], "documents": ["b", "a"]}
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    chunks = _data_get_chunks("doc.pdf", 1, 1)
    assert [c["chunk_idx"] for c in chunks] == [0, 1]


def test_data_get_chunks_missing_page_returns_empty(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_get_chunks

    col = MagicMock()
    col.get.return_value = {"metadatas": [], "documents": []}
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    assert _data_get_chunks("doc.pdf", 99, 99) == []


def test_data_get_chunks_shape(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_get_chunks

    col = MagicMock()
    col.get.return_value = {"metadatas": [META], "documents": ["hello"]}
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    chunks = _data_get_chunks("doc.pdf", 1, 1)
    assert len(chunks) == 1
    for key in ("filename", "page", "page_label", "chunk_idx", "text", "source"):
        assert key in chunks[0]


# ── _data_get_neighbors ────────────────────────────────────────────────────────


def _seq_col(n_pages=3, chunks_per_page=2):
    """Build a collection mock with a predictable (page, chunk) sequence."""
    metas = [
        {**META, "page": p, "chunk": c, "page_label": str(p)}
        for p in range(1, n_pages + 1)
        for c in range(chunks_per_page)
    ]
    col = MagicMock()

    def _get(where=None, include=None):
        if where and "filename" in where:
            # _get_chunk_sequence call: return all metas for the file
            return {"metadatas": metas, "documents": [], "ids": []}
        # per-chunk fetch
        and_clauses = where.get("$and", [])
        page = next((c["page"] for c in and_clauses if "page" in c), None)
        chunk = next((c["chunk"] for c in and_clauses if "chunk" in c), None)
        matched = [m for m in metas if m["page"] == page and m.get("chunk") == chunk]
        docs = ["text"] * len(matched)
        return {"metadatas": matched, "documents": docs, "ids": []}

    col.get.side_effect = _get
    return col


def test_data_get_neighbors_next(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_get_neighbors

    col = _seq_col(n_pages=3, chunks_per_page=1)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    result = _data_get_neighbors("doc.pdf", page=1, chunk=0, direction="next", count=1)
    assert len(result) == 1
    assert result[0]["page"] == 2


def test_data_get_neighbors_prev(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_get_neighbors

    col = _seq_col(n_pages=3, chunks_per_page=1)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    result = _data_get_neighbors("doc.pdf", page=3, chunk=0, direction="prev", count=1)
    assert len(result) == 1
    assert result[0]["page"] == 2


def test_data_get_neighbors_both(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_get_neighbors

    col = _seq_col(n_pages=5, chunks_per_page=1)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    # anchor = page 3, direction=both count=1 → page 2 and page 4
    result = _data_get_neighbors("doc.pdf", page=3, chunk=0, direction="both", count=1)
    pages = [r["page"] for r in result]
    assert 2 in pages
    assert 4 in pages
    assert len(result) == 2


def test_data_get_neighbors_boundary_start(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_get_neighbors

    col = _seq_col(n_pages=3, chunks_per_page=1)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    result = _data_get_neighbors("doc.pdf", page=1, chunk=0, direction="prev", count=1)
    assert result == []


def test_data_get_neighbors_anchor_not_found(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_get_neighbors

    col = _seq_col(n_pages=3, chunks_per_page=1)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    result = _data_get_neighbors("doc.pdf", page=99, chunk=99, direction="next", count=1)
    assert result == []


def test_data_get_neighbors_invalid_direction(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_get_neighbors

    col = _seq_col()
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    with pytest.raises(ValueError, match="direction"):
        _data_get_neighbors("doc.pdf", page=1, chunk=0, direction="sideways")


# ── _data_list_projects ────────────────────────────────────────────────────────


def test_data_list_projects_empty(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_list_projects

    col = MagicMock()
    col.count.return_value = 0
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    assert _data_list_projects() == []


def test_data_list_projects_aggregates(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_list_projects

    metas = [
        {**META, "project": "proj_a", "filename": "a.pdf"},
        {**META, "project": "proj_a", "filename": "a.pdf"},
        {**META, "project": "proj_b", "filename": "b.pdf"},
    ]
    col = _col(metas)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    result = _data_list_projects()
    names = {p["name"] for p in result}
    assert names == {"proj_a", "proj_b"}
    proj_a = next(p for p in result if p["name"] == "proj_a")
    assert proj_a["chunk_count"] == 2
    assert proj_a["file_count"] == 1


# ── _data_list_files ───────────────────────────────────────────────────────────


def test_data_list_files_shape(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_list_files

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    result = _data_list_files()
    assert len(result) == 1
    for key in ("filename", "project", "chunks", "pages", "indexed_at", "stale_status", "source"):
        assert key in result[0]


def test_data_list_files_project_filter(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_list_files

    metas = [
        {**META, "project": "proj_a", "source": "/a.pdf", "filename": "a.pdf"},
        {**META, "project": "proj_b", "source": "/b.pdf", "filename": "b.pdf"},
    ]
    col = _col(metas)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    result = _data_list_files(project="proj_a")
    assert len(result) == 1
    assert result[0]["project"] == "proj_a"


# ── _data_info ─────────────────────────────────────────────────────────────────


def test_data_info_empty(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_info

    col = MagicMock()
    col.count.return_value = 0
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    d = _data_info()
    assert d["total_chunks"] == 0


def test_data_info_shape(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_info

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    d = _data_info()
    for key in ("total_chunks", "files", "projects", "model", "chunk_size", "last_indexed"):
        assert key in d


# ── _data_project_activate / _data_project_deactivate ─────────────────────────


@pytest.fixture
def patched_session(tmp_path, monkeypatch):
    from engra import storage

    monkeypatch.setattr(storage, "STATE_FILE", tmp_path / "state.toml")
    return tmp_path / "state.toml"


def test_data_project_activate_valid(monkeypatch, patched_session):
    import engra.commands as cmd
    from engra.commands import _data_project_activate

    metas = [{**META, "project": "iso_7"}]
    col = _col(metas)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    r = _data_project_activate(["iso_7"])
    assert "iso_7" in r["active_projects"]
    assert r["unknown"] == []


def test_data_project_activate_unknown(monkeypatch, patched_session):
    import engra.commands as cmd
    from engra.commands import _data_project_activate

    metas = [{**META, "project": "iso_7"}]
    col = _col(metas)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    r = _data_project_activate(["nonexistent"])
    assert "nonexistent" in r["unknown"]
    # Session should not have been written with the unknown project
    assert "nonexistent" not in r["active_projects"]


def test_data_project_deactivate(monkeypatch, patched_session):
    import engra.commands as cmd
    from engra.commands import _data_project_activate, _data_project_deactivate

    metas = [{**META, "project": "p"}]
    col = _col(metas)
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    _data_project_activate(["p"])

    r = _data_project_deactivate()
    assert r == {"active_projects": []}


# ── MCP dispatch layer ─────────────────────────────────────────────────────────


def _import_mcp_server():
    """Import mcp_server with the mcp package stubbed out."""
    import sys
    import types

    # Stub the mcp package so mcp_server.py can be imported without it installed
    mcp_stub = types.ModuleType("mcp")
    server_stub = types.ModuleType("mcp.server")
    stdio_stub = types.ModuleType("mcp.server.stdio")
    types_stub = types.ModuleType("mcp.types")

    class FakeServer:
        def __init__(self, name):
            self.name = name
            self._list_tools_fn = None
            self._call_tool_fn = None

        def list_tools(self):
            def decorator(fn):
                self._list_tools_fn = fn
                return fn

            return decorator

        def call_tool(self):
            def decorator(fn):
                self._call_tool_fn = fn
                return fn

            return decorator

        def create_initialization_options(self):
            return {}

    class TextContent:
        def __init__(self, type, text):
            self.type = type
            self.text = text

    class Tool:
        def __init__(self, name, description="", inputSchema=None):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema or {}

    server_stub.Server = FakeServer
    stdio_stub.stdio_server = None
    types_stub.TextContent = TextContent
    types_stub.Tool = Tool
    mcp_stub.server = server_stub
    mcp_stub.types = types_stub

    sys.modules.setdefault("mcp", mcp_stub)
    sys.modules.setdefault("mcp.server", server_stub)
    sys.modules.setdefault("mcp.server.stdio", stdio_stub)
    sys.modules.setdefault("mcp.types", types_stub)

    # Force reimport with stubs in place
    sys.modules.pop("engra.mcp_server", None)
    import engra.mcp_server as ms

    return ms


def test_mcp_list_tools_returns_all_eleven():
    import asyncio

    ms = _import_mcp_server()
    tools = asyncio.run(ms.server._list_tools_fn())
    names = {t.name for t in tools}
    expected = {
        "engra_search",
        "engra_get_chunk",
        "engra_get_neighbors",
        "engra_list_projects",
        "engra_list_files",
        "engra_index",
        "engra_info",
        "engra_project_activate",
        "engra_project_deactivate",
        "engra_project_describe",
        "engra_project_autodescribe",
    }
    assert names == expected


def test_mcp_call_tool_search_round_trips(monkeypatch):
    import asyncio

    import engra.commands as cmd

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    ms = _import_mcp_server()
    results = asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "test"}))
    assert len(results) == 1
    data = json.loads(results[0].text)
    # Should be a list of result dicts (possibly empty if score filter cuts them)
    assert isinstance(data, list)


def test_mcp_call_tool_unknown_returns_error():
    import asyncio

    ms = _import_mcp_server()
    results = asyncio.run(ms.server._call_tool_fn("not_a_tool", {}))
    data = json.loads(results[0].text)
    assert "error" in data
    assert "not_a_tool" in data["error"]


def test_mcp_call_tool_exception_returns_error(monkeypatch):
    import asyncio

    import engra.commands as cmd

    monkeypatch.setattr(
        cmd, "_data_search", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("db down"))
    )

    ms = _import_mcp_server()
    results = asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "q"}))
    data = json.loads(results[0].text)
    assert "error" in data
    assert "db down" in data["error"]


def test_mcp_list_files_dispatch(monkeypatch):
    import asyncio

    import engra.commands as cmd

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    ms = _import_mcp_server()
    results = asyncio.run(ms.server._call_tool_fn("engra_list_files", {}))
    data = json.loads(results[0].text)
    assert isinstance(data, list)


def test_mcp_info_dispatch(monkeypatch):
    import asyncio

    import engra.commands as cmd

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    ms = _import_mcp_server()
    results = asyncio.run(ms.server._call_tool_fn("engra_info", {}))
    data = json.loads(results[0].text)
    assert "total_chunks" in data


def test_mcp_list_projects_dispatch(monkeypatch):
    import asyncio

    import engra.commands as cmd

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    ms = _import_mcp_server()
    results = asyncio.run(ms.server._call_tool_fn("engra_list_projects", {}))
    data = json.loads(results[0].text)
    assert isinstance(data, list)

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
        def __init__(self, name, description="", inputSchema=None, annotations=None):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema or {}
            self.annotations = annotations

    class ToolAnnotations:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class CallToolResult:
        def __init__(self, content, isError=False):
            self.content = content
            self.isError = isError

    server_stub.Server = FakeServer
    stdio_stub.stdio_server = None
    types_stub.TextContent = TextContent
    types_stub.Tool = Tool
    types_stub.ToolAnnotations = ToolAnnotations
    types_stub.CallToolResult = CallToolResult
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


def test_mcp_list_tools_returns_all_twelve():
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
        "engra_list_members",
        "engra_index",
        "engra_info",
        "engra_project_activate",
        "engra_project_deactivate",
        "engra_project_describe",
        "engra_project_autodescribe",
    }
    assert names == expected


def test_mcp_search_schema_exposes_follow_links():
    import asyncio

    ms = _import_mcp_server()
    tools = asyncio.run(ms.server._list_tools_fn())
    search_tool = next(t for t in tools if t.name == "engra_search")
    assert "follow_links" in search_tool.inputSchema["properties"]
    assert search_tool.inputSchema["properties"]["follow_links"]["default"] is False


def test_mcp_read_only_tools_carry_read_only_hint():
    import asyncio

    ms = _import_mcp_server()
    tools = asyncio.run(ms.server._list_tools_fn())
    read_only = {
        "engra_search",
        "engra_get_chunk",
        "engra_get_neighbors",
        "engra_list_projects",
        "engra_list_files",
        "engra_list_members",
        "engra_info",
    }
    for tool in tools:
        if tool.name in read_only:
            assert tool.annotations.readOnlyHint is True, tool.name
        else:
            assert tool.annotations.readOnlyHint is False, tool.name


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
    # engra_search now returns a {results, not_found} envelope, not a bare list.
    assert isinstance(data, dict)
    assert isinstance(data["results"], list)
    assert "not_found" in data


def test_mcp_call_tool_unknown_returns_error():
    import asyncio

    ms = _import_mcp_server()
    result = asyncio.run(ms.server._call_tool_fn("not_a_tool", {}))
    assert result.isError is True
    data = json.loads(result.content[0].text)
    assert "error" in data
    assert "not_a_tool" in data["error"]
    assert data["tool"] == "not_a_tool"


def test_mcp_call_tool_exception_returns_error(monkeypatch):
    import asyncio

    import engra.commands as cmd

    monkeypatch.setattr(
        cmd, "_data_search", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("db down"))
    )

    ms = _import_mcp_server()
    result = asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "q"}))
    assert result.isError is True
    data = json.loads(result.content[0].text)
    assert "error" in data
    assert "db down" in data["error"]
    assert data["tool"] == "engra_search"


def test_mcp_list_files_dispatch(monkeypatch):
    import asyncio

    import engra.commands as cmd

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)

    ms = _import_mcp_server()
    results = asyncio.run(ms.server._call_tool_fn("engra_list_files", {}))
    data = json.loads(results[0].text)
    assert isinstance(data["items"], list)
    assert data["total"] == len(data["items"])
    assert data["offset"] == 0
    assert data["limit"] == 50


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


# ── DEFAULT_MIN_SCORE applied by engra_search (Feature 1) ────────────────────


def test_mcp_search_default_min_score_applied(monkeypatch):
    """When min_score is not passed, engra_search should use DEFAULT_MIN_SCORE."""
    import asyncio

    import engra.commands as cmd
    from engra.commands import DEFAULT_MIN_SCORE

    captured = {}

    def fake_search(**kwargs):
        captured["min_score"] = kwargs.get("min_score")
        return []

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "_data_search", fake_search)

    ms = _import_mcp_server()
    asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "test"}))
    assert captured["min_score"] == DEFAULT_MIN_SCORE


def test_mcp_search_explicit_min_score_overrides(monkeypatch):
    """An explicit min_score of 0.0 should override the default."""
    import asyncio

    import engra.commands as cmd

    captured = {}

    def fake_search(**kwargs):
        captured["min_score"] = kwargs.get("min_score")
        return []

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "_data_search", fake_search)

    ms = _import_mcp_server()
    asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "test", "min_score": 0.0}))
    assert captured["min_score"] == 0.0


# ── engra_search not_found envelope ──────────────────────────────────────────


def test_mcp_search_envelope_found(monkeypatch):
    """Confident results are wrapped in {results, not_found: False}."""
    import asyncio

    import engra.commands as cmd

    def fake_search(**kwargs):
        return [{"text": "a", "score": 0.9, "confidence": 1.0}]

    monkeypatch.setattr(cmd, "_data_search", fake_search)

    ms = _import_mcp_server()
    result = asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "q"}))
    data = json.loads(result[0].text)
    assert data["not_found"] is False
    assert len(data["results"]) == 1
    assert "reason" not in data


def test_mcp_search_envelope_empty_is_not_found(monkeypatch):
    """An empty result set surfaces not_found with a reason."""
    import asyncio

    import engra.commands as cmd

    monkeypatch.setattr(cmd, "_data_search", lambda **kwargs: [])

    ms = _import_mcp_server()
    result = asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "q"}))
    data = json.loads(result[0].text)
    assert data["not_found"] is True
    assert data["results"] == []
    assert isinstance(data["reason"], str) and data["reason"]


def test_mcp_search_envelope_low_confidence_is_not_found(monkeypatch):
    """Results whose best confidence is below DEFAULT_MIN_SCORE are not_found."""
    import asyncio

    import engra.commands as cmd
    from engra.commands import DEFAULT_MIN_SCORE

    low = round(DEFAULT_MIN_SCORE - 0.05, 4)

    def fake_search(**kwargs):
        return [
            {"text": "a", "score": 0.4, "confidence": low},
            {"text": "b", "score": 0.3, "confidence": 0.0},
        ]

    monkeypatch.setattr(cmd, "_data_search", fake_search)

    ms = _import_mcp_server()
    result = asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "q"}))
    data = json.loads(result[0].text)
    assert data["not_found"] is True
    assert len(data["results"]) == 2  # still returned, just flagged
    assert "confidence" in data["reason"]


def test_mcp_search_schema_mentions_not_found():
    """The tool description advertises the not_found envelope."""
    import asyncio

    ms = _import_mcp_server()
    tools = asyncio.run(ms.server._list_tools_fn())
    search_tool = next(t for t in tools if t.name == "engra_search")
    assert "not_found" in search_tool.description


# ── engra_list_members dispatch (Feature 4) ──────────────────────────────────

META_WITH_LABEL = {
    **META,
    "page_label": "MyClass",
    "breadcrumb": "NS > MyClass",
    "cross_refs": "",
}


def test_mcp_list_members_dispatch(monkeypatch):
    import asyncio

    import engra.commands as cmd

    col = _col([META_WITH_LABEL])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    ms = _import_mcp_server()
    results = asyncio.run(ms.server._call_tool_fn("engra_list_members", {"filename": "doc.pdf"}))
    data = json.loads(results[0].text)
    assert isinstance(data["items"], list)
    assert data["total"] == len(data["items"])


def test_mcp_list_members_passes_section_filter(monkeypatch):
    """section_filter argument should be forwarded to _data_list_members."""
    import asyncio

    import engra.commands as cmd

    captured = {}

    def fake_list_members(**kwargs):
        captured.update(kwargs)
        return []

    col = _col([META_WITH_LABEL])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "read_session", lambda: [])
    monkeypatch.setattr(cmd, "_data_list_members", fake_list_members)

    ms = _import_mcp_server()
    asyncio.run(
        ms.server._call_tool_fn(
            "engra_list_members",
            {"filename": "doc.pdf", "section_filter": "My"},
        )
    )
    assert captured.get("section_filter") == "My"
    assert captured.get("filename") == "doc.pdf"


# ── engra_search follow_links wiring ──────────────────────────────────────────


def test_mcp_search_passes_follow_links(monkeypatch):
    import asyncio

    import engra.commands as cmd

    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "_data_search", fake_search)

    ms = _import_mcp_server()
    asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "q", "follow_links": True}))
    assert captured.get("follow_links") is True


def test_mcp_search_follow_links_defaults_false(monkeypatch):
    import asyncio

    import engra.commands as cmd

    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "_data_search", fake_search)

    ms = _import_mcp_server()
    asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "q"}))
    assert captured.get("follow_links") is False


# ── engra_search rerank wiring ────────────────────────────────────────────────


def test_mcp_search_schema_exposes_rerank():
    import asyncio

    ms = _import_mcp_server()
    tools = asyncio.run(ms.server._list_tools_fn())
    search_tool = next(t for t in tools if t.name == "engra_search")
    assert "rerank" in search_tool.inputSchema["properties"]
    assert search_tool.inputSchema["properties"]["rerank"]["default"] is True


def test_mcp_search_rerank_defaults_true_when_omitted(monkeypatch):
    import asyncio

    import engra.commands as cmd

    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "_data_search", fake_search)

    ms = _import_mcp_server()
    asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "q"}))
    assert captured.get("rerank") is True


def test_mcp_search_rerank_explicit_false_respected(monkeypatch):
    import asyncio

    import engra.commands as cmd

    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    col = _col([META])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "_data_search", fake_search)

    ms = _import_mcp_server()
    asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "q", "rerank": False}))
    assert captured.get("rerank") is False


# ── engra_index path allowlist ────────────────────────────────────────────────


def test_index_path_inside_allowlist_accepted(tmp_path):
    ms = _import_mcp_server()

    root = tmp_path / "allowed"
    root.mkdir()
    inside = root / "sub" / "file.txt"
    inside.parent.mkdir()
    inside.write_text("x")

    assert ms._is_path_allowed(inside, [root.resolve()]) is True


def test_index_path_outside_allowlist_rejected(tmp_path):
    ms = _import_mcp_server()

    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside" / "file.txt"
    outside.parent.mkdir()
    outside.write_text("x")

    assert ms._is_path_allowed(outside, [root.resolve()]) is False


def test_index_path_traversal_rejected(tmp_path):
    ms = _import_mcp_server()

    root = tmp_path / "allowed"
    root.mkdir()
    (tmp_path / "outside").mkdir()
    traversal = root / ".." / "outside" / "file.txt"

    assert ms._is_path_allowed(traversal, [root.resolve()]) is False


def test_validate_index_paths_raises_for_disallowed(tmp_path, monkeypatch):
    ms = _import_mcp_server()

    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setattr(ms, "_index_allowlist_roots", lambda: [root.resolve()])

    with pytest.raises(ValueError, match="allowlist"):
        ms._validate_index_paths([tmp_path / "outside.txt"])

    ms._validate_index_paths([root / "ok.txt"])  # does not raise


# ── doc_id disambiguation (same-basename collision) ───────────────────────────


def _match(meta, where):
    """Evaluate a (subset of) chromadb where-clause against a single metadata dict."""
    if not where:
        return True
    if "$and" in where:
        return all(_match(meta, c) for c in where["$and"])
    for key, val in where.items():
        if isinstance(val, dict) and "$in" in val:
            if meta.get(key) not in val["$in"]:
                return False
        elif meta.get(key) != val:
            return False
    return True


def _where_col(metas, docs=None):
    """Collection mock whose .get/.query honour the where-clause (incl. doc_id)."""
    docs = docs or [f"doc{i}" for i in range(len(metas))]
    col = MagicMock()
    col.count.return_value = len(metas)

    def _get(where=None, include=None, limit=None):
        rows = [(m, d, f"id{i}") for i, (m, d) in enumerate(zip(metas, docs)) if _match(m, where)]
        if limit is not None:
            rows = rows[:limit]
        return {
            "metadatas": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "ids": [r[2] for r in rows],
        }

    def _query(**kwargs):
        where = kwargs.get("where")
        n = kwargs.get("n_results", len(metas))
        rows = [(m, d) for m, d in zip(metas, docs) if _match(m, where)][:n]
        return {
            "documents": [[r[1] for r in rows]],
            "metadatas": [[r[0] for r in rows]],
            "distances": [[0.1] * len(rows)],
            "ids": [[f"id{i}" for i in range(len(rows))]],
        }

    col.get.side_effect = _get
    col.query.side_effect = _query
    return col


def _collision_metas():
    """Two documents sharing basename 'report.pdf' in different projects.

    Each document has two chunks (page 1, chunks 0 and 1).
    """
    a = {
        **META,
        "filename": "report.pdf",
        "doc_id": "report.pdf_aaaaaaaa",
        "source": "/projA/report.pdf",
        "project": "projA",
    }
    b = {
        **META,
        "filename": "report.pdf",
        "doc_id": "report.pdf_bbbbbbbb",
        "source": "/projB/report.pdf",
        "project": "projB",
    }
    metas = [
        {**a, "page": 1, "chunk": 0},
        {**a, "page": 1, "chunk": 1},
        {**b, "page": 1, "chunk": 0},
        {**b, "page": 1, "chunk": 1},
    ]
    docs = ["A0", "A1", "B0", "B1"]
    return metas, docs


def test_get_chunks_by_doc_id_isolates_document(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_get_chunks

    metas, docs = _collision_metas()
    monkeypatch.setattr(cmd, "get_collection", lambda: _where_col(metas, docs))

    result = _data_get_chunks("report.pdf", 1, 1, doc_id="report.pdf_aaaaaaaa")
    texts = {c["text"] for c in result}
    assert texts == {"A0", "A1"}  # only document A, never B


def test_get_neighbors_by_doc_id_isolates_document(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_get_neighbors

    metas, docs = _collision_metas()
    monkeypatch.setattr(cmd, "get_collection", lambda: _where_col(metas, docs))

    result = _data_get_neighbors(
        "report.pdf", page=1, chunk=0, direction="next", count=1, doc_id="report.pdf_aaaaaaaa"
    )
    assert len(result) == 1
    assert result[0]["text"] == "A1"  # A's next chunk, not B's


def test_list_members_by_doc_id_isolates_document(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_list_members

    metas, docs = _collision_metas()
    monkeypatch.setattr(cmd, "get_collection", lambda: _where_col(metas, docs))
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    result = _data_list_members("report.pdf", doc_id="report.pdf_bbbbbbbb")
    chunk_texts = {c["text"] for section in result for c in section["chunks"]}
    assert chunk_texts == {"B0", "B1"}  # only document B


def test_ambiguous_filename_without_doc_id_does_not_merge(monkeypatch):
    """Calling by bare filename when it maps to two documents must raise, not merge."""
    import engra.commands as cmd
    from engra.commands import AmbiguousFilenameError, _data_get_chunks

    metas, docs = _collision_metas()
    monkeypatch.setattr(cmd, "get_collection", lambda: _where_col(metas, docs))

    with pytest.raises(AmbiguousFilenameError) as exc_info:
        _data_get_chunks("report.pdf", 1, 1)
    # Both candidate doc_ids are surfaced so the caller can disambiguate
    candidate_ids = {c["doc_id"] for c in exc_info.value.candidates}
    assert candidate_ids == {"report.pdf_aaaaaaaa", "report.pdf_bbbbbbbb"}


def test_list_members_ambiguous_filename_raises(monkeypatch):
    import engra.commands as cmd
    from engra.commands import AmbiguousFilenameError, _data_list_members

    metas, docs = _collision_metas()
    monkeypatch.setattr(cmd, "get_collection", lambda: _where_col(metas, docs))
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    with pytest.raises(AmbiguousFilenameError):
        _data_list_members("report.pdf")


def test_backward_compat_single_doc_without_doc_id(monkeypatch):
    """Legacy chunks lacking a doc_id must still resolve by filename alone."""
    import engra.commands as cmd
    from engra.commands import _data_get_chunks

    # No doc_id key at all — pre-feature index entries
    legacy = [
        {**META, "page": 1, "chunk": 0},
        {**META, "page": 1, "chunk": 1},
    ]
    legacy = [{k: v for k, v in m.items() if k != "doc_id"} for m in legacy]
    docs = ["L0", "L1"]
    monkeypatch.setattr(cmd, "get_collection", lambda: _where_col(legacy, docs))

    result = _data_get_chunks("doc.pdf", 1, 1)  # no doc_id passed
    assert {c["text"] for c in result} == {"L0", "L1"}


def test_backward_compat_missing_doc_id_not_treated_as_ambiguous(monkeypatch):
    """Two legacy chunks (no doc_id) for the same file are one bucket, not ambiguous."""
    from engra.commands import _resolve_doc_scope

    legacy = [
        {"filename": "doc.pdf", "source": "/tmp/doc.pdf", "page": 1, "chunk": 0},
        {"filename": "doc.pdf", "source": "/tmp/doc.pdf", "page": 2, "chunk": 0},
    ]
    col = _where_col(legacy, ["a", "b"])
    # Must not raise, and must fall back to a filename scope
    scope = _resolve_doc_scope(col, "doc.pdf", None)
    assert scope == {"filename": "doc.pdf"}


def test_data_search_includes_doc_id_field(monkeypatch):
    import engra.commands as cmd
    from engra.commands import _data_search

    col = _col([{**META, "doc_id": "doc.pdf_deadbeef"}], ["some text"])
    monkeypatch.setattr(cmd, "get_collection", lambda: col)
    monkeypatch.setattr(cmd, "load_model", _model)
    monkeypatch.setattr(cmd, "read_session", lambda: [])

    results = _data_search("query", top_k=5)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc.pdf_deadbeef"


# ── MCP schemas expose doc_id ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tool_name",
    ["engra_get_chunk", "engra_get_neighbors", "engra_list_members"],
)
def test_mcp_schema_accepts_doc_id(tool_name):
    import asyncio

    ms = _import_mcp_server()
    tools = asyncio.run(ms.server._list_tools_fn())
    tool = next(t for t in tools if t.name == tool_name)
    assert "doc_id" in tool.inputSchema["properties"], tool_name


def test_mcp_get_chunk_passes_doc_id(monkeypatch):
    import asyncio

    import engra.commands as cmd

    captured = {}

    def fake_get_chunks(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cmd, "_data_get_chunks", fake_get_chunks)

    ms = _import_mcp_server()
    asyncio.run(
        ms.server._call_tool_fn(
            "engra_get_chunk",
            {"filename": "report.pdf", "page": 1, "doc_id": "report.pdf_aaaaaaaa"},
        )
    )
    assert captured.get("doc_id") == "report.pdf_aaaaaaaa"


# ── engra_search lean result payload ──────────────────────────────────────────


def _full_hit(**overrides):
    hit = {
        "filename": "doc.pdf",
        "doc_id": "doc.pdf_deadbeef",
        "page": 1,
        "page_label": "1",
        "total_pages": 3,
        "chunk": 0,
        "score": 0.9,
        "text": "hello world",
        "project": "proj",
        "source": "/tmp/doc.pdf",
        "indexed_at": "2026-01-01T00:00:00+00:00",
        "source_mtime": 1234567890.0,
        "notable": False,
        "links_to": "",
        "linked_from": [],
        "breadcrumb": "NS > MyClass",
        "cross_references": [],
        "retrieval": "dense",
        "confidence": 1.0,
    }
    hit.update(overrides)
    return hit


def test_mcp_search_result_is_lean(monkeypatch):
    """Search results are pruned to fields an agent needs to read/cite/navigate."""
    import asyncio

    import engra.commands as cmd

    monkeypatch.setattr(cmd, "_data_search", lambda **kwargs: [_full_hit()])

    ms = _import_mcp_server()
    result = asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "q"}))
    data = json.loads(result[0].text)
    hit = data["results"][0]

    keep = {
        "filename",
        "doc_id",
        "page",
        "page_label",
        "chunk",
        "text",
        "project",
        "score",
        "confidence",
        "retrieval",
        "breadcrumb",
        "links_to",
        "cross_references",
        "notable",
    }
    drop = {"source", "total_pages", "indexed_at", "source_mtime", "linked_from"}

    for field in keep:
        assert field in hit, field
    for field in drop:
        assert field not in hit, field


def test_mcp_search_result_lean_in_not_found_envelope(monkeypatch):
    """Low-confidence results are also pruned before being returned."""
    import asyncio

    import engra.commands as cmd
    from engra.commands import DEFAULT_MIN_SCORE

    low = round(DEFAULT_MIN_SCORE - 0.05, 4)
    monkeypatch.setattr(cmd, "_data_search", lambda **kwargs: [_full_hit(confidence=low)])

    ms = _import_mcp_server()
    result = asyncio.run(ms.server._call_tool_fn("engra_search", {"query": "q"}))
    data = json.loads(result[0].text)
    assert data["not_found"] is True
    assert "source" not in data["results"][0]


# ── engra_list_files / engra_list_members pagination ─────────────────────────


def test_mcp_list_files_pagination(monkeypatch):
    import asyncio

    import engra.commands as cmd

    items = [{"filename": f"f{i}.pdf"} for i in range(5)]
    monkeypatch.setattr(cmd, "_data_list_files", lambda **kwargs: items)

    ms = _import_mcp_server()
    result = asyncio.run(
        ms.server._call_tool_fn("engra_list_files", {"offset": 1, "limit": 2})
    )
    data = json.loads(result[0].text)
    assert data["items"] == items[1:3]
    assert data["total"] == 5
    assert data["offset"] == 1
    assert data["limit"] == 2


def test_mcp_list_members_pagination(monkeypatch):
    import asyncio

    import engra.commands as cmd

    items = [{"section": f"s{i}"} for i in range(5)]
    monkeypatch.setattr(cmd, "_data_list_members", lambda **kwargs: items)

    ms = _import_mcp_server()
    result = asyncio.run(
        ms.server._call_tool_fn(
            "engra_list_members",
            {"filename": "doc.pdf", "offset": 1, "limit": 2},
        )
    )
    data = json.loads(result[0].text)
    assert data["items"] == items[1:3]
    assert data["total"] == 5
    assert data["offset"] == 1
    assert data["limit"] == 2


def test_mcp_list_files_invalid_offset_errors(monkeypatch):
    import asyncio

    import engra.commands as cmd

    monkeypatch.setattr(cmd, "_data_list_files", lambda **kwargs: [])

    ms = _import_mcp_server()
    result = asyncio.run(
        ms.server._call_tool_fn("engra_list_files", {"offset": -1})
    )
    assert result.isError is True


def test_mcp_list_files_invalid_limit_errors(monkeypatch):
    import asyncio

    import engra.commands as cmd

    monkeypatch.setattr(cmd, "_data_list_files", lambda **kwargs: [])

    ms = _import_mcp_server()
    result = asyncio.run(
        ms.server._call_tool_fn("engra_list_files", {"limit": 0})
    )
    assert result.isError is True


def test_mcp_list_files_schema_exposes_pagination():
    import asyncio

    ms = _import_mcp_server()
    tools = asyncio.run(ms.server._list_tools_fn())
    tool = next(t for t in tools if t.name == "engra_list_files")
    assert tool.inputSchema["properties"]["offset"]["default"] == 0
    assert tool.inputSchema["properties"]["limit"]["default"] == 50


def test_mcp_list_members_schema_exposes_pagination():
    import asyncio

    ms = _import_mcp_server()
    tools = asyncio.run(ms.server._list_tools_fn())
    tool = next(t for t in tools if t.name == "engra_list_members")
    assert tool.inputSchema["properties"]["offset"]["default"] == 0
    assert tool.inputSchema["properties"]["limit"]["default"] == 50

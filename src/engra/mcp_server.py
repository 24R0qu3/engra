"""MCP stdio server for engra.

Exposes engra's retrieval capabilities as MCP tools so that AI assistants
can search, retrieve, and index documents without running shell commands.

Install the optional dependency first:
    pip install 'engra[mcp]'

Then register in your MCP client config:
    {"command": "engra", "args": ["mcp"]}
"""

import json
import logging
import sys
from pathlib import Path

try:
    from mcp import types as mcp_types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
except ImportError as _exc:
    raise ImportError(
        "MCP support requires the 'mcp' package. Install with: pip install 'engra[mcp]'"
    ) from _exc

from engra.commands import (
    DEFAULT_MIN_SCORE,
    _data_get_chunks,
    _data_get_neighbors,
    _data_index,
    _data_info,
    _data_list_files,
    _data_list_members,
    _data_list_projects,
    _data_project_activate,
    _data_project_autodescribe,
    _data_project_deactivate,
    _data_project_describe,
    _data_search,
)
from engra.config import init as init_config
from engra.config import load as load_config

logger = logging.getLogger(__name__)

server = Server("engra")


def _annotations(**kwargs) -> "mcp_types.ToolAnnotations | None":
    """Build ToolAnnotations if the installed mcp package supports them.

    Older 1.x releases within the pinned mcp>=1.0,<2.0 range may predate
    ToolAnnotations, so this degrades to None rather than raising.
    """
    if not hasattr(mcp_types, "ToolAnnotations"):
        return None
    return mcp_types.ToolAnnotations(**kwargs)


# ── Tool manifest ─────────────────────────────────────────────────────────────


@server.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="engra_search",
            description=(
                "Semantic search over indexed documents. "
                "Returns ranked chunks with filename, page, score, and text. "
                "projects=null uses the active session; pass [] to search globally."
            ),
            annotations=_annotations(readOnlyHint=True),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "projects": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "description": "Restrict to these projects (null = use active session)",
                    },
                    "top": {"type": "integer", "default": 5, "description": "Max results"},
                    "min_score": {
                        "type": "number",
                        "default": DEFAULT_MIN_SCORE,
                        "description": (
                            f"Minimum similarity score 0–1. Defaults to {DEFAULT_MIN_SCORE} "
                            "to suppress low-confidence matches. Pass 0.0 to see all results."
                        ),
                    },
                    "filename": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Restrict to one file",
                    },
                    "follow_links": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Also append the best chunk from files linked by top results "
                            "(links_to/linked_from/cross_references)"
                        ),
                    },
                    "rerank": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "Re-score the fused candidate pool with a cross-encoder for "
                            "higher precision. Defaults on for MCP search; silently falls "
                            "back to unreranked results if the optional dependency is missing."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["dense", "keyword", "hybrid"],
                        "default": "hybrid",
                        "description": (
                            "Retrieval mode. 'hybrid' (default) fuses vector similarity "
                            "with BM25 keyword matching via Reciprocal Rank Fusion — best "
                            "for exact tokens (part numbers, error/PGN codes, symbol names) "
                            "that pure vector search misses. 'dense' = vector only; "
                            "'keyword' = BM25 only."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="engra_get_chunk",
            description=(
                "Retrieve full text of all chunks on a page (or a specific chunk). "
                "Use after search to read beyond the snippet."
            ),
            annotations=_annotations(readOnlyHint=True),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "page": {"type": "integer", "description": "Physical page number"},
                    "page_end": {
                        "type": ["integer", "null"],
                        "default": None,
                        "description": "End of page range (inclusive)",
                    },
                    "chunk": {
                        "type": ["integer", "null"],
                        "default": None,
                        "description": "Specific chunk index",
                    },
                    "doc_id": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": (
                            "Authoritative document id (from search results). "
                            "Disambiguates when the same filename exists in multiple projects."
                        ),
                    },
                },
                "required": ["filename", "page"],
            },
        ),
        mcp_types.Tool(
            name="engra_get_neighbors",
            description=(
                "Retrieve chunks adjacent to a known position. "
                "Use to expand context around a search hit. "
                "direction: 'next' | 'prev' | 'both'."
            ),
            annotations=_annotations(readOnlyHint=True),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "page": {"type": "integer"},
                    "chunk": {"type": "integer", "description": "Anchor chunk index"},
                    "direction": {
                        "type": "string",
                        "enum": ["next", "prev", "both"],
                        "default": "next",
                    },
                    "count": {
                        "type": "integer",
                        "default": 1,
                        "description": "Chunks to retrieve in each direction",
                    },
                    "doc_id": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": (
                            "Authoritative document id (from search results). "
                            "Disambiguates when the same filename exists in multiple projects."
                        ),
                    },
                },
                "required": ["filename", "page", "chunk"],
            },
        ),
        mcp_types.Tool(
            name="engra_list_projects",
            description=(
                "List all projects in the index with file/chunk counts, "
                "descriptions, and keywords. Call this first to identify which "
                "project to search in."
            ),
            annotations=_annotations(readOnlyHint=True),
            inputSchema={"type": "object", "properties": {}},
        ),
        mcp_types.Tool(
            name="engra_list_files",
            description="List all indexed files with chunk counts and staleness status.",
            annotations=_annotations(readOnlyHint=True),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Filter by project",
                    },
                },
            },
        ),
        mcp_types.Tool(
            name="engra_list_members",
            description=(
                "List all indexed sections for a file, grouped by section heading. "
                "Use for structured browsing or absence-checking "
                "(e.g. 'does class X have a callback for Y?') without a similarity query. "
                "projects=null uses the active session; pass [] to search globally."
            ),
            annotations=_annotations(readOnlyHint=True),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename to browse (as stored in the index)",
                    },
                    "projects": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "description": "Restrict to these projects (null = use active session)",
                    },
                    "section_filter": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Case-insensitive substring to filter section labels",
                    },
                    "doc_id": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": (
                            "Authoritative document id (from search results). "
                            "Disambiguates when the same filename exists in multiple projects."
                        ),
                    },
                },
                "required": ["filename"],
            },
        ),
        mcp_types.Tool(
            name="engra_index",
            description=(
                "Index one or more files or directories. "
                "May take 30+ seconds on first run (model download ~1 GB). "
                "Returns indexed file count, total chunks, and skipped files with reasons. "
                "Automatically generates description and keywords via the configured AI backend."
            ),
            annotations=_annotations(readOnlyHint=False),
            inputSchema={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File or directory paths to index",
                    },
                    "project": {"type": ["string", "null"], "default": None},
                    "force": {
                        "type": "boolean",
                        "default": False,
                        "description": "Re-index even if already present",
                    },
                    "description": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "User-provided project description",
                    },
                    "auto_describe": {
                        "type": "boolean",
                        "default": True,
                        "description": "Generate AI description and keywords for new projects",
                    },
                },
                "required": ["paths"],
            },
        ),
        mcp_types.Tool(
            name="engra_project_describe",
            description="Set the user-provided description and/or keywords for a project.",
            annotations=_annotations(readOnlyHint=False, idempotentHint=True),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name"},
                    "description": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Human-readable description of the project",
                    },
                    "keywords": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "description": "Key terms for this project",
                    },
                },
                "required": ["name"],
            },
        ),
        mcp_types.Tool(
            name="engra_project_autodescribe",
            description=(
                "Generate (or regenerate) AI description and keywords for an existing project "
                "using the configured backend (Ollama or Claude). "
                "Use this to describe projects that were indexed before auto-description was added."
            ),
            annotations=_annotations(readOnlyHint=False, idempotentHint=True),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name"},
                },
                "required": ["name"],
            },
        ),
        mcp_types.Tool(
            name="engra_info",
            description="Return global index statistics: chunk count, file count, model, etc.",
            annotations=_annotations(readOnlyHint=True),
            inputSchema={"type": "object", "properties": {}},
        ),
        mcp_types.Tool(
            name="engra_project_activate",
            description="Activate one or more projects for the current session (8-hour TTL).",
            annotations=_annotations(readOnlyHint=False, idempotentHint=True),
            inputSchema={
                "type": "object",
                "properties": {
                    "names": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["names"],
            },
        ),
        mcp_types.Tool(
            name="engra_project_deactivate",
            description="Clear the active project session; subsequent searches are global.",
            annotations=_annotations(readOnlyHint=False, idempotentHint=True),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ── Tool dispatch ─────────────────────────────────────────────────────────────


@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[mcp_types.TextContent] | mcp_types.CallToolResult:
    try:
        result = _dispatch(name, arguments)
        return [mcp_types.TextContent(type="text", text=json.dumps(result, default=str))]
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        payload = json.dumps({"error": str(exc), "tool": name})
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=payload)],
            isError=True,
        )


def _index_allowlist_roots() -> list[Path]:
    allowlist = load_config().get("mcp", {}).get("index_allowlist") or []
    if not allowlist:
        return [Path.home().resolve()]
    return [Path(root).expanduser().resolve() for root in allowlist]


def _is_path_allowed(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in roots)


def _validate_index_paths(paths: list[Path]) -> None:
    roots = _index_allowlist_roots()
    for p in paths:
        if not _is_path_allowed(p, roots):
            raise ValueError(f"Path not permitted by MCP index allowlist: {p}")


def _dispatch(name: str, args: dict):
    if name == "engra_search":
        return _data_search(
            query=args["query"],
            top_k=args.get("top", 5),
            min_score=args.get("min_score", DEFAULT_MIN_SCORE),
            filename=args.get("filename"),
            projects=args.get("projects"),
            rerank=args.get("rerank", True),
            follow_links=args.get("follow_links", False),
            mode=args.get("mode", "hybrid"),
        )
    elif name == "engra_get_chunk":
        page = args["page"]
        return _data_get_chunks(
            filename=args["filename"],
            page_start=page,
            page_end=args.get("page_end") or page,
            chunk=args.get("chunk"),
            doc_id=args.get("doc_id"),
        )
    elif name == "engra_get_neighbors":
        return _data_get_neighbors(
            filename=args["filename"],
            page=args["page"],
            chunk=args["chunk"],
            direction=args.get("direction", "next"),
            count=args.get("count", 1),
            doc_id=args.get("doc_id"),
        )
    elif name == "engra_list_projects":
        return _data_list_projects()
    elif name == "engra_list_members":
        return _data_list_members(
            filename=args["filename"],
            projects=args.get("projects"),
            section_filter=args.get("section_filter"),
            doc_id=args.get("doc_id"),
        )
    elif name == "engra_list_files":
        return _data_list_files(project=args.get("project"))
    elif name == "engra_index":
        index_paths = [Path(p) for p in args["paths"]]
        _validate_index_paths(index_paths)
        return _data_index(
            paths=index_paths,
            force=args.get("force", False),
            project=args.get("project"),
            description=args.get("description"),
            auto_describe=args.get("auto_describe", True),
        )
    elif name == "engra_project_describe":
        return _data_project_describe(
            name=args["name"],
            description=args.get("description"),
            keywords=args.get("keywords"),
        )
    elif name == "engra_project_autodescribe":
        return _data_project_autodescribe(name=args["name"])
    elif name == "engra_info":
        return _data_info()
    elif name == "engra_project_activate":
        return _data_project_activate(names=args["names"])
    elif name == "engra_project_deactivate":
        return _data_project_deactivate()
    else:
        raise ValueError(f"Unknown tool: {name!r}")


# ── Server entry point ────────────────────────────────────────────────────────


def run_mcp_server() -> None:
    """Start the MCP stdio server.

    Redirects the Rich console to stderr so stdout stays clean for JSON-RPC.
    """
    import asyncio

    from rich.console import Console as RichConsole

    import engra.commands as _cmds

    # stdout is the JSON-RPC channel — redirect all console output to stderr
    _cmds.console = RichConsole(file=sys.stderr, highlight=False)

    async def _main() -> None:
        init_config()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_main())

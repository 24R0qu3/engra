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

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
except ImportError as _exc:
    raise ImportError(
        "MCP support requires the 'mcp' package. "
        "Install with: pip install 'engra[mcp]'"
    ) from _exc

from engra.commands import (
    _data_get_chunks,
    _data_get_neighbors,
    _data_index,
    _data_info,
    _data_list_files,
    _data_list_projects,
    _data_project_activate,
    _data_project_deactivate,
    _data_search,
)
from engra.config import init as init_config

logger = logging.getLogger(__name__)

server = Server("engra")


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
                    "min_score": {"type": "number", "default": 0.0, "description": "Minimum similarity 0–1"},
                    "filename": {"type": ["string", "null"], "default": None, "description": "Restrict to one file"},
                },
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="engra_get_chunk",
            description="Retrieve full text of all chunks on a page (or a specific chunk). Use after search to read beyond the snippet.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "page": {"type": "integer", "description": "Physical page number"},
                    "page_end": {"type": ["integer", "null"], "default": None, "description": "End of page range (inclusive)"},
                    "chunk": {"type": ["integer", "null"], "default": None, "description": "Specific chunk index"},
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
                    "count": {"type": "integer", "default": 1, "description": "Chunks to retrieve in each direction"},
                },
                "required": ["filename", "page", "chunk"],
            },
        ),
        mcp_types.Tool(
            name="engra_list_projects",
            description="List all projects in the index with file and chunk counts.",
            inputSchema={"type": "object", "properties": {}},
        ),
        mcp_types.Tool(
            name="engra_list_files",
            description="List all indexed files with chunk counts and staleness status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": ["string", "null"], "default": None, "description": "Filter by project"},
                },
            },
        ),
        mcp_types.Tool(
            name="engra_index",
            description=(
                "Index one or more files or directories. "
                "May take 30+ seconds on first run (model download ~1 GB). "
                "Returns indexed file count, total chunks, and skipped files with reasons."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File or directory paths to index",
                    },
                    "project": {"type": ["string", "null"], "default": None},
                    "force": {"type": "boolean", "default": False, "description": "Re-index even if already present"},
                },
                "required": ["paths"],
            },
        ),
        mcp_types.Tool(
            name="engra_info",
            description="Return global index statistics: chunk count, file count, model, etc.",
            inputSchema={"type": "object", "properties": {}},
        ),
        mcp_types.Tool(
            name="engra_project_activate",
            description="Activate one or more projects for the current session (8-hour TTL).",
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
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ── Tool dispatch ─────────────────────────────────────────────────────────────


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    try:
        result = _dispatch(name, arguments)
        return [mcp_types.TextContent(type="text", text=json.dumps(result, default=str))]
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return [mcp_types.TextContent(type="text", text=json.dumps({"error": str(exc)}))]


def _dispatch(name: str, args: dict):
    from pathlib import Path

    if name == "engra_search":
        return _data_search(
            query=args["query"],
            top_k=args.get("top", 5),
            min_score=args.get("min_score", 0.0),
            filename=args.get("filename"),
            projects=args.get("projects"),
        )
    elif name == "engra_get_chunk":
        page = args["page"]
        return _data_get_chunks(
            filename=args["filename"],
            page_start=page,
            page_end=args.get("page_end") or page,
            chunk=args.get("chunk"),
        )
    elif name == "engra_get_neighbors":
        return _data_get_neighbors(
            filename=args["filename"],
            page=args["page"],
            chunk=args["chunk"],
            direction=args.get("direction", "next"),
            count=args.get("count", 1),
        )
    elif name == "engra_list_projects":
        return _data_list_projects()
    elif name == "engra_list_files":
        return _data_list_files(project=args.get("project"))
    elif name == "engra_index":
        return _data_index(
            paths=[Path(p) for p in args["paths"]],
            force=args.get("force", False),
            project=args.get("project"),
        )
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

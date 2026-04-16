import os
import warnings

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning, module="fastembed")

import argparse  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from engra import __version__  # noqa: E402
from engra.commands import (  # noqa: E402
    cmd_ask,
    cmd_bookmark_list,
    cmd_bookmark_remove,
    cmd_bookmark_run,
    cmd_bookmark_save,
    cmd_export,
    cmd_get,
    cmd_import,
    cmd_import_soft,
    cmd_index,
    cmd_info,
    cmd_list,
    cmd_mcp,
    cmd_setup_gpu,
    cmd_project_activate,
    cmd_project_active,
    cmd_project_autodescribe,
    cmd_project_deactivate,
    cmd_project_describe,
    cmd_project_list,
    cmd_project_remove,
    cmd_project_rename,
    cmd_remove,
    cmd_search,
    parse_page_range,
)
from engra.config import init as init_config  # noqa: E402
from engra.log import setup as setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def run() -> None:
    init_config()

    parser = argparse.ArgumentParser(
        prog="engra",
        description="Local-first semantic search over your documents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log", default="WARNING", choices=LEVELS, help="Console log level")
    parser.add_argument("--log-file", default="DEBUG", choices=LEVELS, help="File log level")
    parser.add_argument("--log-path", default=None, help="Path to log file")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # index
    p_index = sub.add_parser("index", help="Index files or directories")
    p_index.add_argument(
        "files",
        nargs="*",
        type=Path,
        metavar="FILE_OR_DIR",
        help="Files or directories (pdf, txt, md, rst, html, docx, pptx, epub)",
    )
    p_index.add_argument("--force", action="store_true", help="Re-index even if already present")
    p_index.add_argument("--project", default=None, help="Project name (default: parent dir name)")
    p_index.add_argument("--description", default=None, help="Short description for the project")
    p_index.add_argument(
        "--no-autodescribe",
        action="store_true",
        dest="no_autodescribe",
        help="Skip AI auto-description generation",
    )
    p_index.add_argument(
        "--check",
        action="store_true",
        help="Report stale or missing source files without re-indexing",
    )
    p_index.add_argument(
        "--profile",
        action="store_true",
        help="Print per-phase timing breakdown after indexing",
    )
    store_group = p_index.add_mutually_exclusive_group()
    store_group.add_argument("--link", action="store_true", help="Symlink files instead of copying")
    store_group.add_argument("--no-store", action="store_true", help="Do not copy or link the file")

    # search
    p_search = sub.add_parser("search", help="Search the knowledge base")
    p_search.add_argument("query")
    p_search.add_argument(
        "--top", type=int, default=5, metavar="N", help="Number of results (default: 5)"
    )
    p_search.add_argument(
        "--min-score", type=float, default=0.0, metavar="S", help="Minimum similarity score 0–1"
    )
    p_search.add_argument("--file", metavar="FILENAME", help="Restrict search to a specific file")
    p_search.add_argument(
        "--project",
        action="append",
        metavar="PROJECT",
        dest="projects",
        default=None,
        help="Search in this project (repeatable; overrides session)",
    )
    p_search.add_argument(
        "--all",
        dest="search_all",
        action="store_true",
        help="Search globally, ignoring active session",
    )
    p_search.add_argument(
        "--full",
        action="store_true",
        help="Show complete chunk text instead of a short snippet",
    )
    p_search.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    p_search.add_argument(
        "--rerank",
        action="store_true",
        help="Re-rank results with a cross-encoder (requires: pip install 'engra[rerank]')",
    )
    p_search.add_argument(
        "--links",
        action="store_true",
        help="Surface chunks from files linked by the top results (HTML only)",
    )

    # get
    p_get = sub.add_parser("get", help="Retrieve full chunk text by file and page")
    p_get.add_argument("filename", metavar="FILENAME")
    p_get.add_argument(
        "page", metavar="PAGE_OR_RANGE", help="Page number or range (e.g. 5 or 5-10)"
    )
    p_get.add_argument(
        "--chunk",
        type=int,
        default=None,
        metavar="N",
        help="Retrieve a specific chunk index (default: all chunks on the page)",
    )
    nav_group = p_get.add_mutually_exclusive_group()
    nav_group.add_argument(
        "--next",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="K",
        help="Show K chunks after the specified position (default K=1)",
    )
    nav_group.add_argument(
        "--prev",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="K",
        help="Show K chunks before the specified position (default K=1)",
    )

    # info
    p_info = sub.add_parser("info", help="Show index statistics")
    p_info.add_argument(
        "--file",
        metavar="FILENAME",
        default=None,
        help="Show per-file stats for a specific file",
    )

    # ask
    p_ask = sub.add_parser("ask", help="Answer a question using indexed documents (RAG)")
    p_ask.add_argument("question", help="Question to answer")
    p_ask.add_argument(
        "--project",
        action="append",
        metavar="PROJECT",
        dest="projects",
        default=None,
        help="Restrict context to this project (repeatable; overrides session)",
    )
    p_ask.add_argument("--file", metavar="FILENAME", help="Restrict context to a specific file")
    p_ask.add_argument(
        "--chunks",
        type=int,
        default=None,
        metavar="N",
        help="Number of context chunks to retrieve (default: from config, usually 5)",
    )
    p_ask.add_argument(
        "--all",
        dest="ask_all",
        action="store_true",
        help="Search globally, ignoring active session",
    )

    # list
    sub.add_parser("list", help="List indexed documents")

    # setup-gpu
    sub.add_parser(
        "setup-gpu",
        help="Install the correct onnxruntime-gpu wheel for CUDA 12 (run after pipx install '[gpu]' --force)",
    )

    # mcp
    p_mcp = sub.add_parser("mcp", help="Start MCP stdio server (requires pip install 'engra[mcp]')")
    p_mcp.add_argument(
        "--print-config",
        action="store_true",
        dest="mcp_print_config",
        help="Print the MCP client config snippet and exit",
    )

    # remove
    p_remove = sub.add_parser("remove", help="Remove a document from the index")
    p_remove.add_argument("pdf", type=Path)

    # bookmark
    p_bm = sub.add_parser("bookmark", help="Manage saved searches")
    bm_sub = p_bm.add_subparsers(dest="bm_cmd", required=True)

    p_bm_save = bm_sub.add_parser("save", help="Save a named search")
    p_bm_save.add_argument("name")
    p_bm_save.add_argument("query", help="Search query to save")
    p_bm_save.add_argument("--project", default=None, metavar="PROJECT")
    p_bm_save.add_argument("--top", type=int, default=5, metavar="N")
    p_bm_save.add_argument("--min-score", type=float, default=None, metavar="S", dest="min_score")

    p_bm_run = bm_sub.add_parser("run", help="Re-run a saved search")
    p_bm_run.add_argument("name")

    bm_sub.add_parser("list", help="List all bookmarks")

    p_bm_remove = bm_sub.add_parser("remove", help="Remove a bookmark")
    p_bm_remove.add_argument("name")

    # project
    p_proj = sub.add_parser("project", help="Manage projects")
    proj_sub = p_proj.add_subparsers(dest="proj_cmd", required=True)

    proj_sub.add_parser("list", help="List all projects")
    proj_sub.add_parser("active", help="Show active project(s)")
    proj_sub.add_parser("deactivate", help="Clear active session")

    p_activate = proj_sub.add_parser("activate", help="Activate one or more projects")
    p_activate.add_argument("names", nargs="+", metavar="PROJECT")

    p_rename = proj_sub.add_parser("rename", help="Rename a project")
    p_rename.add_argument("old_name")
    p_rename.add_argument("new_name")

    p_proj_remove = proj_sub.add_parser("remove", help="Remove a project from the index")
    p_proj_remove.add_argument("name")

    p_describe = proj_sub.add_parser("describe", help="Set description and/or keywords")
    p_describe.add_argument("name", metavar="PROJECT")
    p_describe.add_argument("--description", default=None, help="Short description")
    p_describe.add_argument(
        "--keywords",
        nargs="+",
        metavar="KEYWORD",
        default=None,
        help="Space-separated keywords",
    )

    p_autodescribe = proj_sub.add_parser(
        "autodescribe", help="Generate AI description and keywords"
    )
    p_autodescribe.add_argument("name", metavar="PROJECT")

    # export
    p_export = sub.add_parser("export", help="Export a project to a portable archive")
    p_export.add_argument("project", metavar="PROJECT", help="Project name to export")
    p_export.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        type=Path,
        default=None,
        help="Output path (default: <project>.engra.tar.gz)",
    )

    # import
    p_import = sub.add_parser("import", help="Import a project from an archive or directory")
    p_import.add_argument(
        "source",
        metavar="SOURCE",
        type=Path,
        help="Archive (.tar.gz) for hard import, or directory path with --soft",
    )
    p_import.add_argument(
        "--soft",
        action="store_true",
        help="Soft import: index a directory with symlinks (no re-embedding if already indexed)",
    )
    p_import.add_argument(
        "--project",
        metavar="NAME",
        default=None,
        help="Project name override (soft import only; default: directory name)",
    )
    p_import.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing project data (hard import only)",
    )

    args = parser.parse_args()

    log_kwargs: dict = {"console_level": args.log, "file_level": args.log_file}
    if args.log_path:
        log_kwargs["log_path"] = args.log_path
    setup_logging(**log_kwargs)
    logger.info("engra %s started", __version__)

    if args.cmd == "index":
        if args.check:
            cmd_index([], check=True)
        else:
            if not args.files:
                p_index.error("the following arguments are required: FILE_OR_DIR")
            copy = False if args.link else None
            cmd_index(
                args.files,
                force=args.force,
                copy=copy,
                store=not args.no_store,
                project=args.project,
                description=args.description,
                auto_describe=not args.no_autodescribe,
                profile=args.profile,
            )
    elif args.cmd == "search":
        cmd_search(
            args.query,
            top_k=args.top,
            min_score=args.min_score,
            filename=args.file,
            projects=args.projects if not args.search_all else None,
            full=args.full,
            output_format=args.output_format,
            rerank=args.rerank,
            follow_links=args.links,
        )
    elif args.cmd == "get":
        try:
            page_start, page_end = parse_page_range(args.page)
        except ValueError as exc:
            p_get.error(str(exc))
        if page_start != page_end and args.chunk is not None:
            p_get.error("--chunk is not allowed with a page range")
        if page_start != page_end and (args.next is not None or args.prev is not None):
            p_get.error("--next/--prev are not allowed with a page range")
        cmd_get(
            args.filename,
            page_start,
            page_end,
            chunk=args.chunk,
            next_k=args.next,
            prev_k=args.prev,
        )
    elif args.cmd == "ask":
        cmd_ask(
            args.question,
            projects=args.projects if not args.ask_all else None,
            filename=args.file,
            context_chunks=args.chunks,
        )
    elif args.cmd == "info":
        cmd_info(filename=args.file)
    elif args.cmd == "list":
        cmd_list()
    elif args.cmd == "setup-gpu":
        cmd_setup_gpu()
    elif args.cmd == "mcp":
        if args.mcp_print_config:
            import json

            print(
                json.dumps(
                    {"mcpServers": {"engra": {"command": "engra", "args": ["mcp"]}}},
                    indent=2,
                )
            )
        else:
            cmd_mcp()
    elif args.cmd == "remove":
        cmd_remove(args.pdf)
    elif args.cmd == "bookmark":
        if args.bm_cmd == "save":
            cmd_bookmark_save(
                args.name,
                query=args.query,
                project=args.project,
                top=args.top,
                min_score=args.min_score,
            )
        elif args.bm_cmd == "run":
            cmd_bookmark_run(args.name)
        elif args.bm_cmd == "list":
            cmd_bookmark_list()
        elif args.bm_cmd == "remove":
            cmd_bookmark_remove(args.name)
    elif args.cmd == "export":
        cmd_export(args.project, output_path=args.output)
    elif args.cmd == "import":
        if args.soft:
            cmd_import_soft(args.source, project=args.project)
        else:
            if args.project:
                p_import.error("--project is only valid with --soft")
            cmd_import(args.source, overwrite=args.overwrite)
    elif args.cmd == "project":
        if args.proj_cmd == "list":
            cmd_project_list()
        elif args.proj_cmd == "active":
            cmd_project_active()
        elif args.proj_cmd == "activate":
            cmd_project_activate(args.names)
        elif args.proj_cmd == "deactivate":
            cmd_project_deactivate()
        elif args.proj_cmd == "rename":
            cmd_project_rename(args.old_name, args.new_name)
        elif args.proj_cmd == "remove":
            cmd_project_remove(args.name)
        elif args.proj_cmd == "describe":
            cmd_project_describe(
                args.name,
                description=args.description,
                keywords=args.keywords,
            )
        elif args.proj_cmd == "autodescribe":
            cmd_project_autodescribe(args.name)


if __name__ == "__main__":
    run()

import argparse
import logging
from pathlib import Path

from engram import __version__
from engram.commands import (
    cmd_get,
    cmd_index,
    cmd_info,
    cmd_list,
    parse_page_range,
    cmd_project_activate,
    cmd_project_active,
    cmd_project_deactivate,
    cmd_project_list,
    cmd_project_remove,
    cmd_project_rename,
    cmd_remove,
    cmd_search,
)
from engram.config import init as init_config
from engram.log import setup as setup_logging

logger = logging.getLogger(__name__)

LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def run() -> None:
    init_config()

    parser = argparse.ArgumentParser(
        prog="engram",
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
    p_index.add_argument(
        "--check",
        action="store_true",
        help="Report stale or missing source files without re-indexing",
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
        "--project", default=None, help="Search in this project (overrides session)"
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

    # get
    p_get = sub.add_parser("get", help="Retrieve full chunk text by file and page")
    p_get.add_argument("filename", metavar="FILENAME")
    p_get.add_argument("page", metavar="PAGE_OR_RANGE", help="Page number or range (e.g. 5 or 5-10)")
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

    # list
    sub.add_parser("list", help="List indexed documents")

    # remove
    p_remove = sub.add_parser("remove", help="Remove a document from the index")
    p_remove.add_argument("pdf", type=Path)

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

    args = parser.parse_args()

    log_kwargs: dict = {"console_level": args.log, "file_level": args.log_file}
    if args.log_path:
        log_kwargs["log_path"] = args.log_path
    setup_logging(**log_kwargs)
    logger.info("engram %s started", __version__)

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
            )
    elif args.cmd == "search":
        cmd_search(
            args.query,
            top_k=args.top,
            min_score=args.min_score,
            filename=args.file,
            project=args.project if not args.search_all else None,
            full=args.full,
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
        cmd_get(args.filename, page_start, page_end, chunk=args.chunk, next_k=args.next, prev_k=args.prev)
    elif args.cmd == "info":
        cmd_info(filename=args.file)
    elif args.cmd == "list":
        cmd_list()
    elif args.cmd == "remove":
        cmd_remove(args.pdf)
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


if __name__ == "__main__":
    run()

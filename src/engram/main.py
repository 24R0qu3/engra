import argparse
import logging
from pathlib import Path

from engram import __version__
from engram.commands import (
    cmd_index,
    cmd_list,
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
    p_index = sub.add_parser("index", help="Index one or more PDF files")
    p_index.add_argument("pdfs", nargs="+", type=Path)
    p_index.add_argument("--force", action="store_true", help="Re-index even if already present")
    p_index.add_argument("--project", default=None, help="Project name (default: parent dir name)")
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
        copy = False if args.link else None
        cmd_index(
            args.pdfs,
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
        )
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

import datetime
import hashlib
import logging
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from engram.config import BOOKMARKS_PATH, load as load_config
from engram.readers import SUPPORTED_EXTENSIONS, read_file
from engram.storage import (
    DB_DIR,
    clear_session,
    ensure_dirs,
    read_session,
    remove_file,
    store_file,
    write_session,
)

logger = logging.getLogger(__name__)
console = Console()

MODEL_NAME = "intfloat/multilingual-e5-large"
MIN_CHARS = 80
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def get_collection() -> chromadb.Collection:
    ensure_dirs()
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name="pages",
        metadata={"hnsw:space": "cosine"},
    )


def load_model() -> TextEmbedding:
    console.print(f"[dim]Loading model '{MODEL_NAME}'...[/dim]")
    return TextEmbedding(MODEL_NAME)


def doc_id_prefix(pdf_path: Path) -> str:
    digest = hashlib.md5(str(pdf_path.resolve()).encode()).hexdigest()[:8]
    return f"{pdf_path.name}_{digest}"


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping fixed-size character windows."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def default_project(file_path: Path) -> str:
    """Derive a project name from the file's parent directory."""
    parent = file_path.resolve().parent.name
    return parent if parent else "default"


def expand_paths(paths: list[Path]) -> list[Path]:
    """Expand directories to all supported files within them."""
    result = []
    for path in paths:
        if path.is_dir():
            for ext in sorted(SUPPORTED_EXTENSIONS):
                result.extend(sorted(path.rglob(f"*{ext}")))
        else:
            result.append(path)
    return result


def _build_where(
    projects: list[str] | None,
    filename: str | None,
) -> dict | None:
    """Build a chromadb where clause from project + filename filters."""
    clauses = []
    if projects:
        if len(projects) == 1:
            clauses.append({"project": projects[0]})
        else:
            clauses.append({"project": {"$in": projects}})
    if filename:
        clauses.append({"filename": filename})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ── Staleness helpers ─────────────────────────────────────────────────────────


def _stale_warning(source: str, indexed_at: str | None, source_mtime: float | None) -> str | None:
    """Return a warning string if the source file is stale or missing, else None.

    Returns None when the mtime check is unavailable (old index entry, OSError).
    """
    path = Path(source)
    if not path.exists():
        return (
            f"⚠  source file no longer exists at {source}. "
            f"Run: engram remove {path.name} to clean up."
        )
    if source_mtime is None:
        return None  # pre-dates staleness tracking
    try:
        current_mtime = path.stat().st_mtime
    except OSError:
        return None  # unreliable mtime (e.g. network path)
    if current_mtime != source_mtime:
        date_str = (indexed_at or "")[:10] or "unknown"
        return (
            f"⚠  {source} has changed since last indexed ({date_str}). "
            f"Run: engram index <path> to update."
        )
    return None


def _warn_stale_from_metas(metas: list[dict]) -> None:
    """Print stale warnings for unique source files found in a metadata list."""
    seen: set[str] = set()
    for m in metas:
        src = m.get("source", "")
        if not src or src in seen:
            continue
        seen.add(src)
        warning = _stale_warning(src, m.get("indexed_at"), m.get("source_mtime"))
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")


# ── Commands ──────────────────────────────────────────────────────────────────


def cmd_index(
    paths: list[Path],
    force: bool = False,
    copy: bool | None = None,
    store: bool = True,
    project: str | None = None,
    check: bool = False,
) -> None:
    cfg = load_config()

    if check:
        collection = get_collection()
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        if not all_meta:
            console.print("[yellow]Knowledge base is empty.[/yellow]")
            return
        seen: set[str] = set()
        found_stale = False
        for m in all_meta:
            src = m.get("source", "")
            if not src or src in seen:
                continue
            seen.add(src)
            warning = _stale_warning(src, m.get("indexed_at"), m.get("source_mtime"))
            if warning:
                console.print(f"[yellow]{warning}[/yellow]")
                found_stale = True
        if not found_stale:
            console.print("[green]All indexed files are up to date.[/green]")
        return

    if copy is None:
        copy = cfg["index"]["copy"]

    if not copy and sys.platform == "win32":
        console.print(
            "[yellow][warning][/yellow] Symlinks on Windows require Admin privileges or "
            "Developer Mode. Falling back to copy."
        )
        copy = True

    collection = get_collection()
    model = load_model()

    for file_path in expand_paths(paths):
        if not file_path.exists():
            console.print(f"[yellow][skip][/yellow] not found: {file_path}")
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            console.print(f"[yellow][skip][/yellow] unsupported type: {file_path.name}")
            continue

        proj = project or default_project(file_path)
        prefix = doc_id_prefix(file_path)
        existing = collection.get(where={"source": str(file_path.resolve())})

        if existing["ids"] and not force:
            console.print(
                f"[yellow][skip][/yellow] already indexed "
                f"({len(existing['ids'])} chunks): {file_path.name}"
            )
            console.print("       Use [bold]--force[/bold] to re-index.")
            continue

        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            console.print(
                f"[dim][re-index] removed {len(existing['ids'])} old chunks "
                f"for {file_path.name}[/dim]"
            )

        if store:
            stored_path = store_file(file_path, copy=copy)
            action = "copied" if copy else "linked"
            console.print(f"[dim]File {action} → {stored_path}[/dim]")

        try:
            sections = read_file(file_path)
        except Exception as exc:
            console.print(f"[red][error][/red] Could not read {file_path.name}: {exc}")
            continue

        total_sections = sections[0].total if sections else 0
        console.print(
            f"Indexing [bold]{file_path.name}[/bold] ({total_sections} sections) "
            f"→ project [cyan]{proj}[/cyan]"
        )

        indexed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            source_mtime: float | None = file_path.stat().st_mtime
        except OSError:
            source_mtime = None

        try:
            chunk_texts, chunk_metas = [], []
            for section in sections:
                if len(section.text) < MIN_CHARS:
                    continue
                for chunk_idx, chunk in enumerate(chunk_text(section.text)):
                    chunk_texts.append(chunk)
                    chunk_metas.append(
                        {
                            "source": str(file_path.resolve()),
                            "filename": file_path.name,
                            "page": section.phys_page,
                            "page_label": section.page_label,
                            "total_pages": section.total,
                            "chunk": chunk_idx,
                            "project": proj,
                            "indexed_at": indexed_at,
                            "source_mtime": source_mtime,
                            "model": MODEL_NAME,
                            "chunk_size": CHUNK_SIZE,
                            "chunk_overlap": CHUNK_OVERLAP,
                        }
                    )

            ids, embeddings, documents, metadatas = [], [], [], []
            embed_gen = model.embed(chunk_texts, batch_size=32)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Embedding", total=len(chunk_texts))
                for chunk, meta, embedding in zip(chunk_texts, chunk_metas, embed_gen):
                    ids.append(f"{prefix}_p{meta['page']}_c{meta['chunk']}")
                    embeddings.append(embedding.tolist())
                    documents.append(chunk)
                    metadatas.append(meta)
                    progress.advance(task)

        except KeyboardInterrupt:
            console.print(
                f"\n[red][interrupted][/red] Cleaning up partial index for {file_path.name}..."
            )
            partial = collection.get(where={"source": str(file_path.resolve())})
            if partial["ids"]:
                collection.delete(ids=partial["ids"])
            console.print("[red][interrupted][/red] No partial data left. Re-run to index.")
            raise SystemExit(1)

        indexed = len({m["page"] for m in metadatas})
        console.print(
            f"  → [green]{len(ids)} chunks[/green] from "
            f"{indexed}/{total_sections} sections "
            f"(skipped {total_sections - indexed} near-blank)"
        )

        if ids:
            batch = 100
            for i in range(0, len(ids), batch):
                collection.add(
                    ids=ids[i : i + batch],
                    embeddings=embeddings[i : i + batch],
                    documents=documents[i : i + batch],
                    metadatas=metadatas[i : i + batch],
                )

    total = collection.count()
    console.print(f"\nKnowledge base: [bold]{total} chunks[/bold] total.")


def cmd_search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    filename: str | None = None,
    projects: list[str] | None = None,
    full: bool = False,
) -> None:
    collection = get_collection()
    if collection.count() == 0:
        console.print(
            "[yellow]Knowledge base is empty.[/yellow] Run: [bold]engram index <file.pdf>[/bold]"
        )
        return

    # Resolve active projects: explicit flag(s) override session
    if projects:
        active_projects = projects
    else:
        active_projects = read_session()

    if active_projects:
        console.print(f"[dim]Searching in project(s): {', '.join(active_projects)}[/dim]")

    where = _build_where(active_projects, filename)

    model = load_model()
    query_embedding = next(model.query_embed([query])).tolist()

    query_kwargs: dict = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k * 3 if min_score > 0 else top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    _warn_stale_from_metas(metas)
    console.print(f'\nResults for: [bold]"{query}"[/bold]')
    console.rule()

    # Show project prefix when multiple projects are in scope (or global search with mixed results)
    result_projects = {m.get("project", "") for m in metas}
    show_project = len(active_projects) != 1 and len(result_projects) > 1

    shown = 0
    for text, meta, dist in zip(docs, metas, distances):
        score = 1.0 - dist
        if score < min_score:
            continue
        shown += 1
        if shown > top_k:
            break

        doc_label = meta.get("page_label", str(meta["page"]))
        page_str = doc_label if doc_label != str(meta["page"]) else str(meta["page"])
        chunk_info = f"  chunk {meta['chunk']}" if meta.get("chunk", 0) > 0 else ""
        phys_info = f"  (phys. {meta['page']})" if doc_label != str(meta["page"]) else ""
        if show_project:
            file_label = f"[dim]{meta.get('project', '')}[/dim] › {meta['filename']}"
        else:
            file_label = meta["filename"]

        if full:
            snippet = " ".join(text.split())
        else:
            normalized = " ".join(text.split())
            snippet = normalized[:220] + ("…" if len(normalized) > 220 else "")

        console.print(
            f"[bold cyan][{shown}][/bold cyan] {file_label}  —  "
            f"p.{page_str}/{meta['total_pages']}{chunk_info}{phys_info}  "
            f"[dim](score: {score:.3f})[/dim]"
        )
        console.print(f"    [dim]{snippet}[/dim]")
        console.print()

    if shown == 0:
        msg = f"No results above score {min_score:.2f}."
        if active_projects:
            msg += f" (searching in: {', '.join(active_projects)})"
        console.print(f"[yellow]{msg}[/yellow]")


def parse_page_range(page_arg: str) -> tuple[int, int]:
    """Parse a page argument into an inclusive (start, end) range.

    Accepts "42" (single page) or "42-59" (range).
    Raises ValueError for invalid input or reversed ranges.
    """
    if "-" in page_arg:
        raw_start, _, raw_end = page_arg.partition("-")
        try:
            start, end = int(raw_start), int(raw_end)
        except ValueError:
            raise ValueError(f"invalid page range {page_arg!r} — expected PAGE or PAGE-PAGE")
    else:
        try:
            start = end = int(page_arg)
        except ValueError:
            raise ValueError(f"invalid page {page_arg!r} — expected an integer")
    if start > end:
        raise ValueError("start page must be less than or equal to end page")
    return start, end


def _format_missing_pages(pages: list[int]) -> str:
    """Format a sorted list of page numbers as a compact range string."""
    if not pages:
        return ""
    groups: list[str] = []
    seg_start = seg_end = pages[0]
    for p in pages[1:]:
        if p == seg_end + 1:
            seg_end = p
        else:
            groups.append(f"{seg_start}-{seg_end}" if seg_start != seg_end else str(seg_start))
            seg_start = seg_end = p
    groups.append(f"{seg_start}-{seg_end}" if seg_start != seg_end else str(seg_start))
    return ", ".join(groups)


def _get_chunk_sequence(col, filename: str) -> list[tuple[int, int, str]]:
    """Return sorted (page, chunk_idx, page_label) tuples for every chunk of a file."""
    results = col.get(where={"filename": filename}, include=["metadatas"])
    metas: list[dict] = results.get("metadatas") or []
    seen: set[tuple[int, int]] = set()
    items: list[tuple[int, int, str]] = []
    for m in metas:
        key = (m.get("page", 0), m.get("chunk", 0))
        if key not in seen:
            seen.add(key)
            items.append((m["page"], m.get("chunk", 0), m.get("page_label", str(m["page"]))))
    return sorted(items, key=lambda x: (x[0], x[1]))


def _find_seq_index(sequence: list[tuple[int, int, str]], page: int, chunk: int) -> int | None:
    """Return the index of (page, chunk) in sequence, or None if not found."""
    for i, (p, c, _) in enumerate(sequence):
        if p == page and c == chunk:
            return i
    return None


def _print_nav_hint(filename: str, sequence: list[tuple[int, int, str]], first_idx: int, last_idx: int) -> None:
    """Print ← prev / next → navigation hint based on displayed range."""
    if first_idx > 0:
        pp, pc, _ = sequence[first_idx - 1]
        prev_line = f"[dim]← prev: engram get {filename} {pp} --chunk {pc}[/dim]"
    else:
        prev_line = "[dim]← (start of document)[/dim]"

    if last_idx < len(sequence) - 1:
        np_, nc, _ = sequence[last_idx + 1]
        next_line = f"[dim]   next: engram get {filename} {np_} --chunk {nc} →[/dim]"
    else:
        next_line = "[dim]   (end of document) →[/dim]"

    console.print(prev_line)
    console.print(next_line)


def _print_chunk(filename: str, meta: dict, text: str) -> None:
    pg = meta.get("page", 0)
    label = meta.get("page_label", str(pg))
    chunk_idx = meta.get("chunk", 0)
    console.print(f"[bold cyan]{filename}[/bold cyan]  p.{label}  chunk {chunk_idx}")
    if meta.get("source"):
        console.print(f"[dim]{meta['source']}[/dim]")
    console.print(text)
    console.print()


def cmd_get(
    filename: str,
    page_start: int,
    page_end: int,
    chunk: int | None = None,
    next_k: int | None = None,
    prev_k: int | None = None,
) -> None:
    """Retrieve full chunk text from the index by filename and page range."""
    col = get_collection()

    # Stale/missing check for this file
    file_metas = col.get(where={"filename": filename}, include=["metadatas"])
    file_metas_list: list[dict] = file_metas.get("metadatas") or []
    if file_metas_list:
        _warn_stale_from_metas([file_metas_list[0]])

    sequence = _get_chunk_sequence(col, filename)
    single_page = page_start == page_end
    nav_mode = next_k is not None or prev_k is not None

    if nav_mode:
        # Find reference (page, chunk) position in sequence
        ref_page = page_start  # ranges disallowed with nav in main.py
        if chunk is None:
            page_seq_idxs = [i for i, (p, _, _) in enumerate(sequence) if p == ref_page]
            if not page_seq_idxs:
                console.print(f"[yellow]No chunks found for[/yellow] {filename!r} page {ref_page}")
                return
            ref_idx = page_seq_idxs[-1] if next_k is not None else page_seq_idxs[0]
        else:
            ref_idx = _find_seq_index(sequence, ref_page, chunk)
            if ref_idx is None:
                console.print(f"[yellow]Chunk not found:[/yellow] {filename!r} page {ref_page} chunk {chunk}")
                return

        if next_k is not None:
            target = range(ref_idx + 1, min(ref_idx + 1 + next_k, len(sequence)))
        else:
            target = range(max(0, ref_idx - prev_k), ref_idx)

        if not target:
            edge = "end" if next_k is not None else "beginning"
            console.print(f"[yellow]Already at the {edge} of {filename!r}[/yellow]")
            return

        for seq_idx in target:
            tp, tc, _ = sequence[seq_idx]
            r = col.get(
                where={"$and": [{"filename": filename}, {"page": tp}, {"chunk": tc}]},
                include=["documents", "metadatas"],
            )
            for meta, text in zip(r.get("metadatas") or [], r.get("documents") or []):
                _print_chunk(filename, meta, text)

        _print_nav_hint(filename, sequence, target.start, target.stop - 1)
        return

    # Normal mode: fetch page range
    all_pairs: list[tuple[dict, str]] = []
    missing_pages: list[int] = []

    for page in range(page_start, page_end + 1):
        clauses: list[dict] = [{"filename": filename}, {"page": page}]
        if chunk is not None:
            clauses.append({"chunk": chunk})
        results = col.get(where={"$and": clauses}, include=["documents", "metadatas"])
        docs: list[str] = results.get("documents") or []
        metas: list[dict] = results.get("metadatas") or []
        if not docs:
            missing_pages.append(page)
            continue
        all_pairs.extend(
            sorted(zip(metas, docs), key=lambda x: (x[0].get("page", page), x[0].get("chunk", 0)))
        )

    if not all_pairs:
        if single_page:
            msg = f"[yellow]No chunks found for[/yellow] {filename!r} page {page_start}"
            if chunk is not None:
                msg += f" chunk {chunk}"
        else:
            msg = f"[yellow]No chunks found for[/yellow] {filename!r} pages {page_start}-{page_end}"
        console.print(msg)
        return

    for meta, text in all_pairs:
        _print_chunk(filename, meta, text)

    if missing_pages:
        console.print(f"[yellow]pages {_format_missing_pages(missing_pages)} not found in index[/yellow]")

    # Navigation hint for single-page fetches
    if single_page and sequence:
        first_m, last_m = all_pairs[0][0], all_pairs[-1][0]
        f_idx = _find_seq_index(sequence, first_m.get("page", page_start), first_m.get("chunk", 0))
        l_idx = _find_seq_index(sequence, last_m.get("page", page_start), last_m.get("chunk", 0))
        if f_idx is not None and l_idx is not None:
            _print_nav_hint(filename, sequence, f_idx, l_idx)


def cmd_info(filename: str | None = None) -> None:
    """Display index statistics, optionally scoped to a single file."""
    collection = get_collection()
    if collection.count() == 0:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return

    all_meta: list[dict] = collection.get(include=["metadatas"])["metadatas"]

    if filename:
        file_meta = [m for m in all_meta if m.get("filename") == filename]
        if not file_meta:
            console.print(f"[yellow]No indexed file named {filename!r}[/yellow]")
            return
        sources = {m["source"] for m in file_meta}
        chunk_count = len(file_meta)
        page_count = len({m["page"] for m in file_meta})
        indexed_at = file_meta[0].get("indexed_at", "unknown")
        source_mtime_val = file_meta[0].get("source_mtime")
        source_mtime_str = (
            datetime.datetime.fromtimestamp(source_mtime_val).isoformat()
            if source_mtime_val is not None
            else "unknown"
        )
        stale = _stale_warning(file_meta[0]["source"], file_meta[0].get("indexed_at"), source_mtime_val)
        stale_str = "[green]ok[/green]" if stale is None else "[yellow]⚠  stale[/yellow]" if "changed" in (stale or "") else "[red]missing[/red]"
        rows = [
            ("File", filename),
            ("Source(s)", ", ".join(sorted(sources))),
            ("Chunks", str(chunk_count)),
            ("Pages", str(page_count)),
            ("Indexed at", indexed_at),
            ("Source mtime", source_mtime_str),
            ("Stale", stale_str),
        ]
    else:
        sample = all_meta[0] if all_meta else {}
        model = sample.get("model", "unknown")
        chunk_size = sample.get("chunk_size", "unknown")
        chunk_overlap = sample.get("chunk_overlap", "unknown")
        total_files = len({m["source"] for m in all_meta})
        total_chunks = len(all_meta)
        total_pages = len({(m["source"], m["page"]) for m in all_meta})
        indexed_ats = [m["indexed_at"] for m in all_meta if m.get("indexed_at")]
        last_indexed = max(indexed_ats) if indexed_ats else "unknown"
        rows = [
            ("Index path", str(DB_DIR)),
            ("Embedding model", str(model)),
            ("Chunk size", str(chunk_size)),
            ("Chunk overlap", str(chunk_overlap)),
            ("Total files", str(total_files)),
            ("Total chunks", str(total_chunks)),
            ("Total pages", str(total_pages)),
            ("Last indexed", last_indexed),
        ]

    width = max(len(k) for k, _ in rows) + 2
    for key, val in rows:
        console.print(f"[bold]{key:<{width}}[/bold]{val}")


def cmd_list() -> None:
    collection = get_collection()
    if collection.count() == 0:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    files: dict[str, dict] = {}
    for m in all_meta:
        src = m["source"]
        if src not in files:
            files[src] = {
                "filename": m["filename"],
                "project": m.get("project", "—"),
                "chunks": 0,
                "pages": set(),
                "total": m["total_pages"],
                "indexed_at": m.get("indexed_at"),
                "source_mtime": m.get("source_mtime"),
            }
        files[src]["chunks"] += 1
        files[src]["pages"].add(m["page"])

    table = Table(show_header=True, header_style="bold")
    table.add_column("Project", style="cyan")
    table.add_column("File", style="bold")
    table.add_column("Chunks", justify="right")
    table.add_column("Pages", justify="right")
    table.add_column("Stale", justify="center")
    table.add_column("Path", style="dim")

    for src, info in sorted(files.items(), key=lambda x: (x[1]["project"], x[1]["filename"])):
        warning = _stale_warning(src, info.get("indexed_at"), info.get("source_mtime"))
        if warning is None:
            stale_cell = "[green]ok[/green]"
        elif "no longer exists" in warning:
            stale_cell = "[red]missing[/red]"
        else:
            stale_cell = "[yellow]⚠[/yellow]"
        table.add_row(
            info["project"],
            info["filename"],
            str(info["chunks"]),
            str(len(info["pages"])),
            stale_cell,
            src,
        )

    console.print(table)
    console.print(f"Total chunks in index: [bold]{collection.count()}[/bold]")


def cmd_remove(pdf_path: Path) -> None:
    collection = get_collection()
    source = str(pdf_path.resolve())
    existing = collection.get(where={"source": source})

    if not existing["ids"]:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        matches = {m["source"] for m in all_meta if m["filename"] == pdf_path.name}
        if not matches:
            console.print(f"[yellow]Not found in index:[/yellow] {pdf_path.name}")
            return
        if len(matches) > 1:
            console.print(
                f"[yellow]Multiple files named '{pdf_path.name}' found "
                f"— use the full path:[/yellow]"
            )
            for src in sorted(matches):
                console.print(f"  {src}")
            return
        source = matches.pop()
        existing = collection.get(where={"source": source})

    collection.delete(ids=existing["ids"])
    remove_file(Path(source).name)
    console.print(
        f"[green]Removed[/green] {len(existing['ids'])} chunks for "
        f"{Path(source).name}  [dim]({source})[/dim]"
    )


# ── Bookmark commands ─────────────────────────────────────────────────────────


def _load_bookmarks() -> dict[str, dict]:
    """Load bookmarks from disk. Returns a dict keyed by bookmark name."""
    if not BOOKMARKS_PATH.exists():
        return {}
    try:
        import json as _json
        return _json.loads(BOOKMARKS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_bookmarks(bookmarks: dict[str, dict]) -> None:
    import json as _json
    BOOKMARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOOKMARKS_PATH.write_text(_json.dumps(bookmarks, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_bookmark_save(
    name: str,
    query: str,
    project: str | None = None,
    top: int = 5,
    min_score: float | None = None,
) -> None:
    bookmarks = _load_bookmarks()
    if name in bookmarks:
        answer = input(f"Overwrite existing bookmark '{name}'? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            console.print("[dim]Aborted.[/dim]")
            return
    bookmarks[name] = {
        "name": name,
        "query": query,
        "project": project,
        "top": top,
        "min_score": min_score,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _save_bookmarks(bookmarks)
    console.print(f"[green]Saved[/green] bookmark [bold]{name!r}[/bold]")


def cmd_bookmark_run(name: str) -> None:
    bookmarks = _load_bookmarks()
    if name not in bookmarks:
        console.print(f"[yellow]No bookmark named {name!r}.[/yellow]")
        if bookmarks:
            console.print("Available: " + ", ".join(sorted(bookmarks)))
        return
    b = bookmarks[name]
    projects = [b["project"]] if b.get("project") else None
    cmd_search(
        query=b["query"],
        top_k=b.get("top", 5),
        min_score=b.get("min_score") or 0.0,
        projects=projects,
    )


def cmd_bookmark_list() -> None:
    bookmarks = _load_bookmarks()
    if not bookmarks:
        console.print("[yellow]No bookmarks saved.[/yellow]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="bold cyan")
    table.add_column("Query")
    table.add_column("Project", style="dim")
    table.add_column("Top", justify="right")
    table.add_column("Min-score", justify="right")
    for b in sorted(bookmarks.values(), key=lambda x: x["name"]):
        table.add_row(
            b["name"],
            b["query"],
            b.get("project") or "—",
            str(b.get("top", 5)),
            str(b.get("min_score") or "—"),
        )
    console.print(table)


def cmd_bookmark_remove(name: str) -> None:
    bookmarks = _load_bookmarks()
    if name not in bookmarks:
        console.print(f"[yellow]No bookmark named {name!r}.[/yellow]")
        if bookmarks:
            console.print("Available: " + ", ".join(sorted(bookmarks)))
        return
    del bookmarks[name]
    _save_bookmarks(bookmarks)
    console.print(f"[green]Removed[/green] bookmark [bold]{name!r}[/bold]")


# ── Project commands ──────────────────────────────────────────────────────────


def cmd_project_list() -> None:
    collection = get_collection()
    if collection.count() == 0:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    projects: dict[str, dict] = {}
    for m in all_meta:
        proj = m.get("project", "—")
        if proj not in projects:
            projects[proj] = {"chunks": 0, "files": set()}
        projects[proj]["chunks"] += 1
        projects[proj]["files"].add(m["filename"])

    active = read_session()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Project", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Active")

    for name, info in sorted(projects.items()):
        is_active = "● " if name in active else ""
        table.add_row(
            name,
            str(len(info["files"])),
            str(info["chunks"]),
            f"[green]{is_active}yes[/green]" if name in active else "[dim]no[/dim]",
        )

    console.print(table)


def cmd_project_activate(names: list[str]) -> None:
    collection = get_collection()
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    known = {m.get("project") for m in all_meta if m.get("project")}

    valid, unknown = [], []
    for name in names:
        (valid if name in known else unknown).append(name)

    if unknown:
        console.print(f"[yellow]Unknown project(s):[/yellow] {', '.join(unknown)}")
        console.print(f"Known projects: {', '.join(sorted(known))}")
        if not valid:
            return

    current = read_session()
    merged = list(dict.fromkeys(current + valid))  # preserve order, deduplicate
    write_session(merged)
    console.print(f"[green]Active projects:[/green] {', '.join(merged)}")


def cmd_project_deactivate() -> None:
    clear_session()
    console.print("[green]Session cleared.[/green] Searching globally.")


def cmd_project_active() -> None:
    active = read_session()
    if active:
        console.print(f"Active projects: [cyan]{', '.join(active)}[/cyan]")
    else:
        console.print("[dim]No active project — searching globally.[/dim]")


def cmd_project_rename(old_name: str, new_name: str) -> None:
    collection = get_collection()
    existing = collection.get(where={"project": old_name})
    if not existing["ids"]:
        console.print(f"[yellow]Project not found:[/yellow] {old_name}")
        return

    new_metas = [{**m, "project": new_name} for m in existing["metadatas"]]
    collection.update(ids=existing["ids"], metadatas=new_metas)
    console.print(
        f"[green]Renamed[/green] '{old_name}' → '{new_name}' "
        f"({len(existing['ids'])} chunks updated)"
    )

    # Update session if old name was active
    active = read_session()
    if old_name in active:
        updated = [new_name if p == old_name else p for p in active]
        write_session(updated)
        console.print(f"[dim]Session updated: {', '.join(updated)}[/dim]")


def cmd_project_remove(name: str) -> None:
    collection = get_collection()
    existing = collection.get(where={"project": name})
    if not existing["ids"]:
        console.print(f"[yellow]Project not found:[/yellow] {name}")
        return

    # Collect unique filenames for storage cleanup
    filenames = {m["filename"] for m in existing["metadatas"]}
    collection.delete(ids=existing["ids"])

    for filename in filenames:
        # Only remove stored file if no other chunks reference it
        remaining = collection.get(where={"filename": filename})
        if not remaining["ids"]:
            remove_file(filename)

    console.print(
        f"[green]Removed[/green] project '{name}': "
        f"{len(existing['ids'])} chunks, {len(filenames)} file(s)"
    )

    # Remove from session if active
    active = read_session()
    if name in active:
        updated = [p for p in active if p != name]
        write_session(updated) if updated else clear_session()
        console.print("[dim]Removed from active session.[/dim]")

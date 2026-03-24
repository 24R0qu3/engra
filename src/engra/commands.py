import datetime
import hashlib
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from engra.config import BOOKMARKS_PATH, load as load_config
from engra.readers import SUPPORTED_EXTENSIONS, read_file
from engra.storage import (
    CACHE_DIR,
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


def _model_is_cached() -> bool:
    """Return True if the model ONNX file exists in the cache directory."""
    model_dir = CACHE_DIR / "models" / MODEL_NAME.replace("/", "__")
    return any(model_dir.glob("*.onnx")) if model_dir.exists() else False


def load_model():
    from fastembed import TextEmbedding  # noqa: PLC0415 – lazy import avoids ORT load on non-embedding commands

    if _model_is_cached():
        console.print(f"[dim]Loading model '{MODEL_NAME}'...[/dim]")
    else:
        console.print(
            f"[bold]First run:[/bold] downloading model [cyan]{MODEL_NAME}[/cyan] "
            f"to [dim]{CACHE_DIR / 'models'}[/dim] (one-time, ~1 GB)…"
        )
    return TextEmbedding(MODEL_NAME, cache_dir=str(CACHE_DIR / "models"))


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


def _stale_status(source: str, source_mtime: float | None) -> str:
    """Return staleness status: 'ok' | 'stale' | 'missing' | 'unknown'.

    'unknown' means source_mtime was never stored (pre-feature index entry) or
    the mtime is unavailable (OSError on network paths).
    """
    path = Path(source)
    if not path.exists():
        return "missing"
    if source_mtime is None:
        return "unknown"
    try:
        return "ok" if path.stat().st_mtime == source_mtime else "stale"
    except OSError:
        return "unknown"


def _stale_warning(source: str, indexed_at: str | None, source_mtime: float | None) -> str | None:
    """Return a warning string for stale or missing files, else None.

    Returns None for 'ok' and 'unknown' — unknown entries are silent in
    search/get output; cmd_list renders them distinctly via _stale_status.
    """
    status = _stale_status(source, source_mtime)
    if status == "missing":
        return (
            f"⚠  source file no longer exists at {source}. "
            f"Run: engra remove {Path(source).name} to clean up."
        )
    if status == "stale":
        date_str = (indexed_at or "")[:10] or "unknown"
        return (
            f"⚠  {source} has changed since last indexed ({date_str}). "
            f"Run: engra index <path> to update."
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


# ── Data functions (pure retrieval, no console output) ────────────────────────


def _data_index(
    paths: list[Path],
    force: bool = False,
    copy: bool = True,
    store: bool = True,
    project: str | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Index files. Returns {total_chunks, results: [{path, filename, status, ...}]}.

    Each result has status 'indexed' | 'skipped' | 'error'.
    on_progress(done, total, filename) is called after each chunk is embedded.
    Raises RuntimeError on KeyboardInterrupt (cleaned up, no partial data).
    """
    collection = get_collection()
    model = load_model()
    expanded = expand_paths(paths)

    results: list[dict] = []

    for file_path in expanded:
        entry: dict = {"path": str(file_path), "filename": file_path.name}

        if not file_path.exists():
            results.append({**entry, "status": "skipped", "reason": "not found"})
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            results.append({
                **entry, "status": "skipped",
                "reason": f"unsupported type {file_path.suffix}",
            })
            continue

        proj = project or default_project(file_path)
        prefix = doc_id_prefix(file_path)
        existing = collection.get(where={"source": str(file_path.resolve())})

        if existing["ids"] and not force:
            results.append({
                **entry, "status": "skipped",
                "reason": f"already indexed ({len(existing['ids'])} chunks)",
            })
            continue

        reindexed = False
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            reindexed = True

        store_action: str | None = None
        if store:
            store_file(file_path, copy=copy)
            store_action = "copied" if copy else "linked"

        try:
            sections = read_file(file_path)
        except Exception as exc:
            results.append({**entry, "status": "error", "reason": f"read error: {exc}"})
            continue

        total_sections = sections[0].total if sections else 0
        indexed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            source_mtime: float | None = file_path.stat().st_mtime
        except OSError:
            source_mtime = None

        chunk_texts: list[str] = []
        chunk_metas: list[dict] = []
        for section in sections:
            if len(section.text) < MIN_CHARS:
                continue
            for chunk_idx, chunk in enumerate(chunk_text(section.text)):
                chunk_texts.append(chunk)
                chunk_metas.append({
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
                })

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        try:
            embed_gen = model.embed(chunk_texts, batch_size=32)
            for chunk, meta, embedding in zip(chunk_texts, chunk_metas, embed_gen):
                ids.append(f"{prefix}_p{meta['page']}_c{meta['chunk']}")
                embeddings.append(embedding.tolist())
                documents.append(chunk)
                metadatas.append(meta)
                if on_progress:
                    on_progress(len(ids), len(chunk_texts), file_path.name)
        except KeyboardInterrupt:
            partial = collection.get(where={"source": str(file_path.resolve())})
            if partial["ids"]:
                collection.delete(ids=partial["ids"])
            raise RuntimeError(
                f"Interrupted while indexing {file_path.name}. No partial data left."
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

        sections_indexed = len({m["page"] for m in metadatas})
        results.append({
            **entry,
            "status": "indexed",
            "reason": None,
            "project": proj,
            "chunks_added": len(ids),
            "sections_indexed": sections_indexed,
            "total_sections": total_sections,
            "reindexed": reindexed,
            "store_action": store_action,
        })

    return {"total_chunks": collection.count(), "results": results}


def _data_search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    filename: str | None = None,
    projects: list[str] | None = None,
) -> list[dict]:
    """Semantic search. Returns list of result dicts with text and metadata.

    projects=None uses the active session; pass [] to search globally.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    active_projects = projects if projects is not None else read_session()
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

    output: list[dict] = []
    shown = 0
    for text, meta, dist in zip(docs, metas, distances):
        score = 1.0 - dist
        if score < min_score:
            continue
        shown += 1
        if shown > top_k:
            break
        output.append({
            "filename": meta["filename"],
            "page": meta["page"],
            "page_label": meta.get("page_label", str(meta["page"])),
            "total_pages": meta.get("total_pages", 0),
            "chunk": meta.get("chunk", 0),
            "score": round(score, 4),
            "text": text,
            "project": meta.get("project", ""),
            "source": meta.get("source", ""),
            "indexed_at": meta.get("indexed_at"),
            "source_mtime": meta.get("source_mtime"),
        })

    return output


def _data_get_chunks(
    filename: str,
    page_start: int,
    page_end: int,
    chunk: int | None = None,
) -> list[dict]:
    """Retrieve full chunk text for a page range. Returns list of chunk dicts."""
    col = get_collection()
    all_pairs: list[tuple[dict, str]] = []

    for page in range(page_start, page_end + 1):
        clauses: list[dict] = [{"filename": filename}, {"page": page}]
        if chunk is not None:
            clauses.append({"chunk": chunk})
        r = col.get(where={"$and": clauses}, include=["documents", "metadatas"])
        docs: list[str] = r.get("documents") or []
        metas: list[dict] = r.get("metadatas") or []
        if not docs:
            continue
        all_pairs.extend(
            sorted(zip(metas, docs), key=lambda x: (x[0].get("page", page), x[0].get("chunk", 0)))
        )

    return [
        {
            "filename": filename,
            "page": meta.get("page", page_start),
            "page_label": meta.get("page_label", str(meta.get("page", page_start))),
            "chunk_idx": meta.get("chunk", 0),
            "text": text,
            "source": meta.get("source", ""),
        }
        for meta, text in all_pairs
    ]


def _data_get_neighbors(
    filename: str,
    page: int,
    chunk: int,
    direction: str = "next",
    count: int = 1,
) -> list[dict]:
    """Retrieve adjacent chunks. direction is 'next', 'prev', or 'both'.

    'both' returns prev chunks first, then next chunks.
    Returns [] if the anchor (page, chunk) is not found or already at boundary.
    """
    if direction not in ("next", "prev", "both"):
        raise ValueError(f"direction must be 'next', 'prev', or 'both', got {direction!r}")

    col = get_collection()
    sequence = _get_chunk_sequence(col, filename)
    ref_idx = _find_seq_index(sequence, page, chunk)
    if ref_idx is None:
        return []

    if direction == "next":
        targets = list(range(ref_idx + 1, min(ref_idx + 1 + count, len(sequence))))
    elif direction == "prev":
        targets = list(range(max(0, ref_idx - count), ref_idx))
    else:  # both
        targets = (
            list(range(max(0, ref_idx - count), ref_idx))
            + list(range(ref_idx + 1, min(ref_idx + 1 + count, len(sequence))))
        )

    result: list[dict] = []
    for seq_idx in targets:
        tp, tc, _ = sequence[seq_idx]
        r = col.get(
            where={"$and": [{"filename": filename}, {"page": tp}, {"chunk": tc}]},
            include=["documents", "metadatas"],
        )
        for meta, text in zip(r.get("metadatas") or [], r.get("documents") or []):
            result.append({
                "filename": filename,
                "page": meta.get("page", tp),
                "page_label": meta.get("page_label", str(tp)),
                "chunk_idx": meta.get("chunk", tc),
                "text": text,
                "source": meta.get("source", ""),
            })

    return result


def _data_list_projects() -> list[dict]:
    """Return per-project stats. Returns [] when index is empty."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    projects: dict[str, dict] = {}
    for m in all_meta:
        proj = m.get("project", "—")
        if proj not in projects:
            projects[proj] = {"chunks": 0, "files": set()}
        projects[proj]["chunks"] += 1
        projects[proj]["files"].add(m["filename"])

    return [
        {"name": name, "file_count": len(info["files"]), "chunk_count": info["chunks"]}
        for name, info in sorted(projects.items())
    ]


def _data_list_files(project: str | None = None) -> list[dict]:
    """Return per-file stats, optionally filtered by project."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    files: dict[str, dict] = {}
    for m in all_meta:
        src = m["source"]
        if project and m.get("project") != project:
            continue
        if src not in files:
            files[src] = {
                "filename": m["filename"],
                "project": m.get("project", "—"),
                "chunks": 0,
                "pages": set(),
                "indexed_at": m.get("indexed_at"),
                "source_mtime": m.get("source_mtime"),
            }
        files[src]["chunks"] += 1
        files[src]["pages"].add(m["page"])

    return [
        {
            "filename": info["filename"],
            "project": info["project"],
            "chunks": info["chunks"],
            "pages": len(info["pages"]),
            "indexed_at": info.get("indexed_at") or "unknown",
            "stale_status": _stale_status(src, info.get("source_mtime")),
            "source": src,
        }
        for src, info in sorted(files.items(), key=lambda x: (x[1]["project"], x[1]["filename"]))
    ]


def _data_info() -> dict:
    """Return global index statistics."""
    collection = get_collection()
    if collection.count() == 0:
        return {
            "total_chunks": 0, "files": 0, "projects": 0,
            "model": MODEL_NAME, "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP, "last_indexed": "unknown",
        }

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sample = all_meta[0]
    return {
        "total_chunks": len(all_meta),
        "files": len({m["source"] for m in all_meta}),
        "projects": len({m.get("project", "") for m in all_meta}),
        "model": sample.get("model", "unknown"),
        "chunk_size": sample.get("chunk_size", "unknown"),
        "chunk_overlap": sample.get("chunk_overlap", "unknown"),
        "last_indexed": max(
            (m["indexed_at"] for m in all_meta if m.get("indexed_at")), default="unknown"
        ),
    }


def _data_project_activate(names: list[str]) -> dict:
    """Activate projects; returns {active_projects, unknown}."""
    collection = get_collection()
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    known = {m.get("project") for m in all_meta if m.get("project")}

    valid: list[str] = []
    unknown: list[str] = []
    for name in names:
        (valid if name in known else unknown).append(name)

    if valid:
        current = read_session()
        merged = list(dict.fromkeys(current + valid))
        write_session(merged)
    else:
        merged = read_session()

    return {"active_projects": merged, "unknown": unknown}


def _data_project_deactivate() -> dict:
    """Clear the active session. Returns {active_projects: []}."""
    clear_session()
    return {"active_projects": []}


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
        found_issue = False
        unknown_count = 0
        for m in all_meta:
            src = m.get("source", "")
            if not src or src in seen:
                continue
            seen.add(src)
            status = _stale_status(src, m.get("source_mtime"))
            if status in ("stale", "missing"):
                warning = _stale_warning(src, m.get("indexed_at"), m.get("source_mtime"))
                console.print(f"[yellow]{warning}[/yellow]")
                found_issue = True
            elif status == "unknown":
                unknown_count += 1
        if unknown_count:
            console.print(
                f"[dim]{unknown_count} file(s) predate staleness tracking "
                f"— re-index to enable mtime checks.[/dim]"
            )
            found_issue = True
        if not found_issue:
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

    expanded = expand_paths(paths)
    if not expanded:
        console.print(
            "[yellow][warning][/yellow] No supported files found. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        return

    tasks: dict[str, int] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        def on_progress(done: int, total: int, filename: str) -> None:
            if filename not in tasks:
                tasks[filename] = progress.add_task(
                    f"Embedding [cyan]{filename}[/cyan]", total=total
                )
            progress.update(tasks[filename], completed=done, total=total)

        try:
            result = _data_index(
                paths,
                force=force,
                copy=copy,
                store=store,
                project=project,
                on_progress=on_progress,
            )
        except RuntimeError as exc:
            console.print(f"\n[red][interrupted][/red] {exc}")
            raise SystemExit(1)

    for r in result["results"]:
        if r["status"] == "skipped":
            reason = r["reason"] or ""
            if "already indexed" in reason:
                console.print(f"[yellow][skip][/yellow] {reason}: {r['filename']}")
                console.print("       Use [bold]--force[/bold] to re-index.")
            elif reason == "not found":
                console.print(f"[yellow][skip][/yellow] not found: {r['path']}")
            else:
                console.print(f"[yellow][skip][/yellow] {reason}: {r['filename']}")
        elif r["status"] == "error":
            console.print(f"[red][error][/red] Could not read {r['filename']}: {r['reason']}")
        elif r["status"] == "indexed":
            if r.get("reindexed"):
                console.print(f"[dim][re-index] {r['filename']}[/dim]")
            if r.get("store_action"):
                console.print(f"[dim]File {r['store_action']}[/dim]")
            skipped = r["total_sections"] - r["sections_indexed"]
            skip_note = (
                f", skipped {skipped} sections shorter than {MIN_CHARS} chars"
                if skipped > 0
                else ""
            )
            console.print(
                f"  → [green]{r['chunks_added']} chunks[/green] from "
                f"{r['sections_indexed']}/{r['total_sections']} sections{skip_note} "
                f"→ project [cyan]{r['project']}[/cyan]"
            )

    console.print(f"\nKnowledge base: [bold]{result['total_chunks']} chunks[/bold] total.")


def cmd_search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    filename: str | None = None,
    projects: list[str] | None = None,
    full: bool = False,
) -> None:
    if get_collection().count() == 0:
        console.print(
            "[yellow]Knowledge base is empty.[/yellow] Run: [bold]engra index <file.pdf>[/bold]"
        )
        return

    active_projects = projects if projects is not None else read_session()
    if active_projects:
        console.print(f"[dim]Searching in project(s): {', '.join(active_projects)}[/dim]")

    hits = _data_search(query, top_k=top_k, min_score=min_score, filename=filename, projects=projects)

    _warn_stale_from_metas([
        {"source": h["source"], "indexed_at": h.get("indexed_at"), "source_mtime": h.get("source_mtime")}
        for h in hits
    ])
    console.print(f'\nResults for: [bold]"{query}"[/bold]')
    console.rule()

    result_projects = {h["project"] for h in hits}
    show_project = len(active_projects) != 1 and len(result_projects) > 1

    for i, h in enumerate(hits, 1):
        page_str = h["page_label"] if h["page_label"] != str(h["page"]) else str(h["page"])
        chunk_info = f"  chunk {h['chunk']}" if h["chunk"] > 0 else ""
        phys_info = f"  (phys. {h['page']})" if h["page_label"] != str(h["page"]) else ""
        file_label = (
            f"[dim]{h['project']}[/dim] › {h['filename']}" if show_project else h["filename"]
        )

        text = h["text"]
        if full:
            snippet = " ".join(text.split())
        else:
            normalized = " ".join(text.split())
            snippet = normalized[:220] + ("…" if len(normalized) > 220 else "")

        console.print(
            f"[bold cyan][{i}][/bold cyan] {file_label}  —  "
            f"p.{page_str}/{h['total_pages']}{chunk_info}{phys_info}  "
            f"[dim](score: {h['score']:.3f})[/dim]"
        )
        console.print(f"    [dim]{snippet}[/dim]")
        console.print()

    if not hits:
        msg = f"No results above score {min_score:.2f}."
        if active_projects:
            msg += f" (searching in: {', '.join(active_projects)})"
        console.print(f"[yellow]{msg}[/yellow]")


def cmd_ask(
    question: str,
    projects: list[str] | None = None,
    filename: str | None = None,
    context_chunks: int | None = None,
) -> None:
    """Answer a question using retrieved chunks as context (RAG)."""
    import urllib.error
    import urllib.request

    cfg = load_config()
    ask_cfg = cfg.get("ask", {})

    api_base = ask_cfg.get("api_base", "http://localhost:11434/v1").rstrip("/")
    model_id = ask_cfg.get("model", "llama3")
    api_key = ask_cfg.get("api_key", "ollama")
    top_k = context_chunks if context_chunks is not None else int(ask_cfg.get("context_chunks", 5))
    system_prompt = ask_cfg.get(
        "system_prompt",
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the context does not contain enough information, say so.",
    )

    collection = get_collection()
    if collection.count() == 0:
        console.print(
            "[yellow]Knowledge base is empty.[/yellow] Run: [bold]engra index <file>[/bold]"
        )
        return

    if projects:
        active_projects = projects
    else:
        active_projects = read_session()

    where = _build_where(active_projects, filename)

    model = load_model()
    query_embedding = next(model.query_embed([question])).tolist()

    query_kwargs: dict = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas"],
    )
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    # Build context block
    context_parts = []
    for i, (text, meta) in enumerate(zip(docs, metas), 1):
        label = f"[{i}] {meta['filename']} p.{meta.get('page_label', meta['page'])}"
        context_parts.append(f"{label}\n{text.strip()}")
    context_text = "\n\n---\n\n".join(context_parts)

    # Print sources header
    console.print(f'\n[bold]Question:[/bold] {question}')
    console.print(f'[dim]Context: {len(docs)} chunk(s) from index[/dim]')
    console.rule()

    # Call LLM
    import json as _json

    payload = _json.dumps(
        {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n\n{context_text}\n\nQuestion: {question}",
                },
            ],
            "stream": False,
        }
    ).encode()

    req = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            response_data = _json.loads(resp.read())
        answer = response_data["choices"][0]["message"]["content"]
    except urllib.error.URLError as exc:
        console.print(f"[red]LLM request failed:[/red] {exc}")
        console.print(
            f"[dim]Ensure your LLM server is running at [bold]{api_base}[/bold] "
            "or configure [bold]ask.api_base[/bold] in ~/.config/engra/config.toml[/dim]"
        )
        return
    except (KeyError, IndexError, _json.JSONDecodeError) as exc:
        console.print(f"[red]Unexpected LLM response:[/red] {exc}")
        return

    console.print(f"\n[bold green]Answer[/bold green] [dim](model: {model_id})[/dim]\n")
    console.print(answer)
    console.print()
    console.rule("Sources")
    for i, meta in enumerate(metas, 1):
        page_label = meta.get("page_label", str(meta["page"]))
        console.print(
            f"  [{i}] {meta['filename']}  p.{page_label}  "
            f"[dim](chunk {meta.get('chunk', 0)})[/dim]"
        )


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
        prev_line = f"[dim]← prev: engra get {filename} {pp} --chunk {pc}[/dim]"
    else:
        prev_line = "[dim]← (start of document)[/dim]"

    if last_idx < len(sequence) - 1:
        np_, nc, _ = sequence[last_idx + 1]
        next_line = f"[dim]   next: engra get {filename} {np_} --chunk {nc} →[/dim]"
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
    file_metas_list: list[dict] = col.get(where={"filename": filename}, include=["metadatas"]).get("metadatas") or []
    if file_metas_list:
        _warn_stale_from_metas([file_metas_list[0]])

    sequence = _get_chunk_sequence(col, filename)
    single_page = page_start == page_end
    nav_mode = next_k is not None or prev_k is not None

    if nav_mode:
        ref_page = page_start
        if chunk is None:
            page_seq_idxs = [i for i, (p, _, _) in enumerate(sequence) if p == ref_page]
            if not page_seq_idxs:
                console.print(f"[yellow]No chunks found for[/yellow] {filename!r} page {ref_page}")
                return
            ref_seq_idx = page_seq_idxs[-1] if next_k is not None else page_seq_idxs[0]
            ref_chunk = sequence[ref_seq_idx][1]
        else:
            ref_chunk = chunk
            if _find_seq_index(sequence, ref_page, chunk) is None:
                console.print(f"[yellow]Chunk not found:[/yellow] {filename!r} page {ref_page} chunk {chunk}")
                return

        direction = "next" if next_k is not None else "prev"
        count = next_k if next_k is not None else (prev_k or 1)
        chunks = _data_get_neighbors(filename, ref_page, ref_chunk, direction=direction, count=count)

        if not chunks:
            edge = "end" if next_k is not None else "beginning"
            console.print(f"[yellow]Already at the {edge} of {filename!r}[/yellow]")
            return

        for c in chunks:
            _print_chunk(filename, {"page": c["page"], "page_label": c["page_label"], "chunk": c["chunk_idx"], "source": c["source"]}, c["text"])

        first_idx = _find_seq_index(sequence, chunks[0]["page"], chunks[0]["chunk_idx"])
        last_idx = _find_seq_index(sequence, chunks[-1]["page"], chunks[-1]["chunk_idx"])
        if first_idx is not None and last_idx is not None:
            _print_nav_hint(filename, sequence, first_idx, last_idx)
        return

    # Normal mode
    chunks = _data_get_chunks(filename, page_start, page_end, chunk=chunk)

    if not chunks:
        if single_page:
            msg = f"[yellow]No chunks found for[/yellow] {filename!r} page {page_start}"
            if chunk is not None:
                msg += f" chunk {chunk}"
        else:
            msg = f"[yellow]No chunks found for[/yellow] {filename!r} pages {page_start}-{page_end}"
        console.print(msg)
        return

    for c in chunks:
        _print_chunk(filename, {"page": c["page"], "page_label": c["page_label"], "chunk": c["chunk_idx"], "source": c["source"]}, c["text"])

    found_pages = {c["page"] for c in chunks}
    missing_pages = [p for p in range(page_start, page_end + 1) if p not in found_pages]
    if missing_pages:
        console.print(f"[yellow]pages {_format_missing_pages(missing_pages)} not found in index[/yellow]")

    if single_page and sequence:
        f_idx = _find_seq_index(sequence, chunks[0]["page"], chunks[0]["chunk_idx"])
        l_idx = _find_seq_index(sequence, chunks[-1]["page"], chunks[-1]["chunk_idx"])
        if f_idx is not None and l_idx is not None:
            _print_nav_hint(filename, sequence, f_idx, l_idx)


def cmd_info(filename: str | None = None) -> None:
    """Display index statistics, optionally scoped to a single file."""
    collection = get_collection()
    if collection.count() == 0:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return

    if filename:
        all_meta: list[dict] = collection.get(include=["metadatas"])["metadatas"]
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
        stale_str = {
            "ok": "[green]ok[/green]",
            "stale": "[yellow]⚠  stale[/yellow]",
            "missing": "[red]missing[/red]",
            "unknown": "[dim]?  unknown (re-index to populate)[/dim]",
        }[_stale_status(file_meta[0]["source"], source_mtime_val)]
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
        d = _data_info()
        rows = [
            ("Index path", str(DB_DIR)),
            ("Embedding model", str(d["model"])),
            ("Chunk size", str(d["chunk_size"])),
            ("Chunk overlap", str(d["chunk_overlap"])),
            ("Total files", str(d["files"])),
            ("Total chunks", str(d["total_chunks"])),
            ("Last indexed", d["last_indexed"]),
        ]

    width = max(len(k) for k, _ in rows) + 2
    for key, val in rows:
        hint = "  [dim](re-index to populate)[/dim]" if val == "unknown" else ""
        console.print(f"[bold]{key:<{width}}[/bold]{val}{hint}")


def cmd_list() -> None:
    if get_collection().count() == 0:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return

    file_list = _data_list_files()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Project", style="cyan")
    table.add_column("File", style="bold")
    table.add_column("Chunks", justify="right")
    table.add_column("Pages", justify="right")
    table.add_column("Stale", justify="center")
    table.add_column("Path", style="dim")

    stale_cells = {
        "ok": "[green]ok[/green]",
        "stale": "[yellow]⚠[/yellow]",
        "missing": "[red]missing[/red]",
        "unknown": "[dim]?[/dim]",
    }
    for f in file_list:
        table.add_row(
            f["project"], f["filename"], str(f["chunks"]),
            str(f["pages"]), stale_cells[f["stale_status"]], f["source"],
        )

    console.print(table)
    console.print(f"Total chunks in index: [bold]{get_collection().count()}[/bold]")


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
    if get_collection().count() == 0:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return

    projects = _data_list_projects()
    active = read_session()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Project", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Active")

    for p in projects:
        is_active = p["name"] in active
        table.add_row(
            p["name"], str(p["file_count"]), str(p["chunk_count"]),
            f"[green]● yes[/green]" if is_active else "[dim]no[/dim]",
        )

    console.print(table)


def cmd_project_activate(names: list[str]) -> None:
    r = _data_project_activate(names)
    if r["unknown"]:
        known = {p["name"] for p in _data_list_projects()}
        console.print(f"[yellow]Unknown project(s):[/yellow] {', '.join(r['unknown'])}")
        console.print(f"Known projects: {', '.join(sorted(known))}")
        if not r["active_projects"]:
            return
    if r["active_projects"]:
        console.print(f"[green]Active projects:[/green] {', '.join(r['active_projects'])}")


def cmd_project_deactivate() -> None:
    _data_project_deactivate()
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


def cmd_mcp() -> None:
    """Start the MCP stdio server. Requires the 'mcp' optional dependency."""
    try:
        from engra.mcp_server import run_mcp_server  # noqa: PLC0415
    except ImportError as exc:
        console.print(f"[red]MCP support not installed:[/red] {exc}")
        console.print("Install with: [bold]pip install 'engra[mcp]'[/bold]")
        raise SystemExit(1)
    run_mcp_server()

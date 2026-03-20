import hashlib
import logging
import sys
from pathlib import Path

import chromadb
import fitz  # pymupdf
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

from engram.config import load as load_config
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


def default_project(pdf_path: Path) -> str:
    """Derive a project name from the PDF's parent directory."""
    parent = pdf_path.resolve().parent.name
    return parent if parent else "default"


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


# ── Commands ──────────────────────────────────────────────────────────────────


def cmd_index(
    pdf_paths: list[Path],
    force: bool = False,
    copy: bool | None = None,
    store: bool = True,
    project: str | None = None,
) -> None:
    cfg = load_config()
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

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            console.print(f"[yellow][skip][/yellow] not found: {pdf_path}")
            continue

        proj = project or default_project(pdf_path)
        prefix = doc_id_prefix(pdf_path)
        existing = collection.get(where={"source": str(pdf_path.resolve())})

        if existing["ids"] and not force:
            console.print(
                f"[yellow][skip][/yellow] already indexed "
                f"({len(existing['ids'])} chunks): {pdf_path.name}"
            )
            console.print("       Use [bold]--force[/bold] to re-index.")
            continue

        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            console.print(
                f"[dim][re-index] removed {len(existing['ids'])} old chunks "
                f"for {pdf_path.name}[/dim]"
            )

        if store:
            stored_path = store_file(pdf_path, copy=copy)
            action = "copied" if copy else "linked"
            console.print(f"[dim]File {action} → {stored_path}[/dim]")

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        console.print(
            f"Indexing [bold]{pdf_path.name}[/bold] ({total_pages} pages) "
            f"→ project [cyan]{proj}[/cyan]"
        )

        try:
            chunk_texts, chunk_metas = [], []
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text().strip()
                if len(text) < MIN_CHARS:
                    continue
                phys_page = page_num + 1
                doc_label = page.get_label() or str(phys_page)
                for chunk_idx, chunk in enumerate(chunk_text(text)):
                    chunk_texts.append(chunk)
                    chunk_metas.append(
                        {
                            "source": str(pdf_path.resolve()),
                            "filename": pdf_path.name,
                            "page": phys_page,
                            "page_label": doc_label,
                            "total_pages": total_pages,
                            "chunk": chunk_idx,
                            "project": proj,
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
            doc.close()
            console.print(
                f"\n[red][interrupted][/red] Cleaning up partial index for {pdf_path.name}..."
            )
            partial = collection.get(where={"source": str(pdf_path.resolve())})
            if partial["ids"]:
                collection.delete(ids=partial["ids"])
            console.print("[red][interrupted][/red] No partial data left. Re-run to index.")
            raise SystemExit(1)

        indexed_pages = len({m["page"] for m in metadatas})
        console.print(
            f"  → [green]{len(ids)} chunks[/green] from "
            f"{indexed_pages}/{total_pages} pages "
            f"(skipped {total_pages - indexed_pages} near-blank)"
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

        doc.close()

    total = collection.count()
    console.print(f"\nKnowledge base: [bold]{total} chunks[/bold] total.")


def cmd_search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    filename: str | None = None,
    project: str | None = None,
) -> None:
    collection = get_collection()
    if collection.count() == 0:
        console.print(
            "[yellow]Knowledge base is empty.[/yellow] Run: [bold]engram index <file.pdf>[/bold]"
        )
        return

    # Resolve active projects: explicit flag overrides session
    if project:
        active_projects = [project]
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

    console.print(f'\nResults for: [bold]"{query}"[/bold]')
    console.rule()

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
        proj_info = f"  [dim][{meta.get('project', '')}][/dim]" if not active_projects else ""

        snippet = " ".join(text.split())[:220]
        if len(" ".join(text.split())) > 220:
            snippet += "…"

        console.print(
            f"[bold cyan][{shown}][/bold cyan] {meta['filename']}{proj_info}  —  "
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
            }
        files[src]["chunks"] += 1
        files[src]["pages"].add(m["page"])

    table = Table(show_header=True, header_style="bold")
    table.add_column("Project", style="cyan")
    table.add_column("File", style="bold")
    table.add_column("Chunks", justify="right")
    table.add_column("Pages", justify="right")
    table.add_column("Path", style="dim")

    for src, info in sorted(files.items(), key=lambda x: (x[1]["project"], x[1]["filename"])):
        table.add_row(
            info["project"],
            info["filename"],
            str(info["chunks"]),
            str(len(info["pages"])),
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

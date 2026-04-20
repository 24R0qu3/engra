import datetime
import gc
import hashlib
import logging
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
    TimeElapsedColumn,
)
from rich.table import Table

from engra.config import BOOKMARKS_PATH
from engra.config import load as load_config
from engra.readers import SUPPORTED_EXTENSIONS, read_file
from engra.storage import (
    CACHE_DIR,
    DB_DIR,
    FILES_DIR,
    clear_session,
    ensure_dirs,
    read_projects,
    read_session,
    remove_file,
    remove_project_meta,
    rename_project_meta,
    store_file,
    update_project_meta,
    write_session,
)

logger = logging.getLogger(__name__)
console = Console()

MODEL_NAME = "intfloat/multilingual-e5-large"
MIN_CHARS = 80
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
DEFAULT_MIN_SCORE = 0.3  # MCP engra_search default; below this is treated as "not found"

_NOTABLE_PATTERN = re.compile(
    r"\b(TBD|TODO|FIXME|stub)\b|not\s+implemented|raise\s+NotImplementedError",
    re.IGNORECASE,
)


def _is_notable(text: str) -> bool:
    """Return True if the chunk contains a stub/TBD/TODO/FIXME marker."""
    return bool(_NOTABLE_PATTERN.search(text))


_RERANKER_DEFAULT_MODEL = "ms-marco-MiniLM-L-12-v2"


def _load_reranker():
    """Load the flashrank cross-encoder; raises RuntimeError if not installed."""
    try:
        from flashrank import Ranker
    except ImportError:
        raise RuntimeError("Re-ranking requires flashrank: pip install 'engra[rerank]'") from None

    cfg = load_config()
    model_name = cfg.get("rerank", {}).get("model", _RERANKER_DEFAULT_MODEL)
    console.print(f"[dim]Loading re-ranker '{model_name}'…[/dim]")
    return Ranker(model_name=model_name, cache_dir=str(CACHE_DIR / "rerankers"))


def _rerank_results(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """Re-rank *hits* with a cross-encoder and return the top *top_k* by rerank score."""
    from flashrank import RerankRequest

    ranker = _load_reranker()
    passages = [{"id": i, "text": h["text"]} for i, h in enumerate(hits)]
    request = RerankRequest(query=query, passages=passages)
    ranked = ranker.rerank(request)

    reranked: list[dict] = []
    for r in ranked[:top_k]:
        hit = dict(hits[r["id"]])
        hit["rerank_score"] = round(float(r["score"]), 4)
        reranked.append(hit)
    return reranked


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores to [0, 1] relative to the result set.

    Returns [1.0, ...] when the list is empty, has one element, or all values
    are identical (no spread to normalise).
    """
    if len(scores) <= 1:
        return [1.0] * len(scores)
    lo, hi = min(scores), max(scores)
    spread = hi - lo
    if spread == 0.0:
        return [1.0] * len(scores)
    return [(s - lo) / spread for s in scores]


AUTO_DESCRIBE_SAMPLE_CHUNKS = 5
AUTO_DESCRIBE_MAX_CHARS = 4000


def _autodescribe_prompt(project: str, sample: str) -> str:
    return (
        f"Analyze these excerpts from a document collection called '{project}'.\n"
        "Return ONLY a JSON object (no markdown, no code fences, no explanation):\n"
        '{"description": "<one sentence, max 25 words, factual>", '
        '"keywords": ["<5 to 10 specific key terms or phrases>"]}\n\n'
        f"Excerpts:\n{sample}"
    )


def _parse_autodescribe_response(text: str) -> tuple[str, list[str]] | None:
    """Extract (description, keywords) from an LLM JSON response."""
    import json as _json
    import re

    # Strip markdown code fences if present
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = _json.loads(match.group())
        desc = str(data.get("description", "")).strip()
        kws = [str(k).strip() for k in data.get("keywords", []) if str(k).strip()]
        if desc or kws:
            return desc, kws
    except Exception:
        pass
    return None


def _auto_describe_openai(prompt: str, cfg: dict) -> tuple[str, list[str]] | None:
    import json as _json
    import urllib.error
    import urllib.request

    api_base = cfg.get("api_base", "http://localhost:11434/v1").rstrip("/")
    model_id = cfg.get("model", "llama3")
    api_key = cfg.get("api_key", "ollama")

    payload = _json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.3,
            "think": False,  # disable reasoning chain for thinking models (e.g. qwen3.5)
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
            data = _json.loads(resp.read())
        msg = data["choices"][0]["message"]
        # Reasoning models (e.g. qwen3.5) may return content in "reasoning" when content is empty
        text = msg.get("content") or msg.get("reasoning") or ""
        return _parse_autodescribe_response(text)
    except Exception:
        logger.warning("Auto-description via OpenAI-compatible endpoint failed")
        return None


def _auto_describe_claude(prompt: str, cfg: dict) -> tuple[str, list[str]] | None:
    import os

    try:
        import anthropic
    except ImportError:
        logger.warning("Claude auto-description requires: pip install 'engra[ai]'")
        return None

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set; skipping auto-description")
        return None

    model_id = cfg.get("claude_model", "claude-haiku-4-5-20251001")
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model_id,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_autodescribe_response(msg.content[0].text)
    except Exception:
        logger.warning("Claude auto-description failed")
        return None


def _auto_describe(project: str, chunks: list[str]) -> tuple[str, list[str]] | None:
    """Generate (description, keywords) using the configured backend. Returns None on failure."""
    cfg = load_config()
    ad_cfg = cfg.get("autodescribe", {})
    backend = ad_cfg.get("backend", "openai")

    if backend == "disabled" or not chunks:
        return None

    sample = "\n\n---\n\n".join(chunks[:AUTO_DESCRIBE_SAMPLE_CHUNKS])[:AUTO_DESCRIBE_MAX_CHARS]
    prompt = _autodescribe_prompt(project, sample)

    if backend == "claude":
        return _auto_describe_claude(prompt, ad_cfg)
    return _auto_describe_openai(prompt, ad_cfg)


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
    import os

    from fastembed import (
        TextEmbedding,  # noqa: PLC0415 – lazy import avoids ORT load on non-embedding commands
    )

    cfg = load_config().get("embedding", {})
    threads_cfg: int = cfg.get("threads", 0)
    threads = threads_cfg if threads_cfg > 0 else (os.cpu_count() or 1)
    provider: str = cfg.get("provider", "cpu").lower()

    if _model_is_cached():
        console.print(f"[dim]Loading model '{MODEL_NAME}'...[/dim]")
    else:
        console.print(
            f"[bold]First run:[/bold] downloading model [cyan]{MODEL_NAME}[/cyan] "
            f"to [dim]{CACHE_DIR / 'models'}[/dim] (one-time, ~1 GB)…"
        )

    kwargs: dict = {
        "cache_dir": str(CACHE_DIR / "models"),
        "threads": threads,
    }
    if provider != "cpu":
        provider_map = {
            "cuda": "CUDAExecutionProvider",
            "rocm": "ROCMExecutionProvider",
            "directml": "DmlExecutionProvider",
        }
        ort_provider = provider_map.get(provider, f"{provider}ExecutionProvider")
        kwargs["providers"] = [ort_provider, "CPUExecutionProvider"]
        try:
            return TextEmbedding(MODEL_NAME, **kwargs)
        except Exception:
            import sys

            venv_python = sys.executable
            console.print(
                f"[yellow]Warning:[/yellow] provider '{provider}' unavailable,"
                f" falling back to CPU.\n"
                f"  To fix, run:\n"
                f"  [bold]{venv_python} -m pip install onnxruntime-gpu "
                f"--extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/[/bold]"
            )
            del kwargs["providers"]

    return TextEmbedding(MODEL_NAME, **kwargs)


def doc_id_prefix(pdf_path: Path) -> str:
    digest = hashlib.md5(str(pdf_path.resolve()).encode()).hexdigest()[:8]
    return f"{pdf_path.name}_{digest}"


def chunk_text(text: str) -> list[str]:
    """Split text at paragraph/sentence boundaries up to CHUNK_SIZE chars with overlap."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    paragraphs = re.split(r"\n{2,}", text)
    raw: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = (current + "\n\n" + para).lstrip()
        else:
            if current:
                raw.append(current)
            if len(para) <= CHUNK_SIZE:
                current = para
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= CHUNK_SIZE:
                        current = (current + " " + sent).lstrip()
                    else:
                        if current:
                            raw.append(current)
                        while len(sent) > CHUNK_SIZE:
                            sp = sent.rfind(" ", 0, CHUNK_SIZE)
                            sp = sp if sp > 0 else CHUNK_SIZE
                            raw.append(sent[:sp])
                            sent = sent[sp:].lstrip()
                        current = sent
    if current:
        raw.append(current)
    if len(raw) <= 1:
        return raw
    # Apply overlap: prefix each chunk (after the first) with the tail of the previous chunk
    result = [raw[0]]
    for i in range(1, len(raw)):
        tail = result[-1][-CHUNK_OVERLAP:]
        result.append(tail + raw[i])
    return result


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


def _prepare_chunks(
    file_path: Path,
    proj: str,
    prefix: str,
    indexed_at: str,
    source_mtime: float | None,
) -> tuple[int, list[str], list[dict]]:
    """Read, parse, and chunk a single file. Returns (total_sections, chunk_texts, chunk_metas).

    Designed to run in a background thread so IO and parsing overlap with ONNX inference.
    """
    sections = read_file(file_path)
    total_sections = sections[0].total if sections else 0
    chunk_texts: list[str] = []
    chunk_metas: list[dict] = []
    for section in sections:
        if len(section.text) < MIN_CHARS:
            continue
        raw_chunks = [section.text] if section.atomic else chunk_text(section.text)
        for chunk_idx, chunk in enumerate(raw_chunks):
            doc_text = f"{section.breadcrumb}\n{chunk}" if section.breadcrumb else chunk
            chunk_texts.append(doc_text)
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
                    "links_to": ",".join(section.links_to),
                    "breadcrumb": section.breadcrumb,
                    "cross_refs": ",".join(section.cross_refs),
                }
            )
    return total_sections, chunk_texts, chunk_metas


def _data_index(
    paths: list[Path],
    force: bool = False,
    copy: bool = True,
    store: bool = True,
    project: str | None = None,
    description: str | None = None,
    auto_describe: bool = True,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_file_done: Callable[[str, str], None] | None = None,
    profile: bool = False,
) -> dict:
    """Index files. Returns {total_chunks, results: [{path, filename, status, ...}]}.

    Each result has status 'indexed' | 'skipped' | 'error'.
    on_progress(done, total, filename) is called after each chunk is embedded.
    on_file_done(filename, status) is called once per file with its terminal status.
    Raises RuntimeError on KeyboardInterrupt (cleaned up, no partial data).
    When profile=True, the returned dict includes a 'timings' key with per-phase seconds.
    """
    collection = get_collection()
    model = load_model()
    expanded = expand_paths(paths)

    # Snapshot which projects already have metadata before this run
    existing_project_names = set(read_projects().keys())

    results: list[dict] = []
    # Collect sample chunk texts per project for auto-description
    project_sample_chunks: dict[str, list[str]] = {}

    # Fetch all already-indexed source paths in one query instead of one per file
    _t_pre = time.perf_counter()
    if not force:
        _all_metas = collection.get(include=["metadatas"])["metadatas"] or []
        already_indexed: set[str] = {m["source"] for m in _all_metas if m.get("source")}
    else:
        already_indexed = set()
    t_chroma_query: float = time.perf_counter() - _t_pre

    # Phase timers (seconds — t_chroma_query initialised above with upfront query cost)
    t_embed: float = 0.0  # model.embed (ONNX inference)
    t_chroma_write: float = 0.0  # collection.add (storing vectors)
    # Note: read+parse+chunk runs in background and overlaps with embed, so we
    # measure it as the wall time the future takes beyond the embed phase.
    t_read_parse: float = 0.0  # fut.result() wait time (background read+chunk)

    _pending: tuple | None = None  # (Future, file_path, entry, proj, reindexed, store_action)

    def _process_pending() -> None:
        nonlocal _pending, t_read_parse, t_embed, t_chroma_write
        if _pending is None:
            return
        fut, fp, entry, proj, reindexed, store_action = _pending
        _pending = None

        _t0 = time.perf_counter()
        try:
            total_sections, chunk_texts, chunk_metas = fut.result()
        except Exception as exc:
            results.append({**entry, "status": "error", "reason": f"read error: {exc}"})
            if on_file_done:
                on_file_done(fp.name, "error")
            return
        t_read_parse += time.perf_counter() - _t0

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        prefix = doc_id_prefix(fp)

        try:
            embed_cfg = load_config().get("embedding", {})
            _t0 = time.perf_counter()
            embed_gen = model.embed(chunk_texts, batch_size=embed_cfg.get("batch_size", 64))
            for doc_text, meta, embedding in zip(chunk_texts, chunk_metas, embed_gen):
                ids.append(f"{prefix}_p{meta['page']}_c{meta['chunk']}")
                embeddings.append(embedding.tolist())
                documents.append(doc_text)
                metadatas.append(meta)
                if on_progress:
                    on_progress(len(ids), len(chunk_texts), fp.name)
            t_embed += time.perf_counter() - _t0
        except KeyboardInterrupt:
            partial = collection.get(where={"source": str(fp.resolve())})
            if partial["ids"]:
                collection.delete(ids=partial["ids"])
            raise RuntimeError(f"Interrupted while indexing {fp.name}. No partial data left.")

        if ids:
            batch = 500
            _t0 = time.perf_counter()
            for i in range(0, len(ids), batch):
                collection.add(
                    ids=ids[i : i + batch],
                    embeddings=embeddings[i : i + batch],
                    documents=documents[i : i + batch],
                    metadatas=metadatas[i : i + batch],
                )
            t_chroma_write += time.perf_counter() - _t0

        sections_indexed = len({m["page"] for m in metadatas})
        results.append(
            {
                **entry,
                "status": "indexed",
                "reason": None,
                "project": proj,
                "chunks_added": len(ids),
                "sections_indexed": sections_indexed,
                "total_sections": total_sections,
                "reindexed": reindexed,
                "store_action": store_action,
            }
        )
        if on_file_done:
            on_file_done(fp.name, "indexed")

        samples = project_sample_chunks.setdefault(proj, [])
        need = AUTO_DESCRIBE_SAMPLE_CHUNKS - len(samples)
        if need > 0 and chunk_texts:
            samples.extend(chunk_texts[:need])

    with ThreadPoolExecutor(max_workers=1) as _pool:
        for file_path in expanded:
            _process_pending()

            entry: dict = {"path": str(file_path), "filename": file_path.name}

            if not file_path.exists():
                results.append({**entry, "status": "skipped", "reason": "not found"})
                if on_file_done:
                    on_file_done(file_path.name, "skipped")
                continue

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                results.append(
                    {
                        **entry,
                        "status": "skipped",
                        "reason": f"unsupported type {file_path.suffix}",
                    }
                )
                if on_file_done:
                    on_file_done(file_path.name, "skipped")
                continue

            proj = project or default_project(file_path)
            prefix = doc_id_prefix(file_path)

            reindexed = False
            source_key = str(file_path.resolve())
            if not force and source_key in already_indexed:
                results.append(
                    {
                        **entry,
                        "status": "skipped",
                        "reason": "already indexed",
                    }
                )
                if on_file_done:
                    on_file_done(file_path.name, "skipped")
                continue
            if force or source_key in already_indexed:
                _t0 = time.perf_counter()
                collection.delete(where={"source": source_key})
                t_chroma_query += time.perf_counter() - _t0
                reindexed = True

            store_action: str | None = None
            if store:
                store_file(file_path, copy=copy)
                store_action = "copied" if copy else "linked"

            indexed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            try:
                source_mtime: float | None = file_path.stat().st_mtime
            except OSError:
                source_mtime = None

            # Submit read+parse+chunk to the background thread; embed runs next iteration
            fut = _pool.submit(_prepare_chunks, file_path, proj, prefix, indexed_at, source_mtime)
            _pending = (fut, file_path, entry, proj, reindexed, store_action)

        _process_pending()  # flush the final file

    # Free the embedding model to release GPU/CPU memory before the LLM autodescribe step
    model = None
    gc.collect()

    # ── Post-indexing: save user description and generate auto-description ──────
    indexed_projects = {r["project"] for r in results if r.get("status") == "indexed"}

    if description:
        # If a project was named explicitly, always update even if nothing new was indexed
        targets = {project} if project else indexed_projects
        for proj_name in targets:
            update_project_meta(proj_name, description=description)

    if auto_describe and project_sample_chunks:
        for proj_name, chunks in project_sample_chunks.items():
            is_new = proj_name not in existing_project_names
            if is_new or force:
                result_ad = _auto_describe(proj_name, chunks)
                if result_ad:
                    desc_ad, kws_ad = result_ad
                    update_project_meta(proj_name, auto_description=desc_ad, auto_keywords=kws_ad)

    timings = {
        "chroma_query_s": round(t_chroma_query, 3),
        "read_parse_s": round(t_read_parse, 3),
        "embed_s": round(t_embed, 3),
        "chroma_write_s": round(t_chroma_write, 3),
    }
    return {"total_chunks": collection.count(), "results": results, "timings": timings}


def _fetch_linked_results(
    query_embedding: list[float],
    primary_hits: list[dict],
    collection: chromadb.Collection,
    active_projects: list[str] | None,
) -> list[dict]:
    """Fetch the most query-relevant chunk from each file linked by the top-3 primary hits.

    Returns result dicts with 'linked_from' populated. Silently skips linked files that
    are not indexed or already appear in the primary results.
    """
    primary_filenames = {h["filename"] for h in primary_hits}

    # Collect linked filename → originating primary filenames (from top-3 only)
    link_sources: dict[str, list[str]] = {}
    for h in primary_hits[:3]:
        for name in h.get("links_to", "").split(","):
            name = name.strip()
            if name and name not in primary_filenames:
                link_sources.setdefault(name, []).append(h["filename"])

    if not link_sources:
        return []

    linked: list[dict] = []
    for filename, source_names in sorted(link_sources.items()):
        where = _build_where(active_projects, filename)
        if where is None:
            where = {"filename": filename}

        # Quick existence check before the more expensive query
        if not collection.get(where=where, include=[], limit=1)["ids"]:
            continue

        r = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        if not r["documents"][0]:
            continue

        text = r["documents"][0][0]
        meta = r["metadatas"][0][0]
        score = round(1.0 - r["distances"][0][0], 4)

        linked.append(
            {
                "filename": meta["filename"],
                "page": meta["page"],
                "page_label": meta.get("page_label", str(meta["page"])),
                "total_pages": meta.get("total_pages", 0),
                "chunk": meta.get("chunk", 0),
                "score": score,
                "text": text,
                "project": meta.get("project", ""),
                "source": meta.get("source", ""),
                "indexed_at": meta.get("indexed_at"),
                "source_mtime": meta.get("source_mtime"),
                "notable": _is_notable(text),
                "links_to": meta.get("links_to", ""),
                "linked_from": source_names,
                "breadcrumb": meta.get("breadcrumb", ""),
                "cross_references": [r for r in meta.get("cross_refs", "").split(",") if r],
            }
        )

    return linked


def _data_search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    filename: str | None = None,
    projects: list[str] | None = None,
    rerank: bool = False,
    follow_links: bool = False,
) -> list[dict]:
    """Semantic search. Returns list of result dicts with text and metadata.

    projects=None uses the active session; pass [] to search globally.
    When rerank=True, fetches top_k*3 candidates and re-scores with a cross-encoder.
    When follow_links=True, appends the best chunk from each file linked by top results.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    active_projects = projects if projects is not None else read_session()
    where = _build_where(active_projects, filename)

    model = load_model()
    query_embedding = next(model.query_embed([query])).tolist()

    n_candidates = top_k * 3 if (min_score > 0 or rerank) else top_k
    if where:
        matched_ids = collection.get(where=where, include=[])["ids"]
        matched_count = len(matched_ids)
        if matched_count == 0:
            return []
        n_candidates = min(n_candidates, matched_count)
    else:
        n_candidates = min(n_candidates, collection.count())

    query_kwargs: dict = dict(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
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
        if not rerank and shown > top_k:
            break
        output.append(
            {
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
                "notable": _is_notable(text),
                "links_to": meta.get("links_to", ""),
                "linked_from": [],
                "breadcrumb": meta.get("breadcrumb", ""),
                "cross_references": [r for r in meta.get("cross_refs", "").split(",") if r],
            }
        )

    if rerank and output:
        output = _rerank_results(query, output, top_k)

    if follow_links and output:
        output.extend(_fetch_linked_results(query_embedding, output, collection, active_projects))

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
            "breadcrumb": meta.get("breadcrumb", ""),
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
        targets = list(range(max(0, ref_idx - count), ref_idx)) + list(
            range(ref_idx + 1, min(ref_idx + 1 + count, len(sequence)))
        )

    result: list[dict] = []
    for seq_idx in targets:
        tp, tc, _ = sequence[seq_idx]
        r = col.get(
            where={"$and": [{"filename": filename}, {"page": tp}, {"chunk": tc}]},
            include=["documents", "metadatas"],
        )
        for meta, text in zip(r.get("metadatas") or [], r.get("documents") or []):
            result.append(
                {
                    "filename": filename,
                    "page": meta.get("page", tp),
                    "page_label": meta.get("page_label", str(tp)),
                    "chunk_idx": meta.get("chunk", tc),
                    "text": text,
                    "source": meta.get("source", ""),
                }
            )

    return result


def _data_list_projects() -> list[dict]:
    """Return per-project stats with descriptions and keywords. Returns [] when empty."""
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

    stored = read_projects()
    return [
        {
            "name": name,
            "file_count": len(info["files"]),
            "chunk_count": info["chunks"],
            "description": stored.get(name, {}).get("description", ""),
            "auto_description": stored.get(name, {}).get("auto_description", ""),
            "keywords": stored.get(name, {}).get("keywords", []),
            "auto_keywords": stored.get(name, {}).get("auto_keywords", []),
        }
        for name, info in sorted(projects.items())
    ]


def _data_project_describe(
    name: str,
    description: str | None = None,
    keywords: list[str] | None = None,
) -> dict:
    """Set user-provided description and/or keywords for a project."""
    known = {p["name"] for p in _data_list_projects()}
    if name not in known:
        raise ValueError(f"Project not found: {name!r}")
    kwargs: dict = {}
    if description is not None:
        kwargs["description"] = description
    if keywords is not None:
        kwargs["keywords"] = keywords
    if kwargs:
        update_project_meta(name, **kwargs)
    return {"name": name, **kwargs}


def _data_project_autodescribe(name: str) -> dict:
    """Generate (or regenerate) auto_description and auto_keywords for an existing project."""
    collection = get_collection()
    existing = collection.get(where={"project": name}, include=["documents"])
    if not existing["ids"]:
        raise ValueError(f"Project not found: {name!r}")

    chunks = existing["documents"][:AUTO_DESCRIBE_SAMPLE_CHUNKS]
    result = _auto_describe(name, chunks)
    if result is None:
        return {"name": name, "error": "Auto-description failed or is disabled (check config)"}

    desc, kws = result
    update_project_meta(name, auto_description=desc, auto_keywords=kws)
    return {"name": name, "auto_description": desc, "auto_keywords": kws}


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
            "total_chunks": 0,
            "files": 0,
            "projects": 0,
            "model": MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "last_indexed": "unknown",
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


def _data_list_members(
    filename: str,
    projects: list[str] | None = None,
    section_filter: str | None = None,
) -> list[dict]:
    """Return all indexed sections for a file, grouped by section heading (page_label).

    Useful for structured browsing or absence-checking without a similarity query.
    projects=None uses the active session; [] searches globally.
    section_filter: case-insensitive substring match on section label.

    Returns list of {section, page, chunks: [{chunk_idx, text, breadcrumb}]}.
    """
    from collections import defaultdict

    col = get_collection()
    if col.count() == 0:
        return []

    active_projects = projects if projects is not None else read_session()
    where = _build_where(active_projects, filename)
    if where is None:
        where = {"filename": filename}

    result = col.get(where=where, include=["documents", "metadatas"])
    docs: list[str] = result.get("documents") or []
    metas: list[dict] = result.get("metadatas") or []
    if not docs:
        return []

    sections: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for doc, meta in sorted(
        zip(docs, metas), key=lambda x: (x[1].get("page", 0), x[1].get("chunk", 0))
    ):
        label = meta.get("page_label", str(meta.get("page", 0)))
        if section_filter and section_filter.lower() not in label.lower():
            continue
        key = (meta.get("page", 0), label)
        sections[key].append(
            {
                "chunk_idx": meta.get("chunk", 0),
                "text": doc,
                "breadcrumb": meta.get("breadcrumb", ""),
            }
        )

    return [
        {"section": label, "page": page, "chunks": chunks}
        for (page, label), chunks in sorted(sections.items())
    ]


EXPORT_FORMAT_VERSION = 2
_EXPORT_BATCH = 500  # max ids per collection.add() call


def _data_export(project: str) -> dict:
    """Return all data needed to archive a project.

    Returns {project, chunk_count, chunks: [{id, embedding, document, metadata}],
             file_paths: [Path]}
    """
    collection = get_collection()
    result = collection.get(
        where={"project": project},
        include=["embeddings", "documents", "metadatas"],
    )
    ids: list[str] = result["ids"]
    if not ids:
        raise ValueError(f"Project {project!r} not found or empty.")

    embeddings: list = result["embeddings"]
    documents: list[str] = result["documents"]
    metadatas: list[dict] = result["metadatas"]

    chunks = [
        {"id": i, "embedding": e, "document": d, "metadata": m}
        for i, e, d, m in zip(ids, embeddings, documents, metadatas)
    ]

    # Collect stored file paths (may be symlinks)
    seen_files: set[str] = set()
    file_paths: list[Path] = []
    for m in metadatas:
        fname = m.get("filename", "")
        if fname and fname not in seen_files:
            seen_files.add(fname)
            candidate = FILES_DIR / fname
            if candidate.exists():
                file_paths.append(candidate)

    project_meta = read_projects().get(project, {})

    return {
        "project": project,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "file_paths": file_paths,
        "project_meta": project_meta,
    }


def _data_import(archive_path: Path, overwrite: bool = False) -> dict:
    """Import a project from a tar.gz archive produced by cmd_export.

    Returns {project, chunks_added, files_restored}.
    Raises ValueError on incompatible model or duplicate project (without overwrite).
    """
    import json as _json
    import tarfile

    with tarfile.open(archive_path, "r:gz") as tf:
        # Read manifest
        manifest_member = tf.getmember("manifest.json")
        manifest: dict = _json.load(tf.extractfile(manifest_member))  # type: ignore[arg-type]

        if manifest.get("engra_export_version") != EXPORT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported export version {manifest.get('engra_export_version')!r}."
            )
        if manifest.get("model") != MODEL_NAME:
            raise ValueError(
                f"Model mismatch: archive uses {manifest['model']!r}, "
                f"this install uses {MODEL_NAME!r}. Embeddings are incompatible."
            )

        project: str = manifest["project"]
        collection = get_collection()

        if not overwrite:
            existing = collection.get(where={"project": project}, include=[])
            if existing["ids"]:
                raise ValueError(
                    f"Project {project!r} already exists. Use --overwrite to replace it."
                )
        else:
            collection.delete(where={"project": project})

        # Read chunks
        chunks_member = tf.getmember("chunks.json")
        chunks: list[dict] = _json.load(tf.extractfile(chunks_member))  # type: ignore[arg-type]

        # Add in batches
        for start in range(0, len(chunks), _EXPORT_BATCH):
            batch = chunks[start : start + _EXPORT_BATCH]
            collection.add(
                ids=[c["id"] for c in batch],
                embeddings=[c["embedding"] for c in batch],
                documents=[c["document"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )

        # Restore project metadata (description, keywords, etc.)
        project_meta: dict = manifest.get("project_meta", {})
        if project_meta:
            update_project_meta(project, **project_meta)

        # Restore files
        ensure_dirs()
        files_restored = 0
        for member in tf.getmembers():
            if member.name.startswith("files/") and not member.isdir():
                fname = Path(member.name).name
                dest = FILES_DIR / fname
                raw = tf.extractfile(member)
                if raw is not None:
                    dest.write_bytes(raw.read())
                    files_restored += 1

    return {"project": project, "chunks_added": len(chunks), "files_restored": files_restored}


# ── Commands ──────────────────────────────────────────────────────────────────


def cmd_index(
    paths: list[Path],
    force: bool = False,
    copy: bool | None = None,
    store: bool = True,
    project: str | None = None,
    description: str | None = None,
    auto_describe: bool = True,
    check: bool = False,
    profile: bool = False,
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

    n_total = len(expanded)
    is_multi = n_total > 1

    t0 = time.monotonic()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        if is_multi:
            # ── Directory mode: one overall file-count bar + one current-file chunk bar ──
            files_done_count = [0]
            overall_task = progress.add_task(
                f"Files  [bold]0 / {n_total}[/bold]",
                total=n_total,
            )
            current_task = progress.add_task("", total=None, visible=False)
            current_file: list[str | None] = [None]

            def on_progress(done: int, total: int, filename: str) -> None:
                if filename != current_file[0]:
                    current_file[0] = filename
                    progress.update(
                        current_task,
                        visible=True,
                        description=f"  [dim]↳[/dim] [cyan]{filename}[/cyan]",
                        completed=0,
                        total=total,
                    )
                progress.update(current_task, completed=done)

            def on_file_done(filename: str, status: str) -> None:
                files_done_count[0] += 1
                n = files_done_count[0]
                progress.update(
                    overall_task,
                    completed=n,
                    description=f"Files  [bold]{n} / {n_total}[/bold]",
                )
                if status != "indexed":
                    progress.update(current_task, visible=False)
                current_file[0] = None

        else:
            # ── Single-file mode: per-chunk bar (unchanged behaviour) ──────────────
            single_tasks: dict[str, int] = {}

            def on_progress(done: int, total: int, filename: str) -> None:
                if filename not in single_tasks:
                    single_tasks[filename] = progress.add_task(
                        f"Embedding [cyan]{filename}[/cyan]", total=total
                    )
                progress.update(single_tasks[filename], completed=done, total=total)

            on_file_done = None  # type: ignore[assignment]

        try:
            result = _data_index(
                paths,
                force=force,
                copy=copy,
                store=store,
                project=project,
                description=description,
                auto_describe=auto_describe,
                on_progress=on_progress,
                on_file_done=on_file_done if is_multi else None,
                profile=profile,
            )
        except RuntimeError as exc:
            console.print(f"\n[red][interrupted][/red] {exc}")
            raise SystemExit(1)

    # ── Post-progress output ──────────────────────────────────────────────────

    if is_multi:
        # Aggregate by project; show one summary line per project
        from collections import defaultdict

        proj_stats: dict[str, dict] = defaultdict(lambda: {"indexed": 0, "chunks": 0})
        error_results: list[dict] = []
        n_skipped = 0

        for r in result["results"]:
            if r["status"] == "indexed":
                proj_stats[r["project"]]["indexed"] += 1
                proj_stats[r["project"]]["chunks"] += r["chunks_added"]
            elif r["status"] == "skipped":
                n_skipped += 1
            elif r["status"] == "error":
                error_results.append(r)

        for proj, stats in sorted(proj_stats.items()):
            console.print(
                f"  → [green]{stats['indexed']} file(s)[/green], "
                f"[green]{stats['chunks']} chunks[/green] added "
                f"→ project [cyan]{proj}[/cyan]"
            )
        if n_skipped:
            console.print(
                f"  [yellow]{n_skipped} file(s) skipped[/yellow]"
                f" (already indexed — use [bold]--force[/bold] to re-index)"
            )
        for r in error_results:
            console.print(f"  [red][error][/red] {r['filename']}: {r['reason']}")

    else:
        # Single-file: keep the existing verbose per-result output
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
                skipped_secs = r["total_sections"] - r["sections_indexed"]
                skip_note = (
                    f", skipped {skipped_secs} sections shorter than {MIN_CHARS} chars"
                    if skipped_secs > 0
                    else ""
                )
                console.print(
                    f"  → [green]{r['chunks_added']} chunks[/green] from "
                    f"{r['sections_indexed']}/{r['total_sections']} sections{skip_note} "
                    f"→ project [cyan]{r['project']}[/cyan]"
                )

    elapsed = time.monotonic() - t0
    elapsed_str = (
        f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed) // 60}m {int(elapsed) % 60}s"
    )
    console.print(
        f"\nKnowledge base: [bold]{result['total_chunks']} chunks[/bold] total"
        f"  [dim]({elapsed_str})[/dim]"
    )

    if profile:
        t = result["timings"]
        total_tracked = t["chroma_query_s"] + t["read_parse_s"] + t["embed_s"] + t["chroma_write_s"]
        console.print("\n[bold]Timing breakdown[/bold]")
        rows = [
            ("ChromaDB existence checks", t["chroma_query_s"]),
            ("Read + parse + chunk (background)", t["read_parse_s"]),
            ("ONNX embedding inference", t["embed_s"]),
            ("ChromaDB write (collection.add)", t["chroma_write_s"]),
        ]
        for label, secs in rows:
            pct = secs / total_tracked * 100 if total_tracked > 0 else 0
            bar = "█" * int(pct / 2)
            console.print(f"  {label:<38} {secs:6.1f}s  {pct:4.0f}%  {bar}")


def cmd_search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    filename: str | None = None,
    projects: list[str] | None = None,
    full: bool = False,
    output_format: str = "text",
    rerank: bool = False,
    follow_links: bool = False,
) -> None:
    if get_collection().count() == 0:
        console.print(
            "[yellow]Knowledge base is empty.[/yellow] Run: [bold]engra index <file.pdf>[/bold]"
        )
        return

    active_projects = projects if projects is not None else read_session()
    if active_projects:
        console.print(f"[dim]Searching in project(s): {', '.join(active_projects)}[/dim]")

    hits = _data_search(
        query,
        top_k=top_k,
        min_score=min_score,
        filename=filename,
        projects=projects,
        rerank=rerank,
        follow_links=follow_links,
    )

    if output_format == "json":
        import json as _json

        print(_json.dumps(hits, default=str))
        return

    _warn_stale_from_metas(
        [
            {
                "source": h["source"],
                "indexed_at": h.get("indexed_at"),
                "source_mtime": h.get("source_mtime"),
            }
            for h in hits
        ]
    )
    console.print(f'\nResults for: [bold]"{query}"[/bold]')
    console.rule()

    primary_hits = [h for h in hits if not h.get("linked_from")]
    linked_hits = [h for h in hits if h.get("linked_from")]

    result_projects = {h["project"] for h in primary_hits}
    show_project = len(active_projects) != 1 and len(result_projects) > 1

    use_rerank = rerank and primary_hits and primary_hits[0].get("rerank_score") is not None
    primary_scores = [h["rerank_score"] if use_rerank else h["score"] for h in primary_hits]
    rel_scores = _normalize_scores(primary_scores)

    for i, (h, rel, primary) in enumerate(zip(primary_hits, rel_scores, primary_scores), 1):
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

        if use_rerank:
            score_display = f"rerank: {rel:.2f}  vec: {h['score']:.3f}"
        else:
            score_display = f"score: {rel:.2f}"

        notable_marker = "  [yellow bold][!][/yellow bold]" if h.get("notable") else ""
        console.print(
            f"[bold cyan][{i}][/bold cyan] {file_label}  —  "
            f"p.{page_str}/{h['total_pages']}{chunk_info}{phys_info}  "
            f"[dim]({score_display})[/dim]{notable_marker}"
        )
        console.print(f"    [dim]{snippet}[/dim]")
        console.print()

    if primary_hits:
        footer = (
            "Scores normalised within result set · rerank scores from cross-encoder"
            if use_rerank
            else "Scores normalised within result set · --format json for raw similarity"
        )
        console.print(f"[dim]{footer}[/dim]")
        console.print()

    if linked_hits:
        console.rule("[dim]Related (linked from results above)[/dim]")
        for h in linked_hits:
            page_str = h["page_label"] if h["page_label"] != str(h["page"]) else str(h["page"])
            linked_from_str = ", ".join(h["linked_from"])
            notable_marker = "  [yellow bold][!][/yellow bold]" if h.get("notable") else ""
            text = h["text"]
            normalized = " ".join(text.split())
            snippet = normalized[:220] + ("…" if len(normalized) > 220 else "")
            console.print(
                f"  [cyan]→[/cyan] {h['filename']}  —  p.{page_str}/{h['total_pages']}  "
                f"[dim](score: {h['score']:.3f}  linked from: {linked_from_str})[/dim]"
                f"{notable_marker}"
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

    if get_collection().count() == 0:
        console.print(
            "[yellow]Knowledge base is empty.[/yellow] Run: [bold]engra index <file>[/bold]"
        )
        return

    hits = _data_search(question, top_k=top_k, projects=projects, filename=filename)

    if not hits:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    # Build context block
    context_parts = []
    for i, h in enumerate(hits, 1):
        label = f"[{i}] {h['filename']} p.{h['page_label']}"
        context_parts.append(f"{label}\n{h['text'].strip()}")
    context_text = "\n\n---\n\n".join(context_parts)

    # Print sources header
    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print(f"[dim]Context: {len(hits)} chunk(s) from index[/dim]")
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
            "stream": True,
        }
    ).encode()

    req = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )

    console.print(f"\n[bold green]Answer[/bold green] [dim](model: {model_id})[/dim]\n")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        console.print(delta, end="")
                except (_json.JSONDecodeError, KeyError, IndexError):
                    continue
    except urllib.error.URLError as exc:
        console.print(f"\n[red]LLM request failed:[/red] {exc}")
        console.print(
            f"[dim]Ensure your LLM server is running at [bold]{api_base}[/bold] "
            "or configure [bold]ask.api_base[/bold] in ~/.config/engra/config.toml[/dim]"
        )
        return

    console.print("\n")
    console.rule("Sources")
    for i, h in enumerate(hits, 1):
        console.print(
            f"  [{i}] {h['filename']}  p.{h['page_label']}  [dim](chunk {h['chunk']})[/dim]"
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


def _print_nav_hint(
    filename: str, sequence: list[tuple[int, int, str]], first_idx: int, last_idx: int
) -> None:
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
    file_metas_list: list[dict] = (
        col.get(where={"filename": filename}, include=["metadatas"]).get("metadatas") or []
    )
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
                console.print(
                    f"[yellow]Chunk not found:[/yellow] {filename!r} page {ref_page} chunk {chunk}"
                )
                return

        direction = "next" if next_k is not None else "prev"
        count = next_k if next_k is not None else (prev_k or 1)
        chunks = _data_get_neighbors(
            filename, ref_page, ref_chunk, direction=direction, count=count
        )

        if not chunks:
            edge = "end" if next_k is not None else "beginning"
            console.print(f"[yellow]Already at the {edge} of {filename!r}[/yellow]")
            return

        for c in chunks:
            _print_chunk(
                filename,
                {
                    "page": c["page"],
                    "page_label": c["page_label"],
                    "chunk": c["chunk_idx"],
                    "source": c["source"],
                },
                c["text"],
            )

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
        _print_chunk(
            filename,
            {
                "page": c["page"],
                "page_label": c["page_label"],
                "chunk": c["chunk_idx"],
                "source": c["source"],
            },
            c["text"],
        )

    found_pages = {c["page"] for c in chunks}
    missing_pages = [p for p in range(page_start, page_end + 1) if p not in found_pages]
    if missing_pages:
        console.print(
            f"[yellow]pages {_format_missing_pages(missing_pages)} not found in index[/yellow]"
        )

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
            f["project"],
            f["filename"],
            str(f["chunks"]),
            str(f["pages"]),
            stale_cells[f["stale_status"]],
            f["source"],
        )

    console.print(table)
    console.print(f"Total chunks in index: [bold]{sum(f['chunks'] for f in file_list)}[/bold]")


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
    BOOKMARKS_PATH.write_text(
        _json.dumps(bookmarks, indent=2, ensure_ascii=False), encoding="utf-8"
    )


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
    table.add_column("Description", max_width=45, overflow="fold")
    table.add_column("Keywords", max_width=35, overflow="fold")

    for p in projects:
        is_active = p["name"] in active
        # Prefer user-set values, fall back to AI-generated
        desc = p["description"] or p["auto_description"]
        kws = p["keywords"] or p["auto_keywords"]
        kw_str = ", ".join(kws) if kws else ""
        table.add_row(
            p["name"],
            str(p["file_count"]),
            str(p["chunk_count"]),
            "[green]● yes[/green]" if is_active else "[dim]no[/dim]",
            f"[dim]{desc}[/dim]" if desc else "",
            f"[dim]{kw_str}[/dim]" if kw_str else "",
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

    rename_project_meta(old_name, new_name)

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

    remove_project_meta(name)

    # Remove from session if active
    active = read_session()
    if name in active:
        updated = [p for p in active if p != name]
        write_session(updated) if updated else clear_session()
        console.print("[dim]Removed from active session.[/dim]")


def cmd_project_describe(
    name: str,
    description: str | None = None,
    keywords: list[str] | None = None,
) -> None:
    try:
        _data_project_describe(name, description=description, keywords=keywords)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    if description is not None:
        console.print(f"[green]Description set[/green] for '{name}'.")
    if keywords is not None:
        console.print(f"[green]Keywords set[/green] for '{name}': {', '.join(keywords)}")


def cmd_project_autodescribe(name: str) -> None:
    console.print(f"[dim]Generating auto-description for '{name}'...[/dim]")
    try:
        result = _data_project_autodescribe(name)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    if "error" in result:
        console.print(f"[yellow]{result['error']}[/yellow]")
        return
    console.print(f"[green]Auto-description:[/green] {result['auto_description']}")
    if result["auto_keywords"]:
        console.print(f"[green]Keywords:[/green] {', '.join(result['auto_keywords'])}")


def cmd_export(project: str, output_path: Path | None = None) -> None:
    """Export a project to a tar.gz archive with embeddings and source files."""
    import datetime
    import json as _json
    import tarfile

    try:
        data = _data_export(project)
    except ValueError as exc:
        console.print(f"[red]Export failed:[/red] {exc}")
        raise SystemExit(1)

    if output_path is None:
        safe = project.replace("/", "_").replace(" ", "_")
        output_path = Path(f"{safe}.engra.tar.gz")

    manifest = {
        "engra_export_version": EXPORT_FORMAT_VERSION,
        "model": MODEL_NAME,
        "project": project,
        "exported_at": datetime.datetime.now().isoformat(),
        "chunk_count": data["chunk_count"],
        "file_count": len(data["file_paths"]),
        "project_meta": data["project_meta"],
    }

    with tarfile.open(output_path, "w:gz") as tf:
        # manifest.json
        manifest_bytes = _json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_bytes)
        import io

        tf.addfile(info, io.BytesIO(manifest_bytes))

        # chunks.json – convert numpy arrays to plain lists so JSON round-trips cleanly
        for chunk in data["chunks"]:
            e = chunk["embedding"]
            if hasattr(e, "tolist"):
                chunk["embedding"] = e.tolist()
        chunks_bytes = _json.dumps(data["chunks"]).encode()
        info = tarfile.TarInfo("chunks.json")
        info.size = len(chunks_bytes)
        tf.addfile(info, io.BytesIO(chunks_bytes))

        # files/
        for file_path in data["file_paths"]:
            real = file_path.resolve()
            if real.exists():
                tf.add(real, arcname=f"files/{file_path.name}")

    console.print(
        f"[green]Exported[/green] project [bold]{project}[/bold] → [bold]{output_path}[/bold]"
    )
    console.print(f"  {data['chunk_count']} chunks, {len(data['file_paths'])} file(s)")


def cmd_import(archive_path: Path, overwrite: bool = False) -> None:
    """Import a project from a tar.gz archive."""
    try:
        result = _data_import(archive_path, overwrite=overwrite)
    except (ValueError, KeyError) as exc:
        console.print(f"[red]Import failed:[/red] {exc}")
        raise SystemExit(1)

    console.print(
        f"[green]Imported[/green] project [bold]{result['project']}[/bold] "
        f"from [bold]{archive_path.name}[/bold]"
    )
    console.print(f"  {result['chunks_added']} chunks, {result['files_restored']} file(s) restored")


def cmd_import_soft(source_dir: Path, project: str | None = None) -> None:
    """Index a directory with symlinks (soft import — files stay in place)."""
    if not source_dir.is_dir():
        console.print(f"[red]Not a directory:[/red] {source_dir}")
        raise SystemExit(1)
    effective_project = project or source_dir.resolve().name
    console.print(
        f"[dim]Soft-importing [bold]{source_dir}[/bold] "
        f"as project [bold]{effective_project}[/bold] (symlinking files)[/dim]"
    )
    cmd_index([source_dir], copy=False, store=True, project=effective_project)


_ORT_GPU_INDEX = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"


def cmd_setup_gpu() -> None:
    """Install the correct onnxruntime-gpu wheel for CUDA 12 GPU inference.

    fastembed-gpu pulls in the lightweight add-on onnxruntime-gpu from PyPI, which
    only provides CUDA provider .so files. This command replaces it with the full
    standalone GPU wheel (252 MB) that includes Python bindings + CUDA support.

    Run once after every: pipx install "engra[gpu]" --force
    """
    import subprocess

    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            console.print("[green]CUDA already available[/green] — nothing to do.")
            return
    except Exception:
        pass

    console.print("Installing full onnxruntime-gpu wheel for CUDA 12…")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "onnxruntime-gpu",
            "--no-deps",
            "--force-reinstall",
            "--extra-index-url",
            _ORT_GPU_INDEX,
        ],
        check=False,
    )
    if result.returncode != 0:
        console.print("[red]Installation failed.[/red] Check the output above.")
        raise SystemExit(1)

    # Verify
    import importlib

    import onnxruntime as ort  # noqa: PLC0415

    importlib.reload(ort)
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        console.print("[green]Done.[/green] CUDAExecutionProvider is now available.")
    else:
        console.print(
            "[yellow]Installed, but CUDAExecutionProvider still not detected.[/yellow]\n"
            "Check that cuDNN 9 is installed: [bold]sudo apt install libcudnn9-cuda-12[/bold]"
        )


def cmd_mcp() -> None:
    """Start the MCP stdio server. Requires the 'mcp' optional dependency."""
    try:
        from engra.mcp_server import run_mcp_server  # noqa: PLC0415
    except ImportError as exc:
        console.print(f"[red]MCP support not installed:[/red] {exc}")
        console.print("Install with: [bold]pip install 'engra[mcp]'[/bold]")
        raise SystemExit(1)
    run_mcp_server()

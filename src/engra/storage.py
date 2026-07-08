import json
import logging
import shutil
import sqlite3
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir

logger = logging.getLogger(__name__)

DATA_DIR = Path(user_data_dir("engra", appauthor=False))
CACHE_DIR = Path(user_cache_dir("engra", appauthor=False))
FILES_DIR = DATA_DIR / "files"
DB_DIR = DATA_DIR / "db"
FTS_DB_PATH = DATA_DIR / "fts.db"
STATE_FILE = DATA_DIR / "state.toml"
PROJECTS_FILE = DATA_DIR / "projects.json"

SESSION_TTL_HOURS = 8


def ensure_dirs() -> None:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)


def stored_name(doc_id: str, suffix: str) -> str:
    """Return the on-disk filename for a document, scoped by its unique doc_id.

    Two files that share a basename in different projects resolve to different
    doc_ids, so their stored copies never collide.
    """
    return f"{doc_id}{suffix}"


def store_file(pdf_path: Path, doc_id: str, copy: bool = True) -> Path:
    """Copy or symlink a file into the engra files directory. Returns stored path.

    The destination is scoped by *doc_id* (unique per resolved source path) so
    two files that share a basename in different projects never overwrite each
    other on disk.
    """
    ensure_dirs()
    dest = FILES_DIR / stored_name(doc_id, pdf_path.suffix)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy:
        shutil.copy2(pdf_path, dest)
    else:
        dest.symlink_to(pdf_path.resolve())
    return dest


def remove_file(doc_id: str, suffix: str) -> None:
    """Remove the stored copy or symlink for a document (doc_id-scoped name)."""
    dest = FILES_DIR / stored_name(doc_id, suffix)
    if dest.exists() or dest.is_symlink():
        dest.unlink()


# ── Session ───────────────────────────────────────────────────────────────────


def read_session() -> list[str]:
    """Return active project names. Returns [] if none set or session expired."""
    if not STATE_FILE.exists():
        return []
    try:
        with open(STATE_FILE, "rb") as f:
            state = tomllib.load(f)
        activated_at = datetime.fromisoformat(state["session"]["activated_at"])
        if datetime.now() - activated_at > timedelta(hours=SESSION_TTL_HOURS):
            STATE_FILE.unlink()
            return []
        return state["session"].get("active_projects", [])
    except Exception:
        return []


def write_session(projects: list[str]) -> None:
    ensure_dirs()
    content = (
        "[session]\n"
        f"active_projects = {json.dumps(projects)}\n"
        f'activated_at = "{datetime.now().isoformat()}"\n'
    )
    STATE_FILE.write_text(content)


def clear_session() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


# ── Project metadata ───────────────────────────────────────────────────────────


def read_projects() -> dict[str, dict]:
    """Return {project_name: {description, auto_description, keywords, auto_keywords}}."""
    if not PROJECTS_FILE.exists():
        return {}
    try:
        return json.loads(PROJECTS_FILE.read_text())
    except Exception:
        return {}


def write_projects(data: dict[str, dict]) -> None:
    ensure_dirs()
    PROJECTS_FILE.write_text(json.dumps(data, indent=2))


def update_project_meta(name: str, **kwargs) -> None:
    """Merge non-None kwargs into the named project's metadata entry."""
    data = read_projects()
    entry = data.setdefault(name, {})
    for k, v in kwargs.items():
        if v is not None:
            entry[k] = v
    write_projects(data)


def rename_project_meta(old: str, new: str) -> None:
    data = read_projects()
    if old in data:
        data[new] = data.pop(old)
        write_projects(data)


def remove_project_meta(name: str) -> None:
    data = read_projects()
    if name in data:
        data.pop(name)
        write_projects(data)


# ── Keyword (FTS5) index ───────────────────────────────────────────────────────
#
# A SQLite FTS5 virtual table kept in lockstep with the chromadb collection so
# exact-token queries (part numbers, error/PGN codes, C symbols, section ids)
# that dense embeddings miss can still be retrieved and fused via RRF.
#
# FTS5 is a compile-time-optional SQLite module and is not guaranteed on every
# Python build. If it is unavailable we warn once and degrade to a no-op so all
# keyword/hybrid retrieval simply falls back to dense-only (mirrors the GPU
# provider fallback in commands.load_model()).

_fts_available: bool | None = None  # None = untested, True/False after first probe
_fts_warned = False

_FTS_CREATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    chunk_id UNINDEXED, text, doc_id UNINDEXED, project UNINDEXED, filename UNINDEXED,
    tokenize = 'porter unicode61'
)
"""


def get_fts_connection() -> sqlite3.Connection | None:
    """Open (creating if needed) the FTS5 keyword index.

    Returns an open connection, or None if FTS5 is unavailable on this SQLite
    build (in which case a warning is logged once and callers degrade to no-ops).
    Callers own the returned connection and must close it.
    """
    global _fts_available, _fts_warned
    if _fts_available is False:
        return None
    ensure_dirs()
    try:
        conn = sqlite3.connect(str(FTS_DB_PATH))
        conn.execute(_FTS_CREATE)
        conn.commit()
    except sqlite3.OperationalError as exc:
        _fts_available = False
        if not _fts_warned:
            logger.warning(
                "SQLite FTS5 unavailable (%s); keyword/hybrid search disabled, "
                "falling back to dense-only retrieval.",
                exc,
            )
            _fts_warned = True
        return None
    _fts_available = True
    return conn


def fts_add(rows: list[tuple]) -> None:
    """Insert chunk rows into the keyword index.

    Each row is a (chunk_id, text, doc_id, project, filename) tuple. No-op when
    FTS5 is unavailable or *rows* is empty.
    """
    rows = list(rows)
    if not rows:
        return
    conn = get_fts_connection()
    if conn is None:
        return
    try:
        conn.executemany(
            "INSERT INTO chunks (chunk_id, text, doc_id, project, filename) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def fts_delete_by_ids(ids: list[str]) -> None:
    """Delete keyword-index rows for the given chunk ids."""
    ids = list(ids)
    if not ids:
        return
    conn = get_fts_connection()
    if conn is None:
        return
    try:
        conn.executemany("DELETE FROM chunks WHERE chunk_id = ?", [(i,) for i in ids])
        conn.commit()
    finally:
        conn.close()


def fts_delete_by_doc_id(doc_id: str) -> None:
    """Delete every keyword-index row belonging to a document."""
    conn = get_fts_connection()
    if conn is None:
        return
    try:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.commit()
    finally:
        conn.close()


def fts_delete_by_project(project: str) -> None:
    """Delete every keyword-index row belonging to a project."""
    conn = get_fts_connection()
    if conn is None:
        return
    try:
        conn.execute("DELETE FROM chunks WHERE project = ?", (project,))
        conn.commit()
    finally:
        conn.close()


def fts_update_project(chunk_ids: list[str], new_project: str) -> None:
    """Re-point the given chunk ids at *new_project* (mirrors a project rename)."""
    chunk_ids = list(chunk_ids)
    if not chunk_ids:
        return
    conn = get_fts_connection()
    if conn is None:
        return
    try:
        conn.executemany(
            "UPDATE chunks SET project = ? WHERE chunk_id = ?",
            [(new_project, i) for i in chunk_ids],
        )
        conn.commit()
    finally:
        conn.close()


def _fts_match_query(query: str) -> str:
    """Build a robust FTS5 MATCH expression from an arbitrary user query.

    Each whitespace-separated term is wrapped in a double-quoted phrase (with
    embedded quotes doubled) and OR-ed together. Quoting neutralises FTS5 syntax
    characters (``-``, ``:``, ``()``, ``*`` …) that appear in codes/part numbers,
    and OR maximises recall for the keyword arm feeding rank fusion.
    """
    quoted = []
    for term in query.split():
        term = term.replace('"', '""')
        if term:
            quoted.append(f'"{term}"')
    return " OR ".join(quoted)


def fts_search(query: str, where: str | None = None, limit: int = 15) -> list[dict]:
    """Run a BM25 keyword search, best match first.

    *where* is an optional pre-built SQL fragment constraining the UNINDEXED
    doc_id/project/filename columns. Returns a list of
    {chunk_id, text, doc_id, project, filename, score} dicts (score is SQLite's
    bm25 value: lower = better). Returns [] when FTS5 is unavailable, the query
    is empty, or the MATCH expression is rejected.
    """
    match = _fts_match_query(query)
    if not match:
        return []
    conn = get_fts_connection()
    if conn is None:
        return []
    try:
        sql = (
            "SELECT chunk_id, text, doc_id, project, filename, bm25(chunks) AS score "
            "FROM chunks WHERE chunks MATCH ?"
        )
        params: list = [match]
        if where:
            sql += f" AND ({where})"
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 query failed (%s); returning no keyword results.", exc)
            return []
    finally:
        conn.close()
    return [
        {
            "chunk_id": r[0],
            "text": r[1],
            "doc_id": r[2],
            "project": r[3],
            "filename": r[4],
            "score": r[5],
        }
        for r in rows
    ]

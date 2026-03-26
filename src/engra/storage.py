import json
import shutil
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir

DATA_DIR = Path(user_data_dir("engra", appauthor=False))
CACHE_DIR = Path(user_cache_dir("engra", appauthor=False))
FILES_DIR = DATA_DIR / "files"
DB_DIR = DATA_DIR / "db"
STATE_FILE = DATA_DIR / "state.toml"
PROJECTS_FILE = DATA_DIR / "projects.json"

SESSION_TTL_HOURS = 8


def ensure_dirs() -> None:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)


def store_file(pdf_path: Path, copy: bool = True) -> Path:
    """Copy or symlink a PDF into the engra files directory. Returns stored path."""
    ensure_dirs()
    dest = FILES_DIR / pdf_path.name
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy:
        shutil.copy2(pdf_path, dest)
    else:
        dest.symlink_to(pdf_path.resolve())
    return dest


def remove_file(filename: str) -> None:
    """Remove the stored copy or symlink for a file."""
    dest = FILES_DIR / filename
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
